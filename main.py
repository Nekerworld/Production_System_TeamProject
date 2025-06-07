import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, GRU, BatchNormalization
from tensorflow.keras.utils import plot_model
import streamlit as st
import pandas as pd
import os
from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns

# 1. csv 파일 불러오기 (error lot list csv는 제외)
five_process_180sec = r'C:\\YS\\TUK\\S4E1\\생산시스템구축실무\\TeamProject\\Production_System_TeamProject\\data\\장비이상 조기탐지\\5공정_180sec'
all_csv_files = glob(os.path.join(five_process_180sec, '*.csv'))
csv_files = [f for f in all_csv_files if 'Error Lot list' not in os.path.basename(f)]

# 2. error lot list 불러오기
error_df = pd.read_csv(os.path.join(five_process_180sec, 'Error Lot list.csv'))

# 3. 각 데이터 파일을 보고 에러 csv에서 맞는 날짜를 찾은 후, 에러 csv를 참고해서 process를 모두 anomaly로 표기
dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    
    # 4. 한글 오전/오후를 AM/PM으로 변환
    df['Time'] = (
        df['Time']
        .str.replace('오전', 'AM')
        .str.replace('오후', 'PM')
    )
    df['Time'] = pd.to_datetime(df['Time'], format='%p %I:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
    
    # 5. 날짜와 시간 컬럼 병합
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    
    # 6. 인덱스 컬럼 int로 변환
    df['Index'] = df['Index'].astype(int)
    
    # 3. 에러 프로세스 표기
    df['is_anomaly'] = 0  # 기본값
    for _, row in error_df.iterrows():
        date = str(row.iloc[0]).strip()
        error_processes = set(row.iloc[1:].dropna().astype(int))
        if len(error_processes) > 0:
            mask = (df['Date'] == date) & (df['Process'].isin(error_processes))
            df.loc[mask, 'is_anomaly'] = 1
    
    dataframes.append(df)

# 7. LSTM 입력 형태로 수정
WINDOW_SIZE = 10
X_all = []
y_all = []

for df in dataframes:
    # 각 데이터프레임을 시퀀스로 변환
    features = df[['Temp', 'Current']].values
    labels = df['is_anomaly'].values
    
    # 시퀀스 생성
    for i in range(len(df) - WINDOW_SIZE):
        X_all.append(features[i:i + WINDOW_SIZE])
        # 시퀀스 내에 하나라도 이상이 있으면 1로 라벨링
        y_all.append(1 if np.any(labels[i:i + WINDOW_SIZE]) else 0)

X = np.array(X_all)
y = np.array(y_all)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# LSTM 모델 정의
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), 
         return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),
    GRU(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

class F1ScoreCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(F1ScoreCallback, self).__init__()
        self.validation_data = validation_data
        
    def on_epoch_end(self, epoch, logs=None):
        val_predict = (self.model.predict(self.validation_data[0]) > 0.5).astype(int)
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        print(f' - val_f1_score: {_val_f1:.4f}')
        logs['val_f1_score'] = _val_f1

# Early Stopping 설정
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# F1 Score 콜백 추가
f1_callback = F1ScoreCallback(validation_data=(X_val, y_val))

# 모델 학습
print('\n===================== LSTM =====================')
history = model.fit(
    X_train, 
    y_train, 
    epochs=200,
    batch_size=32, 
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, f1_callback],
    verbose=1
)

# 예측 및 평가
y_scores = model.predict(X_test).flatten()
y_pred = (y_scores >= 0.5).astype(int)

# 결과 출력
print("\n테스트 데이터 평가:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))