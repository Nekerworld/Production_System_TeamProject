# 현재는 discrete하게 결과가 나뉘고 있는데, 이를 연속적으로 한번 해보고 성능 괜찮게 나오면 고장 확률 뱉는 모델로 변경

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.utils import plot_model
import streamlit as st
import pandas as pd
import os
from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, NearMiss, OneSidedSelection
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

# 로컬에서 실행시 경로 알맞게 설정해주세요
five_process_180sec = r'C:\\YS\\TUK\\S4E1\\생산시스템구축실무\\TeamProject\\Production_System_TeamProject\\data\\장비이상 조기탐지\\5공정_180sec'

# 모든 csv 파일 목록을 가져옴 (Error Lot 제외)
all_csv_files = glob(os.path.join(five_process_180sec, '*.csv'))
csv_files = [f for f in all_csv_files if 'Error Lot list' not in os.path.basename(f)]

# 병합할 데이터프레임 리스트
dataframes = []

for file in csv_files:
    df = pd.read_csv(file)

    # 날짜 및 시간 컬럼 병합
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')

    # 파일명을 기준으로 'source_file' 컬럼 생성 (optional: 날짜 추적용)
    df['source_file'] = os.path.basename(file)

    dataframes.append(df)
    
five_process_180sec_merged = pd.concat(dataframes, ignore_index=True)
five_process_180sec_merged.sort_values(by='datetime', inplace=True)
five_process_180sec_merged.reset_index(drop=True, inplace=True)
five_process_180sec_error_lot = pd.read_csv(os.path.join(five_process_180sec, 'Error Lot list.csv'))

csv_data = five_process_180sec_merged.copy()
csv_data['label'] = 0
error_df = five_process_180sec_error_lot.copy()

# 첫 컬럼은 날짜, 나머지는 해당 날짜에 이상이 있었던 Index들
for i in range(len(error_df)):
    date = error_df.iloc[i, 0]
    error_indices = error_df.iloc[i, 1:].dropna().astype(int).tolist()

    # 해당 날짜 & Index에 대해 label = 1 할당
    mask = (csv_data['Date'] == date) & (csv_data['Index'].isin(error_indices))
    csv_data.loc[mask, 'label'] = 1

def create_sequences(df, window_size=10):
    X, y = [], []
    features = df[['Temp', 'Current']].values
    labels = df['label'].values

    for i in range(len(df) - window_size):
        X.append(features[i:i + window_size])
        y.append(labels[i + window_size])

    return np.array(X), np.array(y)

# 윈도우 크기를 변수로 저장
WINDOW_SIZE = 10
X, y = create_sequences(csv_data, window_size=WINDOW_SIZE)
# 평탄화 (LSTM용 시계열 데이터를 2D로 바꿔야 SMOTE 가능)
X_flat = X.reshape(X.shape[0], -1)  # (samples, timesteps * features)

# train_test_split 적용
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y
)

# BorderlineSMOTE 파라미터 조정
k_neighbors = min(5, sum(y_train == 1))  # 이상 샘플 수와 5 중 작은 값 사용
smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 다시 원래 LSTM 입력 형태로 복원
X_train = X_train.reshape(-1, WINDOW_SIZE, X.shape[2])
X_test = X_test.reshape(-1, WINDOW_SIZE, X.shape[2])

model = Sequential([
    LSTM(128, input_shape=(WINDOW_SIZE, X.shape[2]), return_sequences=False),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

print('\n===================== LSTM =====================')
history = model.fit(
    X_train, 
    y_train, 
    epochs=200,  # epochs 증가
    batch_size=64, 
    validation_split=0.1, 
    callbacks=[early_stopping],
    verbose=1
)

# 테스트 데이터에 대한 예측
y_scores = model.predict(X_test).flatten()

# === Autoencoder 기반 ===
# input_dim = X_train.shape[2]
# timesteps = X_train.shape[1]

# # Autoencoder 구성
# inputs = Input(shape=(timesteps, input_dim))
# encoded = LSTM(64, return_sequences=False)(inputs)
# decoded = RepeatVector(timesteps)(encoded)
# decoded = LSTM(input_dim, return_sequences=True)(decoded)
# autoencoder = Model(inputs, decoded)
# autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# print('\n===================== Autoencoder =====================')
# history_ae = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# # 예측 → Reconstruction Error 계산
# X_test_pred = autoencoder.predict(X_test)
# mse = np.mean(np.power(X_test - X_test_pred, 2), axis=(1, 2))
# threshold = np.percentile(mse, 95)
# y_pred_ae = (mse > threshold).astype(int)

# === Transformer 기반 ===
# def transformer_model(input_shape):
#     inputs = Input(shape=input_shape)
#     x = Dense(64)(inputs)
#     attn = MultiHeadAttention(num_heads=2, key_dim=32)(x, x)
#     x = LayerNormalization()(x + attn)
#     x = GlobalAveragePooling1D()(x)
#     x = Dense(32, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     output = Dense(1, activation='sigmoid')(x)
#     model = Model(inputs, output)
#     return model

# transformer = transformer_model((timesteps, input_dim))
# transformer.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print('\n===================== Transformer =====================')
# history_tr = transformer.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# y_pred_tr = (transformer.predict(X_test) > 0.5).astype(int)

def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
    disp.plot(cmap='Blues')
    plt.title(title)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

# ROC 커브 및 최적 임계값 계산
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# 최적 임계값 찾기 (Youden's J statistic)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
# 최적 임계값으로 예측
y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
plot_confusion(y_test, y_pred_optimal, title="Confusion Matrix (Optimal Threshold - Test Data)")

# 기존 0.5 임계값과 비교
y_pred_default = (y_scores >= 0.5).astype(int)
plot_confusion(y_test, y_pred_default, title="Confusion Matrix (Default Threshold 0.5 - Test Data)")

print(f"Test Data F1 Score: {f1_score(y_test, y_pred_optimal, zero_division=0)}")

# 테스트 데이터 F1 스코어 계산
test_f1 = f1_score(y_test, y_pred_optimal, zero_division=0)
print(f"\n테스트 데이터 F1 스코어: {test_f1:.4f}")

plt.show()

# 새로운 데이터 예측
print("\n새로운 데이터 예측을 시작합니다.")
print("현재 시점의 온도와 전류 값을 입력해주세요.")

# 현재 시점의 데이터 입력 받기
current_temp = float(input("현재 온도 값을 입력하세요: "))
current_current = float(input("현재 전류 값을 입력하세요: "))

# 마지막 샘플의 마지막 9개 시점 데이터 추출
last_sample = X_test[-1]  # shape: (10, 2)
last_9_data = last_sample[-9:, :]  # shape: (9, 2)
current_data = np.array([[current_temp, current_current]])  # shape: (1, 2)

# 10개 시점 데이터로 시퀀스 생성
new_sequence = np.vstack([last_9_data, current_data])  # shape: (10, 2)
new_sequence = new_sequence.reshape(1, WINDOW_SIZE, 2)  # (1, 10, 2)

# 예측
prediction = model.predict(new_sequence)[0][0]
print(f"\n예측 결과:")
print(f"이상 확률: {prediction*100:.2f}%")
print(f"정상 확률: {(1-prediction)*100:.2f}%")
print(f"판정: {'이상' if prediction >= optimal_threshold else '정상'}")