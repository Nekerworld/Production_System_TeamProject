import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import streamlit as st
import pandas as pd
import os
from glob import glob
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.covariance import MinCovDet

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

# 복사본 생성
csv_data = five_process_180sec_merged.copy()
csv_data['label'] = 0  # 초기값: 정상(0)

# Error Lot 리스트 불러오기
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
    features = df[['Temp', 'Current']].values  # 딱 한 번만 슬라이싱
    labels = df['label'].values

    for i in range(len(df) - window_size):
        X.append(features[i:i + window_size])
        y.append(labels[i + window_size])

    return np.array(X), np.array(y)

X, y = create_sequences(csv_data, window_size=10)
# 평탄화 (LSTM용 시계열 데이터를 2D로 바꿔야 SMOTE 가능)
X_flat = X.reshape(X.shape[0], -1)  # (samples, timesteps * features)
# SMOTE 적용
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_flat, y)
# 다시 원래 LSTM 입력 형태로 복원
X_resampled = X_resampled.reshape(-1, X.shape[1], X.shape[2])

model = Sequential([
    LSTM(64, input_shape=(X_resampled.shape[1], X_resampled.shape[2]), return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_resampled, y_resampled, epochs=50, batch_size=32, validation_split=0.2)


# 학습 그래프
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b-', label='Training Loss')
plt.plot(history.history['val_loss'], 'r-', label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b-', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()

# Confusion Matrix + Classification Report
y_pred_probs = model.predict(X_resampled)
y_pred = (y_pred_probs > 0.5).astype(int)
cm = confusion_matrix(y_resampled, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")

print("Classification Report:")
print(classification_report(y_resampled, y_pred, target_names=["Normal", "Anomaly"]))

# Mahalanobis 거리 기반 이상도 점수
X_maha = X.reshape(X.shape[0], -1)
y_maha = y
robust_cov = MinCovDet().fit(X_maha[y_maha == 0])
scores = robust_cov.mahalanobis(X_maha)


# 이상 점수 시각화
plt.figure(figsize=(8, 4))
plt.plot(scores, label="Mahalanobis Distance Score")
plt.axhline(np.percentile(scores, 95), color='r', linestyle='--', label='95% Threshold')
plt.title("Anomaly Score (Mahalanobis Distance)")
plt.xlabel("Sample Index")
plt.ylabel("Anomaly Score")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 상위 이상 샘플 출력
top_indices = np.argsort(scores)[::-1][:5]
top_anomalies = X_resampled[top_indices]
top_anomalies

plt.show()