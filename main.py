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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc, roc_auc_score, f1_score
from sklearn.covariance import MinCovDet
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from imblearn.over_sampling import SMOTE

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

X, y = create_sequences(csv_data, window_size=10)
# 평탄화 (LSTM용 시계열 데이터를 2D로 바꿔야 SMOTE 가능)
X_flat = X.reshape(X.shape[0], -1)  # (samples, timesteps * features)
# SMOTE 적용
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_flat, y)
# 다시 원래 LSTM 입력 형태로 복원
X_resampled = X_resampled.reshape(-1, X.shape[1], X.shape[2])

model = Sequential([
    LSTM(128, input_shape=(X_resampled.shape[1], X_resampled.shape[2]), return_sequences=False),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print('\n===================== LSTM =====================')
history = model.fit(
    X_resampled, 
    y_resampled, 
    epochs=200, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stopping],
    verbose=1
)

# # 기존의 X, y에서 train/test 분리
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

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

def plot_history(history, title_prefix="Model"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'b-', label='Training Loss')
    plt.plot(history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'{title_prefix} Loss over Epochs')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title(f'{title_prefix} Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()

def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
    disp.plot(cmap='Blues')
    plt.title(title)
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"]))

def plot_mahalanobis(X_seq, y_true, title="Mahalanobis Distance Score"):
    X_flat = X_seq.reshape(X_seq.shape[0], -1)
    robust_cov = MinCovDet().fit(X_flat[y_true == 0])
    scores = robust_cov.mahalanobis(X_flat)

    plt.figure(figsize=(8, 4))
    plt.plot(scores, label="Mahalanobis Distance Score")
    plt.axhline(np.percentile(scores, 95), color='r', linestyle='--', label='95% Threshold')
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    top_indices = np.argsort(scores)[::-1][:5]
    return top_indices, scores[top_indices]

# LSTM
plot_history(history, title_prefix="LSTM")

# 연속적 이상 점수화 및 ROC 커브 분석
y_scores = model.predict(X_resampled).flatten()

# 이상 점수 분포 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.kdeplot(data=y_scores[y_resampled == 0], label='Normal', fill=True, alpha=0.3)
sns.kdeplot(data=y_scores[y_resampled == 1], label='Anomaly', fill=True, alpha=0.3)
plt.title('Anomaly Score Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()

# ROC 커브 및 최적 임계값 계산
fpr, tpr, thresholds = roc_curve(y_resampled, y_scores)
roc_auc = auc(fpr, tpr)

# 최적 임계값 찾기 (Youden's J statistic)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', 
           label=f'Optimal Threshold: {optimal_threshold:.2f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.grid()
plt.legend(loc="lower right")
plt.tight_layout()

# 최적 임계값으로 예측
y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
plot_confusion(y_resampled, y_pred_optimal, title="Confusion Matrix (Optimal Threshold)")

# 기존 0.5 임계값과 비교
y_pred_default = (y_scores >= 0.5).astype(int)
plot_confusion(y_resampled, y_pred_default, title="Confusion Matrix (Default Threshold 0.5)")

plot_mahalanobis(X, y, title="Mahalanobis Distance Score")

print(f"F1 Score: {f1_score(y_resampled, y_pred_optimal)}")

plt.show()