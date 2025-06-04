import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, Dropout, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
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
from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, NearMiss, OneSidedSelection

# Focal Loss 구현
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon()) + \
                      (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

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
        # 윈도우 내에 하나라도 이상이 있으면 1로 라벨링
        y.append(1 if np.any(labels[i:i + window_size]) else 0)

    return np.array(X), np.array(y)

# 윈도우 크기를 변수로 저장
WINDOW_SIZE = 10
X, y = create_sequences(csv_data, window_size=WINDOW_SIZE)
# 평탄화 (LSTM용 시계열 데이터를 2D로 바꿔야 SMOTE 가능)
X_flat = X.reshape(X.shape[0], -1)  # (samples, timesteps * features)

# 시간순으로 데이터 분할 (과거 80% -> 미래 20%)
train_size = int(len(X_flat) * 0.8)
X_train, X_test = X_flat[:train_size], X_flat[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# validation set도 시간순으로 분할 (train의 마지막 10%)
val_size = int(len(X_train) * 0.1)
X_train, X_val = X_train[:-val_size], X_train[-val_size:]
y_train, y_val = y_train[:-val_size], y_train[-val_size:]

# BorderlineSMOTE로 1:8 비율까지 오버샘플링
k_neighbors = min(5, sum(y_train == 1))
smote = BorderlineSMOTE(sampling_strategy=0.125, random_state=42, k_neighbors=k_neighbors)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 클래스 가중치 계산
pos = sum(y_train == 1)
neg = sum(y_train == 0)
class_weight = {0: 1, 1: neg/pos}

# 다시 원래 LSTM 입력 형태로 복원
X_train = X_train.reshape(-1, WINDOW_SIZE, X.shape[2])
X_test = X_test.reshape(-1, WINDOW_SIZE, X.shape[2])
X_val = X_val.reshape(-1, WINDOW_SIZE, X.shape[2])

model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Focal Loss와 class_weight 모두 사용
model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2., alpha=.25),
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,  # patience 조정
    restore_best_weights=True,
    verbose=1
)

print('\n===================== LSTM =====================')
history = model.fit(
    X_train, 
    y_train, 
    epochs=50,  # epoch 수 조정
    batch_size=32, 
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    class_weight=class_weight,  # 클래스 가중치 적용
    verbose=1
)

def plot_history(history, title_prefix="Model"):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'b-', label='Training Loss')
    plt.plot(history.history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'{title_prefix} Loss over {len(history.history["loss"])} Epochs')
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title(f'{title_prefix} Accuracy over {len(history.history["accuracy"])} Epochs')
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
y_scores = model.predict(X_test).flatten()

# 이상 점수 분포 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.kdeplot(data=y_scores[y_test == 0], label='Normal', fill=True, alpha=0.3)
sns.kdeplot(data=y_scores[y_test == 1], label='Anomaly', fill=True, alpha=0.3)
plt.title('Anomaly Score Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()

# PR-AUC 기반 최적 임계값 찾기
prec, rec, thresholds = precision_recall_curve(y_test, y_scores)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
optimal_idx = np.argmax(f1)
optimal_threshold = thresholds[optimal_idx]

# 최적 임계값으로 예측
y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
plot_confusion(y_test, y_pred_optimal, title="Confusion Matrix (PR-AUC Optimal Threshold - Test Data)")

# PR-AUC 점수 계산
pr_auc = average_precision_score(y_test, y_scores)
print(f"\nPR-AUC Score: {pr_auc:.4f}")
print(f"Test Data F1 Score: {f1_score(y_test, y_pred_optimal, zero_division=0):.4f}")

# Top-n% alert 방식의 임계값도 계산 (상위 0.2%)
top_n_threshold = np.percentile(y_scores, 99.8)
print(f"\nTop 0.2% Alert Threshold: {top_n_threshold:.4f}")

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