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
from imblearn.over_sampling import SMOTE, RandomOverSampler, BorderlineSMOTE, ADASYN, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids, NearMiss, OneSidedSelection

# 로컬에서 실행시 경로 알맞게 설정해주세요
five_process_180sec = r'C:\\YS\\TUK\\S4E1\\생산시스템구축실무\\TeamProject\\Production_System_TeamProject\\data\\장비이상 조기탐지\\5공정_180sec'

# 모든 csv 파일 목록을 가져옴 (Error Lot 제외)
all_csv_files = glob(os.path.join(five_process_180sec, '*.csv'))
csv_files = [f for f in all_csv_files if 'Error Lot list' not in os.path.basename(f)]

# ① 에러 Lot 리스트 읽기
error_df = pd.read_csv(os.path.join(five_process_180sec, 'Error Lot list.csv'))
# ② {날짜(str): {에러 Index, …}} 형태의 딕셔너리 생성
error_dict = {}
for _, row in error_df.iterrows():
    date = str(row.iloc[0]).strip()
    idx_set = set(row.iloc[1:].dropna().astype(int))
    if len(idx_set) > 0:
        error_dict[date] = idx_set

dataframes = []

for file in csv_files:
    df = pd.read_csv(file)

    # 2-1) 한글 '오전/오후' → AM/PM 변환 + 24시간제로 통일
    df['Time'] = (
        df['Time']
        .str.replace('오전', 'AM')
        .str.replace('오후', 'PM')
    )
    df['Time'] = pd.to_datetime(df['Time'], format='%p %I:%M:%S.%f').dt.strftime('%H:%M:%S.%f')

    # 2-2) 날짜·시간 결합
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')

    # 2-3) 원본 파일명 보존(선택)
    df['source_file'] = os.path.basename(file)

    # 2-4) Index 컬럼이 실수형으로 읽히는 경우 대비 → int 변환
    df['Index'] = df['Index'].astype(int)

    # 2-5) ***vectorized*** 방식으로 is_anomaly 생성
    df['is_anomaly'] = 0  # 기본값
    for d, idx_set in error_dict.items():
        mask = (df['Date'] == d) & (df['Index'].isin(idx_set))
        df.loc[mask, 'is_anomaly'] = 1

    # 2-6) 리스트에 적재
    dataframes.append(df)
    
five_process_180sec_merged = pd.concat(dataframes, ignore_index=True)
five_process_180sec_merged.sort_values('datetime', inplace=True)
five_process_180sec_merged.reset_index(drop=True, inplace=True)

# is_anomaly를 label로 사용
csv_data = five_process_180sec_merged.copy()
csv_data['label'] = csv_data['is_anomaly']

# Focal Loss 구현
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon()) + \
                      (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
    return focal_loss_fixed

# 첫 컬럼은 날짜, 나머지는 해당 날짜에 이상이 있었던 Index들
for i in range(len(error_df)):
    date = error_df.iloc[i, 0]
    error_indices = error_df.iloc[i, 1:].dropna().astype(int).tolist()

    # 해당 날짜 & Index에 대해 label = 1 할당
    mask = (csv_data['Date'] == date) & (csv_data['Index'].isin(error_indices))
    csv_data.loc[mask, 'label'] = 1

def create_sequences(df, window_size=10):
    X, y = [], []
    
    # Process별로 그룹화하여 시퀀스 생성
    for process_id, group in df.groupby('Process'):
        features = group[['Temp', 'Current']].values
        labels = group['label'].values
        
        # 각 Process 내에서 시퀀스 생성
        for i in range(len(group) - window_size):
            X.append(features[i:i + window_size])
            # Process 내에 하나라도 이상이 있으면 1로 라벨링
            y.append(1 if np.any(labels[i:i + window_size]) else 0)
    
    return np.array(X), np.array(y)

# Process 단위로 train/test 분할 시 이상 샘플 보장
def split_by_process(X, y, process_ids, test_size=0.2):
    unique_processes = np.unique(process_ids)
    anomaly_processes = set()
    
    # 이상이 있는 Process 식별
    for i, process_id in enumerate(process_ids):
        if y[i] == 1:
            anomaly_processes.add(process_id)
    
    # 이상 Process의 일부를 test set에 포함
    test_anomaly = set(np.random.choice(list(anomaly_processes), 
                                      size=max(1, int(len(anomaly_processes) * test_size)),
                                      replace=False))
    
    # 나머지 Process 분할
    remaining_processes = set(unique_processes) - test_anomaly
    test_normal = set(np.random.choice(list(remaining_processes), 
                                     size=int(len(remaining_processes) * test_size),
                                     replace=False))
    
    test_processes = test_anomaly | test_normal
    train_processes = set(unique_processes) - test_processes
    
    train_mask = np.isin(process_ids, list(train_processes))
    test_mask = np.isin(process_ids, list(test_processes))
    
    return (X[train_mask], X[test_mask], 
            y[train_mask], y[test_mask],
            train_mask, test_mask)

# 윈도우 크기를 변수로 저장
WINDOW_SIZE = 10

# Process 단위로 라벨 전파 (수정)
for process_id, group in csv_data.groupby('Process'):
    if group['is_anomaly'].any():  # is_anomaly를 기준으로 확인
        # 이상이 발생한 시점 주변만 라벨링 (예: ±5개 시점)
        anomaly_indices = group[group['is_anomaly'] == 1].index
        for idx in anomaly_indices:
            start_idx = max(idx - 5, group.index[0])
            end_idx = min(idx + 5, group.index[-1])
            csv_data.loc[start_idx:end_idx, 'label'] = 1

# 시퀀스 생성 전에 라벨 분포 확인
print("라벨 분포:", np.unique(csv_data['label'], return_counts=True))

# 시퀀스 생성 (process별로 자르고 그 안에서 window size로 sliding window 생성)
X, y = create_sequences(csv_data, window_size=WINDOW_SIZE)

# 시퀀스 생성 후 라벨 분포 확인
print("시퀀스 라벨 분포:", np.unique(y, return_counts=True))

# Process ID도 시퀀스에 맞게 조정
process_ids = []
for process_id, group in csv_data.groupby('Process'):
    process_ids.extend([process_id] * (len(group) - WINDOW_SIZE))
process_ids = np.array(process_ids)

# Process 단위로 train/test 분할 (수정)
# 이상이 있는 Process를 test set에도 포함되도록 수정
X_train, X_test, y_train, y_test, train_mask, test_mask = split_by_process(
    X, y, 
    process_ids,
    test_size=0.2
)

# validation set도 Process 단위로 분할
X_train, X_val, y_train, y_val, _, _ = split_by_process(
    X_train, y_train,
    process_ids[train_mask],
    test_size=0.1
)

# SMOTE 적용 전에 2차원으로 평탄화
X_train_flat = X_train.reshape(X_train.shape[0], -1)

# BorderlineSMOTE로 1:8 비율까지 오버샘플링
k_neighbors = min(5, sum(y_train == 1))
if sum(y_train == 1) > 0:  # 이상 클래스가 있는 경우에만 SMOTE 적용
    smote = BorderlineSMOTE(sampling_strategy=0.125, random_state=42, k_neighbors=k_neighbors)
    X_train_flat, y_train = smote.fit_resample(X_train_flat, y_train)

# 다시 원래 LSTM 입력 형태로 복원
X_train = X_train_flat.reshape(-1, WINDOW_SIZE, X.shape[2])
X_test = X_test.reshape(-1, WINDOW_SIZE, X.shape[2])
X_val = X_val.reshape(-1, WINDOW_SIZE, X.shape[2])

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

# Focal Loss와 class_weight 모두 사용
model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2., alpha=.25),
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,  # patience 조정
    restore_best_weights=True,
    verbose=1
)

print('\n===================== LSTM =====================')
history = model.fit(
    X_train, 
    y_train, 
    epochs=100,  # epoch 수 조정
    batch_size=32, 
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
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
plt.title('Anomaly Score Distribution')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()

# Precision-Recall 커브 기반 최적 임계값 찾기
prec, rec, thresholds = precision_recall_curve(y_test, y_scores)
f1 = 2 * prec * rec / (prec + rec + 1e-9)  # F1-score 계산
optimal_idx = np.argmax(f1)
optimal_threshold_pr = thresholds[optimal_idx]

# 최적 임계값으로 예측
y_pred_optimal_pr = (y_scores >= optimal_threshold_pr).astype(int)

plot_confusion(y_test, y_pred_optimal_pr, title="Confusion Matrix (Threshold Precision Recall - Test Data)")


print(f"Test Data pr F1 Score: {f1_score(y_test, y_pred_optimal_pr, zero_division=0):.4f}")

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
print(f"이상 판정 임계값: {optimal_threshold_pr*100:.2f}%")
print(f"판정: {'이상' if prediction >= optimal_threshold_pr else '정상'}")

