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
    patience=15,
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

def plot_time_series_overlay(y_scores, y_true, timestamps, title="Time Series Anomaly Detection"):
    plt.figure(figsize=(15, 6))
    
    # 이상 점수 시계열 플롯
    plt.plot(range(len(y_scores)), y_scores, label='Anomaly Score', alpha=0.7, color='blue')
    
    # 실제 이상 발생 지점 표시
    anomaly_indices = np.where(y_true == 1)[0]
    plt.scatter(anomaly_indices, y_scores[anomaly_indices], 
               color='red', label='Anomalies', alpha=0.7, s=100)
    
    # 임계값 선 추가
    plt.axhline(y=0.5, color='r', linestyle='--', label='Default Threshold (0.5)')
    
    # 그래프 스타일 설정
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def plot_borderline_analysis(X_flat, y, y_scores, title_prefix="Data Analysis"):
    plt.figure(figsize=(15, 10))
    
    # 1. 데이터 분포 시각화
    plt.subplot(2, 2, 1)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_flat)
    
    # 정상/이상 데이터 구분
    normal_mask = y == 0
    anomaly_mask = y == 1
    
    plt.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], 
               c='blue', label='Normal', alpha=0.3)
    plt.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1], 
               c='red', label='Anomaly', alpha=0.3)
    plt.title(f'{title_prefix} - Data Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 이상 점수 분포
    plt.subplot(2, 2, 2)
    sns.kdeplot(data=y_scores[y == 0], label='Normal', fill=True, alpha=0.3)
    sns.kdeplot(data=y_scores[y == 1], label='Anomaly', fill=True, alpha=0.3)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold')
    plt.title(f'{title_prefix} - Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 성능 지표
    plt.subplot(2, 2, 3)
    metrics = {
        'Metrics': {
            'Precision': precision_score(y, y_scores > 0.5, zero_division=0),
            'Recall': recall_score(y, y_scores > 0.5, zero_division=0),
            'F1': f1_score(y, y_scores > 0.5, zero_division=0)
        }
    }
    
    x = np.arange(3)
    width = 0.35
    
    plt.bar(x, list(metrics['Metrics'].values()), width)
    plt.xticks(x, ['Precision', 'Recall', 'F1 Score'])
    plt.title(f'{title_prefix} - Performance Metrics')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def plot_detailed_borderline_analysis(X_flat, y, y_scores, title_prefix="Detailed Analysis"):
    plt.figure(figsize=(20, 15))
    
    # 1. 경계선 영역의 동적 임계값 분석
    plt.subplot(3, 2, 1)
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof_scores = -lof.fit_predict(X_flat)  # 음수 값을 양수로 변환
    
    # 경계선 영역의 동적 임계값 계산
    threshold_percentiles = [50, 75, 90, 95]
    thresholds = [np.percentile(lof_scores, p) for p in threshold_percentiles]
    
    plt.hist(lof_scores, bins=50, alpha=0.5, label='Score Distribution')
    for t, p in zip(thresholds, threshold_percentiles):
        plt.axvline(t, color='r', linestyle='--', 
                   label=f'{p}th percentile: {t:.2f}')
    plt.title(f'{title_prefix} - Dynamic Boundary Thresholds')
    plt.xlabel('LOF Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 데이터 클러스터링 분석
    plt.subplot(3, 2, 2)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_flat)
    
    # 각 클러스터의 중심점 시각화
    centers = kmeans.cluster_centers_
    plt.scatter(X_flat[:, 0], X_flat[:, 1], c=cluster_labels, cmap='viridis', alpha=0.3)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
    
    plt.title(f'{title_prefix} - Cluster Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Precision-Recall 커브
    plt.subplot(3, 2, 3)
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, _ = precision_recall_curve(y, y_scores)
    ap = average_precision_score(y, y_scores)
    
    plt.plot(recall, precision, label=f'AP={ap:.2f}')
    plt.title(f'{title_prefix} - Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 임계값별 성능 변화
    plt.subplot(3, 2, 4)
    thresholds = np.linspace(0, 1, 100)
    metrics = {'precision': [], 'recall': [], 'f1': []}
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        metrics['precision'].append(precision_score(y, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y, y_pred, zero_division=0))
    
    for metric in ['precision', 'recall', 'f1']:
        plt.plot(thresholds, metrics[metric], label=metric.capitalize())
    
    plt.title(f'{title_prefix} - Performance Metrics vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. 특성 중요도 분석
    plt.subplot(3, 2, 5)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_flat, y)
    feature_importance = rf.feature_importances_
    
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title(f'{title_prefix} - Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    
    # 6. 이상 점수 분포의 통계적 분석
    plt.subplot(3, 2, 6)
    from scipy import stats
    
    # 이상 점수 분포 통계
    score_stats = stats.describe(y_scores)
    
    stats_data = {
        'Statistics': [score_stats.mean, score_stats.variance, score_stats.skewness, score_stats.kurtosis]
    }
    
    x = np.arange(4)
    width = 0.35
    
    plt.bar(x, stats_data['Statistics'], width)
    plt.xticks(x, ['Mean', 'Variance', 'Skewness', 'Kurtosis'])
    plt.title(f'{title_prefix} - Statistical Analysis')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

# LSTM
plot_history(history, title_prefix="LSTM")

# Time Series Overlay 시각화
plot_time_series_overlay(y_scores, y_test, None, title="LSTM Anomaly Detection Over Time (Test Data)")

# 데이터 분석 시각화
plot_borderline_analysis(X_test.reshape(X_test.shape[0], -1), y_test, y_scores, title_prefix="Test Data Analysis")

# 상세 데이터 분석 시각화
plot_detailed_borderline_analysis(X_test.reshape(X_test.shape[0], -1), y_test, y_scores, title_prefix="Detailed Test Data Analysis")

# 이상 점수 분포 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.kdeplot(data=y_scores[y_test == 0], label='Normal', fill=True, alpha=0.3)
sns.kdeplot(data=y_scores[y_test == 1], label='Anomaly', fill=True, alpha=0.3)
plt.title('Anomaly Score Distribution (Test Data)')
plt.xlabel('Anomaly Score')
plt.ylabel('Density')
plt.legend()

# ROC 커브 및 최적 임계값 계산
fpr, tpr, thresholds = roc_curve(y_test, y_scores)
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
plt.title('ROC Curve (Test Data)')
plt.grid()
plt.legend(loc="lower right")
plt.tight_layout()

# 최적 임계값으로 예측
y_pred_optimal = (y_scores >= optimal_threshold).astype(int)
plot_confusion(y_test, y_pred_optimal, title="Confusion Matrix (Optimal Threshold - Test Data)")

# 기존 0.5 임계값과 비교
y_pred_default = (y_scores >= 0.5).astype(int)
plot_confusion(y_test, y_pred_default, title="Confusion Matrix (Default Threshold 0.5 - Test Data)")

print(f"Test Data F1 Score: {f1_score(y_test, y_pred_optimal, zero_division=0)}")

plt.show()