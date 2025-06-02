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

X, y = create_sequences(csv_data, window_size=10)
# 평탄화 (LSTM용 시계열 데이터를 2D로 바꿔야 SMOTE 가능)
X_flat = X.reshape(X.shape[0], -1)  # (samples, timesteps * features)
# SMOTE 적용
smote = BorderlineSMOTE(sampling_strategy='auto', random_state=42)
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
    epochs=2, 
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
    
    # 실제 이상 발생 지점 표시 (BorderlineSMOTE로 생성된 샘플 구분)
    original_anomaly_indices = np.where(y_true == 1)[0][:len(y_scores)]  # 원본 이상 샘플
    synthetic_anomaly_indices = np.where(y_true == 1)[0][len(y_scores):]  # BorderlineSMOTE로 생성된 샘플
    
    plt.scatter(original_anomaly_indices, y_scores[original_anomaly_indices], 
               color='red', label='Original Anomalies', alpha=0.7, s=100)
    plt.scatter(synthetic_anomaly_indices, y_scores[synthetic_anomaly_indices], 
               color='orange', label='Synthetic Anomalies', alpha=0.5, s=50)
    
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

def plot_borderline_analysis(X_flat, y, y_scores, title_prefix="BorderlineSMOTE Analysis"):
    plt.figure(figsize=(15, 10))
    
    # 1. BorderlineSMOTE의 경계선 영역 시각화
    plt.subplot(2, 2, 1)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_flat)
    
    # 정상/이상 데이터 구분
    normal_mask = y == 0
    anomaly_mask = y == 1
    
    # 경계선 영역 계산 (KNN 기반)
    nbrs = NearestNeighbors(n_neighbors=5).fit(X_2d[normal_mask])
    distances, _ = nbrs.kneighbors(X_2d)
    boundary_mask = np.percentile(distances, 75) < distances[:, 0]  # 상위 25% 거리에 있는 점들을 경계선으로 간주
    
    plt.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], 
               c='blue', label='Normal', alpha=0.3)
    plt.scatter(X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1], 
               c='red', label='Anomaly', alpha=0.3)
    plt.scatter(X_2d[boundary_mask, 0], X_2d[boundary_mask, 1], 
               c='green', label='Borderline Region', alpha=0.1)
    plt.title(f'{title_prefix} - Borderline Region')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 합성 샘플 생성 과정 분석
    plt.subplot(2, 2, 2)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_flat)
    
    plt.scatter(X_tsne[normal_mask, 0], X_tsne[normal_mask, 1], 
               c='blue', label='Normal', alpha=0.3)
    plt.scatter(X_tsne[anomaly_mask, 0], X_tsne[anomaly_mask, 1], 
               c='red', label='Anomaly', alpha=0.3)
    plt.title(f'{title_prefix} - t-SNE Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 원본/합성 샘플별 성능 지표
    plt.subplot(2, 2, 3)
    original_scores = y_scores[:len(y)]
    synthetic_scores = y_scores[len(y):]
    
    sns.kdeplot(data=original_scores, label='Original Samples', fill=True, alpha=0.3)
    sns.kdeplot(data=synthetic_scores, label='Synthetic Samples', fill=True, alpha=0.3)
    plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold')
    plt.title(f'{title_prefix} - Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 성능 지표 비교
    plt.subplot(2, 2, 4)
    metrics = {
        'Original': {
            'Precision': precision_score(y[:len(y)], original_scores > 0.5),
            'Recall': recall_score(y[:len(y)], original_scores > 0.5),
            'F1': f1_score(y[:len(y)], original_scores > 0.5)
        },
        'Synthetic': {
            'Precision': precision_score(y_resampled[len(y):], synthetic_scores > 0.5),
            'Recall': recall_score(y_resampled[len(y):], synthetic_scores > 0.5),
            'F1': f1_score(y_resampled[len(y):], synthetic_scores > 0.5)
        }
    }
    
    x = np.arange(3)
    width = 0.35
    
    plt.bar(x - width/2, list(metrics['Original'].values()), width, label='Original')
    plt.bar(x + width/2, list(metrics['Synthetic'].values()), width, label='Synthetic')
    plt.xticks(x, ['Precision', 'Recall', 'F1 Score'])
    plt.title(f'{title_prefix} - Performance Metrics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def plot_detailed_borderline_analysis(X_flat, y, y_scores, title_prefix="Detailed BorderlineSMOTE Analysis"):
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
    
    # 2. 합성 샘플 생성 과정의 단계별 시각화
    plt.subplot(3, 2, 2)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(X_flat)
    
    # 각 클러스터의 중심점과 합성 샘플 생성 방향 시각화
    centers = kmeans.cluster_centers_
    plt.scatter(X_flat[:, 0], X_flat[:, 1], c=cluster_labels, cmap='viridis', alpha=0.3)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
    
    # 합성 샘플 생성 방향 표시
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            plt.arrow(centers[i, 0], centers[i, 1],
                     centers[j, 0] - centers[i, 0],
                     centers[j, 1] - centers[i, 1],
                     color='red', alpha=0.3, head_width=0.1)
    
    plt.title(f'{title_prefix} - Synthetic Sample Generation Process')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 원본/합성 샘플별 상세 성능 지표
    plt.subplot(3, 2, 3)
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # 원본 샘플의 PR 커브
    precision_orig, recall_orig, _ = precision_recall_curve(y[:len(y)], y_scores[:len(y)])
    ap_orig = average_precision_score(y[:len(y)], y_scores[:len(y)])
    
    # 합성 샘플의 PR 커브
    precision_synth, recall_synth, _ = precision_recall_curve(y_resampled[len(y):], y_scores[len(y):])
    ap_synth = average_precision_score(y_resampled[len(y):], y_scores[len(y):])
    
    plt.plot(recall_orig, precision_orig, label=f'Original (AP={ap_orig:.2f})')
    plt.plot(recall_synth, precision_synth, label=f'Synthetic (AP={ap_synth:.2f})')
    plt.title(f'{title_prefix} - Precision-Recall Curves')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. 임계값별 성능 변화
    plt.subplot(3, 2, 4)
    thresholds = np.linspace(0, 1, 100)
    metrics_orig = {'precision': [], 'recall': [], 'f1': []}
    metrics_synth = {'precision': [], 'recall': [], 'f1': []}
    
    for threshold in thresholds:
        y_pred_orig = (y_scores[:len(y)] >= threshold).astype(int)
        y_pred_synth = (y_scores[len(y):] >= threshold).astype(int)
        
        metrics_orig['precision'].append(precision_score(y[:len(y)], y_pred_orig))
        metrics_orig['recall'].append(recall_score(y[:len(y)], y_pred_orig))
        metrics_orig['f1'].append(f1_score(y[:len(y)], y_pred_orig))
        
        metrics_synth['precision'].append(precision_score(y_resampled[len(y):], y_pred_synth))
        metrics_synth['recall'].append(recall_score(y_resampled[len(y):], y_pred_synth))
        metrics_synth['f1'].append(f1_score(y_resampled[len(y):], y_pred_synth))
    
    for metric in ['precision', 'recall', 'f1']:
        plt.plot(thresholds, metrics_orig[metric], label=f'Original {metric.capitalize()}')
        plt.plot(thresholds, metrics_synth[metric], label=f'Synthetic {metric.capitalize()}')
    
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
    
    # 원본/합성 샘플의 이상 점수 분포 통계
    orig_stats = stats.describe(y_scores[:len(y)])
    synth_stats = stats.describe(y_scores[len(y):])
    
    stats_data = {
        'Original': [orig_stats.mean, orig_stats.variance, orig_stats.skewness, orig_stats.kurtosis],
        'Synthetic': [synth_stats.mean, synth_stats.variance, synth_stats.skewness, synth_stats.kurtosis]
    }
    
    x = np.arange(4)
    width = 0.35
    
    plt.bar(x - width/2, stats_data['Original'], width, label='Original')
    plt.bar(x + width/2, stats_data['Synthetic'], width, label='Synthetic')
    plt.xticks(x, ['Mean', 'Variance', 'Skewness', 'Kurtosis'])
    plt.title(f'{title_prefix} - Statistical Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

# LSTM
plot_history(history, title_prefix="LSTM")

# 연속적 이상 점수화 및 ROC 커브 분석
y_scores = model.predict(X_resampled).flatten()

# Time Series Overlay 시각화
plot_time_series_overlay(y_scores, y_resampled, None, title="LSTM Anomaly Detection Over Time (BorderlineSMOTE)")

# BorderlineSMOTE 분석 시각화
plot_borderline_analysis(X_flat, y, y_scores, title_prefix="BorderlineSMOTE Analysis")

# 상세 BorderlineSMOTE 분석 시각화
plot_detailed_borderline_analysis(X_flat, y, y_scores, title_prefix="Detailed BorderlineSMOTE Analysis")

# 이상 점수 분포 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.kdeplot(data=y_scores[y_resampled == 0], label='Normal', fill=True, alpha=0.3)
sns.kdeplot(data=y_scores[y_resampled == 1][:len(y)], label='Original Anomalies', fill=True, alpha=0.3)
sns.kdeplot(data=y_scores[y_resampled == 1][len(y):], label='Synthetic Anomalies', fill=True, alpha=0.3)
plt.title('Anomaly Score Distribution (BorderlineSMOTE)')
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

print(f"F1 Score: {f1_score(y_resampled, y_pred_optimal)}")

plt.show()