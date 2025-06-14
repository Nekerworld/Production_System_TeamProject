import numpy as np
import pandas as pd
import os
from glob import glob
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 경로 설정
DATA_DIR = r'C:\YS\TUK\S4E1\생산시스템구축실무\TeamProject\Production_System_TeamProject\data\장비이상 조기탐지\5공정_180sec'
csv_paths = [p for p in glob(os.path.join(DATA_DIR, '*.csv')) if
             'Error Lot list' not in os.path.basename(p)]
error_df   = pd.read_csv(os.path.join(DATA_DIR, 'Error Lot list.csv'))

# 파라미터
WINDOW_WIDTH  = 3    # 한 번에 묶을 CSV 개수
SLIDE_STEP    = 1    # Stride
SEQ_LEN       = 10   # LSTM 시계열 길이
TRAIN_RATIO   = 0.7
VAL_RATIO     = 0.1

def mark_anomaly(df, err):
    df['is_anomaly'] = 0
    for _, row in err.iterrows():
        date  = str(row.iloc[0]).strip()
        procs = set(row.iloc[1:].dropna().astype(int))
        if procs:
            mask = (df['Date'] == date) & (df['Process'].isin(procs))
            df.loc[mask, 'is_anomaly'] = 1
    return df

def load_one(path):
    df = pd.read_csv(path)
    df['Time'] = (df['Time'].str.replace('오전', 'AM')
                            .str.replace('오후', 'PM'))
    df['Time'] = pd.to_datetime(df['Time'], format='%p %I:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['Index'] = df['Index'].astype(int)
    df = mark_anomaly(df, error_df)
    return df

dataframes = [load_one(p) for p in csv_paths]

def seq_generate(df, scaler):
    X, y = [], []
    feat = scaler.transform(df[['Temp', 'Current']].values)  # 스케일링
    lab  = df['is_anomaly'].values
    for i in range(len(df) - SEQ_LEN):
        X.append(feat[i:i+SEQ_LEN])
        y.append(1 if lab[i:i+SEQ_LEN].any() else 0)
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        GRU(64),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# 모델 저장 디렉토리 생성
os.makedirs('models', exist_ok=True)

# 첫 번째 모델 생성
model = build_model((SEQ_LEN, 2))  # 2는 feature 수 (Temp, Current)
early_stopping = EarlyStopping(monitor='val_loss', patience=10,
                   restore_best_weights=True, verbose=1)

checkpoint = ModelCheckpoint(
    'models/prediction_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

y_true_all, y_pred_all = [], []
window_histories = []  # 각 윈도우의 학습 히스토리를 저장할 리스트

for start in range(0, len(dataframes) - WINDOW_WIDTH + 1, SLIDE_STEP):
    window_dfs = dataframes[start:start + WINDOW_WIDTH]
    combined   = pd.concat(window_dfs, ignore_index=True)

    n_total = len(combined)
    n_train = int(n_total * TRAIN_RATIO)
    n_val   = int(n_total * VAL_RATIO)

    train_df = combined.iloc[:n_train]
    val_df   = combined.iloc[n_train:n_train + n_val]
    test_df  = combined.iloc[n_train + n_val:]

    scaler = StandardScaler().fit(train_df[['Temp', 'Current']])

    X_train, y_train = seq_generate(train_df, scaler)
    X_val,   y_val   = seq_generate(val_df,   scaler)
    X_test,  y_test  = seq_generate(test_df,  scaler)

    # 모델 학습 (첫 번째 윈도우에서는 새로 학습, 이후에는 이어서 학습)
    history = model.fit(X_train, y_train, epochs=2, batch_size=32,
              validation_data=(X_val, y_val), 
              callbacks=[early_stopping, checkpoint], 
              verbose=1)
    
    window_histories.append(history.history)

    # 스케일러 저장
    joblib.dump(scaler, f'models/window_{start+1}_scaler.pkl')

    y_pred = (model.predict(X_test) >= 0.5).astype(int).ravel()

    print(f"\n{start+1}~{start+WINDOW_WIDTH}번 CSV 결과")
    # 실제 존재하는 클래스만 평가
    unique_classes = np.unique(np.concatenate([y_test, y_pred]))
    target_names = ['Normal', 'Anomaly']
    existing_target_names = [target_names[i] for i in unique_classes]
    
    print(classification_report(y_test, y_pred,
          labels=unique_classes,
          target_names=existing_target_names))

    y_true_all.extend(y_test)
    y_pred_all.extend(y_pred)

print("\n전체 누적 성능")
# 전체 결과에서도 실제 존재하는 클래스만 평가
unique_classes_all = np.unique(np.concatenate([y_true_all, y_pred_all]))
existing_target_names_all = [['Normal', 'Anomaly'][i] for i in unique_classes_all]

print(classification_report(y_true_all, y_pred_all,
      labels=unique_classes_all,
      target_names=existing_target_names_all))

# 학습 히스토리 시각화
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
for i, hist in enumerate(window_histories):
    plt.plot(hist['accuracy'], alpha=0.3, label=f'Window {i+1} Train')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
for i, hist in enumerate(window_histories):
    plt.plot(hist['loss'], alpha=0.3, label=f'Window {i+1} Train')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Confusion Matrix 시각화
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# 실시간 예측을 위한 함수들
def load_model_and_scalers():
    """모델과 모든 윈도우의 스케일러를 로드"""
    model = load_model('models/continuous_model.h5')
    scalers = []
    for i in range(len(dataframes) - WINDOW_WIDTH + 1):
        scaler_path = f'models/window_{i+1}_scaler.pkl'
        if os.path.exists(scaler_path):
            scalers.append(joblib.load(scaler_path))
    return model, scalers

def prepare_new_data(new_data, scaler):
    """새로운 데이터를 모델 입력 형태로 변환"""
    if len(new_data) < SEQ_LEN:
        raise ValueError(f"데이터 길이가 {SEQ_LEN}보다 작습니다.")
    
    # 스케일링
    scaled_data = scaler.transform(new_data[['Temp', 'Current']].values)
    
    # 시퀀스 생성
    X = []
    for i in range(len(new_data) - SEQ_LEN + 1):
        X.append(scaled_data[i:i+SEQ_LEN])
    return np.array(X)

def predict_anomaly_probability(new_data):
    """새로운 데이터에 대한 이상치 확률 예측"""
    model, scalers = load_model_and_scalers()
    if not scalers:
        raise ValueError("저장된 스케일러가 없습니다.")
    
    all_predictions = []
    
    for scaler in scalers:
        # 데이터 준비
        X = prepare_new_data(new_data, scaler)
        
        # 예측
        predictions = model.predict(X)
        all_predictions.append(predictions)
    
    # 모든 스케일러를 통한 예측 평균
    final_predictions = np.mean(all_predictions, axis=0)
    
    return final_predictions

# plt.show()

# # 예시 사용법
# if __name__ == "__main__":
#     # 모델 저장 디렉토리 생성
#     os.makedirs('models', exist_ok=True)
    
#     # 기존 학습 코드 실행
#     // ... existing training code ...
    
#     # 새로운 데이터 예측 예시
#     # new_data = pd.DataFrame({
#     #     'Temp': [...],  # 새로운 온도 데이터
#     #     'Current': [...]  # 새로운 전류 데이터
#     # })
#     # probabilities = predict_anomaly_probability(new_data)
#     # print(f"이상치 확률: {probabilities[-1][0]*100:.2f}%")