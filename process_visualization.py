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

# 로컬에서 실행시 경로 알맞게 설정해주세요
five_process_180sec = r'C:\\YS\\TUK\\S4E1\\생산시스템구축실무\\TeamProject\\Production_System_TeamProject\\data\\장비이상 조기탐지\\5공정_180sec'

# 모든 csv 파일 목록을 가져옴 (Error Lot 제외)
all_csv_files = glob(os.path.join(five_process_180sec, '*.csv'))
csv_files = [f for f in all_csv_files if 'Error Lot list' not in os.path.basename(f)]

# ① 에러 Lot 리스트 읽기
error_df = pd.read_csv(os.path.join(five_process_180sec, 'Error Lot list.csv'))
# ② {날짜(str): {에러 Process, …}} 형태의 딕셔너리 생성
error_dict = {}
for _, row in error_df.iterrows():
    date = str(row.iloc[0]).strip()
    process_set = set(row.iloc[1:].dropna().astype(int))
    if len(process_set) > 0:
        error_dict[date] = process_set

dataframes = []

for file in csv_files:
    df = pd.read_csv(file)

    # 2-1) 한글 '오전/오후' → AM/PM 변환 + 24시간제로 통일
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

    # 2-5) ***vectorized*** 방식으로 is_anomaly 생성 (Process 기반)
    df['is_anomaly'] = 0  # 기본값
    for d, process_set in error_dict.items():
        mask = (df['Date'] == d) & (df['Process'].isin(process_set))
        df.loc[mask, 'is_anomaly'] = 1

    # 2-6) 리스트에 적재
    dataframes.append(df)
    
five_process_180sec_merged = pd.concat(dataframes, ignore_index=True)
five_process_180sec_merged.sort_values('datetime', inplace=True)
five_process_180sec_merged.reset_index(drop=True, inplace=True)

# is_anomaly를 label로 사용
csv_data = five_process_180sec_merged.copy()
csv_data['is_anomaly']

# Streamlit 앱 제목
st.title('Process별 Temperature vs Current 시각화')

# Process 선택을 위한 체크박스 생성
processes = csv_data['Process'].unique()
selected_processes = st.multiselect(
    '표시할 Process를 선택하세요:',
    processes,
    default=processes  # 기본적으로 모든 Process 선택
)

# 그래프 그리기
fig, ax = plt.subplots(figsize=(12, 8))

# 선택된 Process만 표시
for idx, process in enumerate(selected_processes):
    process_data = csv_data[csv_data['Process'] == process]
    
    # 산점도 그리기
    ax.scatter(process_data['Temp'], process_data['Current'], 
              c=[plt.cm.tab20(idx/len(processes))], 
              label=f'Process {process}', 
              alpha=0.5, marker='o')

ax.set_title('Temperature vs Current by Process', fontsize=14)
ax.set_xlabel('Temperature', fontsize=12)
ax.set_ylabel('Current', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Streamlit에 그래프 표시
st.pyplot(fig)