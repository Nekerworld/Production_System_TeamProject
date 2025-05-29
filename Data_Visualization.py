import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set(font='Malgun Gothic')
sns.set_style("whitegrid")

# 데이터 경로 설정
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
    # 파일명을 기준으로 'source_file' 컬럼 생성
    df['source_file'] = os.path.basename(file)
    dataframes.append(df)
    
five_process_180sec_merged = pd.concat(dataframes, ignore_index=True)
five_process_180sec_merged.sort_values(by='datetime', inplace=True)
five_process_180sec_merged.reset_index(drop=True, inplace=True)
five_process_180sec_error_lot = pd.read_csv(os.path.join(five_process_180sec, 'Error Lot list.csv'))

# 라벨링
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

# 데이터 분포 시각화
plt.figure(figsize=(15, 10))

# 정상 데이터 1% 샘플링
normal_data = csv_data[csv_data['label'] == 0].sample(frac=0.01, random_state=42)
anomaly_data = csv_data[csv_data['label'] == 1]
sampled_data = pd.concat([normal_data, anomaly_data])

# 1. 온도와 전류의 산점도 (라벨별로 다른 색상)
plt.subplot(2, 2, 1)
sns.scatterplot(data=sampled_data, x='Temp', y='Current', hue='label', alpha=0.6)
plt.title('Temperature vs Current Distribution\n(Normal data sampled to 1%)')
plt.xlabel('Temperature')
plt.ylabel('Current')
plt.grid()

# 2. 온도 분포 (라벨별로 다른 색상)
plt.subplot(2, 2, 2)
sns.kdeplot(data=sampled_data, x='Temp', hue='label', fill=True, alpha=0.3)
plt.title('Temperature Distribution\n(Normal data sampled to 1%)')
plt.xlabel('Temperature')
plt.ylabel('Density')
plt.grid()

# 3. 전류 분포 (라벨별로 다른 색상)
plt.subplot(2, 2, 3)
sns.kdeplot(data=sampled_data, x='Current', hue='label', fill=True, alpha=0.3)
plt.title('Current Distribution\n(Normal data sampled to 1%)')
plt.xlabel('Current')
plt.ylabel('Density')
plt.grid()

# 4. 클래스 비율 파이 차트
plt.subplot(2, 2, 4)
class_counts = csv_data['label'].value_counts()
plt.pie(class_counts, labels=['Normal', 'Anomaly'], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Class Distribution')

plt.tight_layout()

# 클래스 비율 출력
print("\n클래스 분포:")
print(f"정상 데이터 수: {class_counts[0]}")
print(f"이상 데이터 수: {class_counts[1]}")
print(f"이상 데이터 비율: {(class_counts[1] / len(csv_data) * 100):.2f}%")

plt.show()
