import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import os
import glob
from scipy.stats import zscore
from scipy.fft import fft
import warnings

warnings.filterwarnings('ignore')

five_process_180sec = r'C:\\YS\\TUK\\S4E1\\생산시스템구축실무\\TeamProject\\Production_System_TeamProject\\data\\장비이상 조기탐지\\5공정_180sec'
five_process_fan_ok = r'C:\\YS\\TUK\\S4E1\\생산시스템구축실무\\TeamProject\\Production_System_TeamProject\\data\\장비이상 조기탐지\\FAN_sound_OK'
five_process_fan_error = r'C:\\YS\\TUK\\S4E1\\생산시스템구축실무\\TeamProject\\Production_System_TeamProject\\data\\장비이상 조기탐지\\FAN_sound_error'
all_csv_files = glob.glob(os.path.join(five_process_180sec, '*.csv'))
csv_files = [f for f in all_csv_files if 'Error Lot list' not in os.path.basename(f)]

dataframes = []
for file in csv_files:
    df = pd.read_csv(file)
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    df['source_file'] = os.path.basename(file)
    dataframes.append(df) 
five_process_180sec_merged = pd.concat(dataframes, ignore_index=True)
five_process_180sec_merged.sort_values(by='datetime', inplace=True)
five_process_180sec_merged.reset_index(drop=True, inplace=True)
five_process_180sec_error_lot = pd.read_csv(os.path.join(five_process_180sec, 'Error Lot list.csv'))
fan_ok_files = glob.glob(os.path.join(five_process_fan_ok, '*.wav'))
fan_error_files = glob.glob(os.path.join(five_process_fan_error, '*.wav'))

def load_numerical_data(file_path):
    df = pd.read_csv(file_path)
    return df

def summarize_dataset(df):
    summary = {
        "행 수": df.shape[0],
        "열 수": df.shape[1],
        "결측치 수": df.isnull().sum().sum(),
        "결측치 비율": df.isnull().mean().mean()
    }
    return summary, df.describe(include='all')

def plot_time_series(df, time_col='Time', temp_col='Temp', current_col='Current'):
    plt.figure(figsize=(15, 6))
    plt.suptitle('시간에 따른 온도와 전류 변화', fontsize=16)
    
    plt.subplot(1, 2, 1)
    sns.lineplot(data=df, x=time_col, y=temp_col)
    plt.title("온도 변화")
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.lineplot(data=df, x=time_col, y=current_col)
    plt.title("전류 변화")
    plt.xticks(rotation=45)
    plt.tight_layout()

def detect_outliers(df, col):
    threshold = 3
    df['z_score'] = zscore(df[col].fillna(df[col].mean()))
    outliers = df[np.abs(df['z_score']) > threshold]
    return outliers

def compare_fault_normal(df, error_lots, time_col='Time'):
    # 한글 시간 형식을 처리하기 위한 함수
    def convert_korean_time(time_str):
        if '오전' in time_str:
            time_str = time_str.replace('오전', 'AM')
        elif '오후' in time_str:
            time_str = time_str.replace('오후', 'PM')
        return pd.to_datetime(time_str, format='%p %I:%M:%S.%f')
    
    # 시간 데이터 변환
    df[time_col] = df[time_col].apply(convert_korean_time)
    error_times = pd.to_datetime(error_lots.iloc[:, 0])
    
    # 고장 여부 표시
    df['fault'] = df[time_col].isin(error_times)
    
    # 결과 반환
    return df.groupby('fault')[['Temp', 'Current']].agg(['mean', 'std'])

def plot_correlation(df):
    plt.figure(figsize=(8, 6))
    corr = df[['Temp', 'Current']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("온도-전류 상관관계", pad=20)
    plt.tight_layout()

def analyze_audio(normal_file, error_file):
    plt.figure(figsize=(20, 15))
    plt.suptitle('정상/에러 오디오 파일 비교 분석', fontsize=16)
    
    y_normal, sr_normal = librosa.load(normal_file)
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(y_normal, sr=sr_normal)
    plt.title('OK Waveform')
    
    plt.subplot(3, 2, 3)
    D_normal = np.abs(librosa.stft(y_normal))
    librosa.display.specshow(librosa.amplitude_to_db(D_normal, ref=np.max),
                           y_axis='log', x_axis='time')
    plt.title('OK Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 2, 5)
    yf_normal = fft(y_normal)
    plt.plot(np.abs(yf_normal[:len(yf_normal)//2]))
    plt.title("OK FFT")
    plt.xlabel("Frequency bin")
    plt.ylabel("Magnitude")
    
    y_error, sr_error = librosa.load(error_file)
    
    plt.subplot(3, 2, 2)
    librosa.display.waveshow(y_error, sr=sr_error)
    plt.title('Error Waveform')
    
    plt.subplot(3, 2, 4)
    D_error = np.abs(librosa.stft(y_error))
    librosa.display.specshow(librosa.amplitude_to_db(D_error, ref=np.max),
                           y_axis='log', x_axis='time')
    plt.title('Error Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 2, 6)
    yf_error = fft(y_error)
    plt.plot(np.abs(yf_error[:len(yf_error)//2]))
    plt.title("Error FFT")
    plt.xlabel("Frequency bin")
    plt.ylabel("Magnitude")
    
    plt.tight_layout()

plt.style.use('seaborn')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("========== 1. 데이터 불러오기 ==========")
numerical_data = five_process_180sec_merged
error_data = five_process_180sec_error_lot
print("데이터 불러오기 완료\n")

print("========== 2. 기본 요약 정보 ==========")
summary, desc = summarize_dataset(numerical_data)
print(summary)
print(desc)
print("기본 요약 정보 완료\n")

# print("========== 3. 시계열 시각화 ==========")
# plot_time_series(numerical_data)
# print("시계열 시각화 완료\n")

print("========== 4. 이상치 탐지 ==========")
outliers = detect_outliers(numerical_data, 'Temp')
print(outliers.head())
print("이상치 탐지 완료\n")

print("========== 5. 고장 구간과 정상 구간 비교 ==========")
result = compare_fault_normal(numerical_data, error_data)
print(result)
print("고장 구간과 정상 구간 비교 완료\n")

print("========== 6. 변수 간 상관관계 분석 ==========")
plot_correlation(numerical_data)
print("변수 간 상관관계 분석 완료\n")

print("========== 7. 음성 파일 분석 ==========")
analyze_audio(fan_ok_files[0], fan_error_files[0])
print("음성 파일 분석 완료\n")

plt.show()