"""
데이터 로딩 및 전처리 모듈
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime
import logging
from glob import glob
from sklearn.preprocessing import StandardScaler

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_files(data_dir: str) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    데이터 디렉토리에서 모든 CSV 파일을 로드합니다.
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        
    Returns:
        Tuple[List[pd.DataFrame], pd.DataFrame]: (데이터프레임 리스트, 에러 데이터프레임)
    """
    try:
        # CSV 파일 경로 리스트 생성
        csv_paths = [p for p in glob(os.path.join(data_dir, '*.csv')) if
                    'Error Lot list' not in os.path.basename(p)]
        error_df = pd.read_csv(os.path.join(data_dir, 'Error Lot list.csv'))
        
        # 각 CSV 파일 로드
        dataframes = [load_one(p) for p in csv_paths]
        
        logger.info(f"총 {len(dataframes)}개의 CSV 파일 로드 완료")
        return dataframes, error_df
    except Exception as e:
        logger.error(f"데이터 파일 로드 실패: {str(e)}")
        raise

def load_one(file_path: str) -> pd.DataFrame:
    """
    단일 CSV 파일을 로드하고 전처리합니다.
    
    Args:
        file_path (str): CSV 파일 경로
        
    Returns:
        pd.DataFrame: 전처리된 데이터프레임
    """
    try:
        df = pd.read_csv(file_path)
        
        # 시간 데이터 처리
        df['Time'] = (df['Time'].str.replace('오전', 'AM')
                                .str.replace('오후', 'PM'))
        df['Time'] = pd.to_datetime(df['Time'], format='%p %I:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df['Index'] = df['Index'].astype(int)
        
        logger.info(f"파일 로드 완료: {os.path.basename(file_path)}")
        return df
    except Exception as e:
        logger.error(f"파일 로드 실패: {str(e)}")
        raise

def mark_anomaly(df: pd.DataFrame, error_df: pd.DataFrame) -> pd.DataFrame:
    """
    이상치 데이터를 마킹합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        error_df (pd.DataFrame): 이상치 정보가 담긴 데이터프레임
        
    Returns:
        pd.DataFrame: 이상치가 마킹된 데이터프레임
    """
    try:
        df['is_anomaly'] = 0
        for _, row in error_df.iterrows():
            date = str(row.iloc[0]).strip()
            procs = set(row.iloc[1:].dropna().astype(int))
            if procs:
                mask = (df['Date'] == date) & (df['Process'].isin(procs))
                df.loc[mask, 'is_anomaly'] = 1
        
        logger.info("이상치 마킹 완료")
        return df
    except Exception as e:
        logger.error(f"이상치 마킹 실패: {str(e)}")
        raise

def seq_generate(df: pd.DataFrame, scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray]:
    """
    시계열 데이터를 시퀀스로 변환합니다.
    
    Args:
        df (pd.DataFrame): 원본 데이터프레임
        scaler (StandardScaler): 스케일러 객체
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) 형태의 시퀀스 데이터
    """
    try:
        X, y = [], []
        feat = scaler.transform(df[['Temp', 'Current']].values)  # 스케일링
        lab = df['is_anomaly'].values
        
        for i in range(len(df) - 10):  # SEQ_LEN = 10
            X.append(feat[i:i+10])
            y.append(1 if lab[i:i+10].any() else 0)
            
        logger.info(f"시퀀스 데이터 준비 완료: {len(X)} 개의 시퀀스")
        return np.array(X), np.array(y)
    except Exception as e:
        logger.error(f"시퀀스 데이터 준비 실패: {str(e)}")
        raise

def prepare_window_data(
    dataframes: List[pd.DataFrame],
    error_df: pd.DataFrame,
    window_width: int,
    start_idx: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    윈도우 단위로 데이터를 준비합니다.
    
    Args:
        dataframes (List[pd.DataFrame]): 데이터프레임 리스트
        error_df (pd.DataFrame): 에러 데이터프레임
        window_width (int): 윈도우 너비
        start_idx (int): 시작 인덱스
        train_ratio (float): 학습 데이터 비율
        val_ratio (float): 검증 데이터 비율
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
            (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    try:
        # 윈도우 데이터 선택
        window_dfs = dataframes[start_idx:start_idx + window_width]
        combined = pd.concat(window_dfs, ignore_index=True)
        
        # 이상치 마킹
        combined = mark_anomaly(combined, error_df)
        
        # 데이터 분할
        n_total = len(combined)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_df = combined.iloc[:n_train]
        val_df = combined.iloc[n_train:n_train + n_val]
        test_df = combined.iloc[n_train + n_val:]
        
        # 스케일러 생성 및 적용
        scaler = StandardScaler().fit(train_df[['Temp', 'Current']])
        
        # 시퀀스 데이터 생성
        X_train, y_train = seq_generate(train_df, scaler)
        X_val, y_val = seq_generate(val_df, scaler)
        X_test, y_test = seq_generate(test_df, scaler)
        
        logger.info(f"윈도우 {start_idx+1}~{start_idx+window_width} 데이터 준비 완료")
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler
    except Exception as e:
        logger.error(f"윈도우 데이터 준비 실패: {str(e)}")
        raise
