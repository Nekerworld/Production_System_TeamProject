"""
모델 학습 모듈
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class _AnomalyModelTrainer:
    """이상치 탐지 모델 학습 클래스 (내부 구현)"""
    
    def __init__(self,
                 model_dir: str = 'models',
                 seq_len: int = 10,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.1):
        """
        Args:
            model_dir (str): 모델 저장 디렉토리
            seq_len (int): 시퀀스 길이
            train_ratio (float): 학습 데이터 비율
            val_ratio (float): 검증 데이터 비율
        """
        self.model_dir = model_dir
        self.seq_len = seq_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.model = None
        self.scaler = None
        
        os.makedirs(model_dir, exist_ok=True)
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        모델 아키텍처를 정의합니다.
        
        Args:
            input_shape (Tuple[int, int]): 입력 데이터 형태
            
        Returns:
            Sequential: 정의된 모델
        """
        try:
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
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info("모델 아키텍처 정의 완료")
            return model
            
        except Exception as e:
            logger.error(f"모델 아키텍처 정의 실패: {str(e)}")
            raise
    
    def prepare_sequences(self,
                         data: pd.DataFrame,
                         scaler: Optional[StandardScaler] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        시퀀스 데이터를 준비합니다.
        
        Args:
            data (pd.DataFrame): 원본 데이터
            scaler (Optional[StandardScaler]): 스케일러
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 입력 시퀀스와 레이블
        """
        try:
            # 특성 스케일링
            if scaler is None:
                scaler = StandardScaler()
                self.scaler = scaler
                features = scaler.fit_transform(data[['Temp', 'Current']].values)
            else:
                features = scaler.transform(data[['Temp', 'Current']].values)
            
            # 시퀀스 생성
            X, y = [], []
            for i in range(len(data) - self.seq_len):
                X.append(features[i:i+self.seq_len])
                y.append(1 if data['is_anomaly'].iloc[i:i+self.seq_len].any() else 0)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"시퀀스 데이터 준비 실패: {str(e)}")
            raise
    
    def split_data(self,
                  data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        데이터를 학습/검증/테스트 세트로 분할합니다.
        
        Args:
            data (pd.DataFrame): 원본 데이터
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 학습/검증/테스트 데이터
        """
        try:
            n_total = len(data)
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)
            
            train_data = data.iloc[:n_train]
            val_data = data.iloc[n_train:n_train + n_val]
            test_data = data.iloc[n_train + n_val:]
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"데이터 분할 실패: {str(e)}")
            raise
    
    def train(self,
              data: pd.DataFrame,
              epochs: int = 100,
              batch_size: int = 32,
              patience: int = 10) -> Dict[str, Any]:
        """
        모델을 학습합니다.
        
        Args:
            data (pd.DataFrame): 학습 데이터
            epochs (int): 학습 에포크 수
            batch_size (int): 배치 크기
            patience (int): 조기 종료 인내심
            
        Returns:
            Dict[str, Any]: 학습 결과
        """
        try:
            # 데이터 분할
            train_data, val_data, test_data = self.split_data(data)
            
            # 시퀀스 준비
            X_train, y_train = self.prepare_sequences(train_data)
            X_val, y_val = self.prepare_sequences(val_data, self.scaler)
            X_test, y_test = self.prepare_sequences(test_data, self.scaler)
            
            # 모델 생성
            if self.model is None:
                self.model = self.build_model((self.seq_len, 2))
            
            # 콜백 설정
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    os.path.join(self.model_dir, 'best_model.h5'),
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # 모델 학습
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # 스케일러 저장
            joblib.dump(self.scaler, os.path.join(self.model_dir, 'scaler.pkl'))
            
            # 테스트 세트 평가
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            
            results = {
                'history': history.history,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }
            
            logger.info(f"모델 학습 완료 (테스트 정확도: {test_accuracy:.4f})")
            return results
            
        except Exception as e:
            logger.error(f"모델 학습 실패: {str(e)}")
            raise
    
    def save_model(self, model_path: Optional[str] = None) -> None:
        """
        모델을 저장합니다.
        
        Args:
            model_path (Optional[str]): 모델 저장 경로
        """
        try:
            if model_path is None:
                model_path = os.path.join(self.model_dir, 'model.h5')
            
            self.model.save(model_path)
            logger.info(f"모델 저장 완료: {model_path}")
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {str(e)}")
            raise
    
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        모델을 로드합니다.
        
        Args:
            model_path (Optional[str]): 모델 로드 경로
        """
        try:
            if model_path is None:
                model_path = os.path.join(self.model_dir, 'model.h5')
            
            self.model = load_model(model_path)
            logger.info(f"모델 로드 완료: {model_path}")
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            raise

def train_model(data: pd.DataFrame,
                model_dir: str = 'models',
                seq_len: int = 10,
                train_ratio: float = 0.7,
                val_ratio: float = 0.1,
                epochs: int = 100,
                batch_size: int = 32,
                patience: int = 10) -> Dict[str, Any]:
    """
    이상치 탐지 모델을 학습합니다.
    
    Args:
        data (pd.DataFrame): 학습 데이터
        model_dir (str): 모델 저장 디렉토리
        seq_len (int): 시퀀스 길이
        train_ratio (float): 학습 데이터 비율
        val_ratio (float): 검증 데이터 비율
        epochs (int): 학습 에포크 수
        batch_size (int): 배치 크기
        patience (int): 조기 종료 인내심
        
    Returns:
        Dict[str, Any]: 학습 결과
    """
    try:
        trainer = _AnomalyModelTrainer(
            model_dir=model_dir,
            seq_len=seq_len,
            train_ratio=train_ratio,
            val_ratio=val_ratio
        )
        
        results = trainer.train(
            data=data,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        trainer.save_model()
        
        return results
        
    except Exception as e:
        logger.error(f"모델 학습 실패: {str(e)}")
        raise

# # 사용 예시
# if __name__ == "__main__":
#     # # 데이터 로드 예시
#     # data = pd.DataFrame({
#     #     'Temp': np.random.normal(25, 2, 1000),
#     #     'Current': np.random.normal(1, 0.2, 1000),
#     #     'is_anomaly': np.random.choice([0, 1], 1000, p=[0.9, 0.1])
#     # })
    
#     # # 모델 학습
#     # results = train_model(
#     #     data=data,
#     #     model_dir='models',
#     #     seq_len=10,
#     #     train_ratio=0.7,
#     #     val_ratio=0.1,
#     #     epochs=10,
#     #     batch_size=32,
#     #     patience=10
#     # )
