"""
예측 모듈
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
import logging
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyPredictor:
    """이상치 예측을 위한 클래스"""
    
    def __init__(self, model_dir: str = 'models'):
        """
        Args:
            model_dir (str): 모델과 스케일러가 저장된 디렉토리 경로
        """
        self.model_dir = model_dir
        self.model = None
        self.scalers = []
        self.seq_len = 10  # 시퀀스 길이
        self.threshold = 0.5  # 이상치 판단 임계값
        
    def load_models(self) -> None:
        """모델과 스케일러를 로드합니다."""
        try:
            # 모델 로드
            model_path = os.path.join(self.model_dir, 'prediction_model.h5')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            self.model = load_model(model_path)
            logger.info("모델 로드 완료")
            
            # 스케일러 로드
            scaler_files = [f for f in os.listdir(self.model_dir) if f.endswith('_scaler.pkl')]
            if not scaler_files:
                raise FileNotFoundError("스케일러 파일을 찾을 수 없습니다.")
            
            self.scalers = []
            for scaler_file in scaler_files:
                scaler_path = os.path.join(self.model_dir, scaler_file)
                scaler = joblib.load(scaler_path)
                self.scalers.append(scaler)
            logger.info(f"{len(self.scalers)}개의 스케일러 로드 완료")
            
        except Exception as e:
            logger.error(f"모델/스케일러 로드 실패: {str(e)}")
            raise
    
    def prepare_sequence(self, data: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
        """
        데이터를 시퀀스 형태로 변환합니다.
        
        Args:
            data (pd.DataFrame): 입력 데이터
            scaler (StandardScaler): 스케일러 객체
            
        Returns:
            np.ndarray: 시퀀스 데이터
        """
        try:
            if len(data) < self.seq_len:
                raise ValueError(f"데이터 길이가 {self.seq_len}보다 작습니다.")
            
            # 스케일링
            scaled_data = scaler.transform(data[['Temp', 'Current']].values)
            
            # 시퀀스 생성
            X = []
            for i in range(len(data) - self.seq_len + 1):
                X.append(scaled_data[i:i+self.seq_len])
            return np.array(X)
            
        except Exception as e:
            logger.error(f"시퀀스 데이터 준비 실패: {str(e)}")
            raise
    
    def format_prediction_result(self, 
                               predictions: np.ndarray, 
                               data: pd.DataFrame,
                               last_probability: float) -> Dict[str, Any]:
        """
        예측 결과를 포맷팅합니다.
        
        Args:
            predictions (np.ndarray): 예측 확률 배열
            data (pd.DataFrame): 원본 데이터
            last_probability (float): 마지막 시퀀스의 예측 확률
            
        Returns:
            Dict[str, Any]: 포맷팅된 예측 결과
        """
        try:
            # 마지막 시퀀스의 데이터 추출
            last_sequence = data.iloc[-self.seq_len:]
            
            result = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'anomaly_probability': float(last_probability),
                'is_anomaly': bool(last_probability >= self.threshold),
                'last_sequence': {
                    'start_time': last_sequence['datetime'].iloc[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end_time': last_sequence['datetime'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'avg_temperature': float(last_sequence['Temp'].mean()),
                    'avg_current': float(last_sequence['Current'].mean()),
                    'process_id': int(last_sequence['Process'].iloc[-1])
                },
                'prediction_history': {
                    'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in data['datetime'].iloc[-len(predictions):]],
                    'probabilities': [float(p[0]) for p in predictions]
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"예측 결과 포맷팅 실패: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        새로운 데이터에 대한 이상치 확률을 예측합니다.
        
        Args:
            data (pd.DataFrame): 예측할 데이터
            
        Returns:
            Tuple[np.ndarray, float, Dict[str, Any]]: 
                (전체 시퀀스에 대한 예측 확률, 마지막 시퀀스의 예측 확률, 포맷팅된 결과)
        """
        try:
            if self.model is None or not self.scalers:
                self.load_models()
            
            all_predictions = []
            
            # 각 스케일러를 사용하여 예측
            for scaler in self.scalers:
                X = self.prepare_sequence(data, scaler)
                predictions = self.model.predict(X)
                all_predictions.append(predictions)
            
            # 모든 스케일러를 통한 예측 평균
            final_predictions = np.mean(all_predictions, axis=0)
            
            # 마지막 시퀀스의 예측 확률
            last_probability = final_predictions[-1][0]
            
            # 결과 포맷팅
            formatted_result = self.format_prediction_result(final_predictions, data, last_probability)
            
            logger.info(f"예측 완료: 마지막 시퀀스의 이상치 확률 = {last_probability*100:.2f}%")
            return final_predictions, last_probability, formatted_result
            
        except Exception as e:
            logger.error(f"예측 실패: {str(e)}")
            raise
    
    def predict_batch(self, data_list: List[pd.DataFrame]) -> List[Tuple[np.ndarray, float, Dict[str, Any]]]:
        """
        여러 데이터에 대한 예측을 수행합니다.
        
        Args:
            data_list (List[pd.DataFrame]): 예측할 데이터 리스트
            
        Returns:
            List[Tuple[np.ndarray, float, Dict[str, Any]]]: 각 데이터에 대한 예측 결과 리스트
        """
        try:
            results = []
            for i, data in enumerate(data_list):
                logger.info(f"데이터 {i+1}/{len(data_list)} 예측 중...")
                result = self.predict(data)
                results.append(result)
            return results
            
        except Exception as e:
            logger.error(f"배치 예측 실패: {str(e)}")
            raise
    
    def predict_realtime(self, data_stream: pd.DataFrame, window_size: int = 100) -> Dict[str, Any]:
        """
        실시간 데이터 스트림에 대한 예측을 수행합니다.
        
        Args:
            data_stream (pd.DataFrame): 실시간 데이터 스트림
            window_size (int): 슬라이딩 윈도우 크기
            
        Returns:
            Dict[str, Any]: 포맷팅된 예측 결과
        """
        try:
            # 최근 데이터만 사용
            recent_data = data_stream.tail(window_size)
            
            # 예측 수행
            _, last_prob, formatted_result = self.predict(recent_data)
            
            return formatted_result
            
        except Exception as e:
            logger.error(f"실시간 예측 실패: {str(e)}")
            raise

def create_predictor(model_dir: str = 'models') -> AnomalyPredictor:
    """
    예측기 객체를 생성합니다.
    
    Args:
        model_dir (str): 모델 디렉토리 경로
        
    Returns:
        AnomalyPredictor: 예측기 객체
    """
    return AnomalyPredictor(model_dir)

# 사용 예시
if __name__ == "__main__":
    # 예측기 생성
    predictor = create_predictor()
    
    # 새로운 데이터 예측
    # new_data = pd.DataFrame({
    #     'Date': ['2024-01-01'] * 10,
    #     'Time': ['10:00:00.000'] * 10,
    #     'datetime': pd.date_range(start='2024-01-01 10:00:00', periods=10, freq='1min'),
    #     'Index': range(10),
    #     'Process': [1] * 10,
    #     'Temp': [25.0] * 10,
    #     'Current': [1.0] * 10
    # })
    # predictions, last_prob, result = predictor.predict(new_data)
    # print(f"이상치 확률: {last_prob*100:.2f}%")
    # print("상세 결과:", result)
