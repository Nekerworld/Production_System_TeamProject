"""
시각화 모듈
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyVisualizer:
    """이상치 시각화를 위한 클래스"""
    
    def __init__(self, output_dir: str = 'results'):
        """
        Args:
            output_dir (str): 시각화 결과 저장 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_training_history(self, 
                            histories: List[Dict[str, List[float]]],
                            save: bool = True) -> None:
        """
        학습 히스토리를 시각화합니다.
        
        Args:
            histories (List[Dict[str, List[float]]]): 학습 히스토리 리스트
            save (bool): 그래프 저장 여부
        """
        try:
            plt.figure(figsize=(15, 5))
            
            # Accuracy plot
            plt.subplot(1, 2, 1)
            for i, hist in enumerate(histories):
                plt.plot(hist['accuracy'], alpha=0.3, label=f'Window {i+1} Train')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            
            # Loss plot
            plt.subplot(1, 2, 2)
            for i, hist in enumerate(histories):
                plt.plot(hist['loss'], alpha=0.3, label=f'Window {i+1} Train')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True)
            
            plt.tight_layout()
            
            if save:
                plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
                plt.close()
            else:
                plt.show()
                
            logger.info("학습 히스토리 시각화 완료")
            
        except Exception as e:
            logger.error(f"학습 히스토리 시각화 실패: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, 
                            cm: np.ndarray,
                            save: bool = True) -> None:
        """
        혼동 행렬을 시각화합니다.
        
        Args:
            cm (np.ndarray): 혼동 행렬
            save (bool): 그래프 저장 여부
        """
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            if save:
                plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
                plt.close()
            else:
                plt.show()
                
            logger.info("혼동 행렬 시각화 완료")
            
        except Exception as e:
            logger.error(f"혼동 행렬 시각화 실패: {str(e)}")
            raise
    
    def plot_anomaly_detection(self,
                             data: pd.DataFrame,
                             predictions: np.ndarray,
                             threshold: float = 0.5,
                             save: bool = True) -> None:
        """
        이상치 탐지 결과를 시각화합니다.
        
        Args:
            data (pd.DataFrame): 원본 데이터
            predictions (np.ndarray): 예측 확률
            threshold (float): 이상치 판단 임계값
            save (bool): 그래프 저장 여부
        """
        try:
            # Plotly를 사용한 인터랙티브 시각화
            fig = make_subplots(rows=3, cols=1,
                              shared_xaxes=True,
                              vertical_spacing=0.05,
                              subplot_titles=('Temperature', 'Current', 'Anomaly Probability'))
            
            # 온도 데이터
            fig.add_trace(
                go.Scatter(x=data['datetime'], y=data['Temp'],
                          name='Temperature', line=dict(color='blue')),
                row=1, col=1
            )
            
            # 전류 데이터
            fig.add_trace(
                go.Scatter(x=data['datetime'], y=data['Current'],
                          name='Current', line=dict(color='green')),
                row=2, col=1
            )
            
            # 이상치 확률
            fig.add_trace(
                go.Scatter(x=data['datetime'], y=predictions,
                          name='Anomaly Probability', line=dict(color='red')),
                row=3, col=1
            )
            
            # 임계값 선
            fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                         annotation_text="Threshold", row=3, col=1)
            
            # 실제 이상치 표시
            anomaly_mask = data['is_anomaly'] == 1
            if anomaly_mask.any():
                anomaly_times = data.loc[anomaly_mask, 'datetime']
                for time in anomaly_times:
                    fig.add_vline(x=time, line_dash="dot", line_color="red",
                                annotation_text="Anomaly", row=1, col=1)
            
            # 레이아웃 설정
            fig.update_layout(
                height=900,
                showlegend=True,
                title_text="Anomaly Detection Results"
            )
            
            if save:
                fig.write_html(os.path.join(self.output_dir, 'anomaly_detection.html'))
            else:
                fig.show()
                
            logger.info("이상치 탐지 결과 시각화 완료")
            
        except Exception as e:
            logger.error(f"이상치 탐지 결과 시각화 실패: {str(e)}")
            raise
    
    def plot_performance_metrics(self,
                               metrics: Dict[str, float],
                               save: bool = True) -> None:
        """
        성능 지표를 시각화합니다.
        
        Args:
            metrics (Dict[str, float]): 성능 지표 딕셔너리
            save (bool): 그래프 저장 여부
        """
        try:
            # 성능 지표 추출
            labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [metrics['accuracy'], metrics['precision'],
                     metrics['recall'], metrics['f1_score']]
            
            # 막대 그래프 생성
            plt.figure(figsize=(10, 6))
            bars = plt.bar(labels, values)
            
            # 값 표시
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
            
            plt.title('Model Performance Metrics')
            plt.ylim(0, 1.1)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            if save:
                plt.savefig(os.path.join(self.output_dir, 'performance_metrics.png'))
                plt.close()
            else:
                plt.show()
                
            logger.info("성능 지표 시각화 완료")
            
        except Exception as e:
            logger.error(f"성능 지표 시각화 실패: {str(e)}")
            raise

def create_visualizer(output_dir: str = 'results') -> AnomalyVisualizer:
    """
    시각화기 객체를 생성합니다.
    
    Args:
        output_dir (str): 결과 저장 디렉토리
        
    Returns:
        AnomalyVisualizer: 시각화기 객체
    """
    return AnomalyVisualizer(output_dir)

# 사용 예시
if __name__ == "__main__":
    # 시각화기 생성
    visualizer = create_visualizer()
    
    # # 학습 히스토리 시각화 예시
    # histories = [
    #     {'accuracy': [0.8, 0.85, 0.9], 'loss': [0.5, 0.4, 0.3]},
    #     {'accuracy': [0.82, 0.87, 0.92], 'loss': [0.48, 0.38, 0.28]}
    # ]
    # visualizer.plot_training_history(histories)
    
    # # 혼동 행렬 시각화 예시
    # cm = np.array([[100, 10], [5, 85]])
    # visualizer.plot_confusion_matrix(cm)
    
    # # 이상치 탐지 결과 시각화 예시
    # data = pd.DataFrame({
    #     'datetime': pd.date_range(start='2024-01-01', periods=100, freq='1min'),
    #     'Temp': np.random.normal(25, 2, 100),
    #     'Current': np.random.normal(1, 0.2, 100),
    #     'is_anomaly': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    # })
    # predictions = np.random.random(100)
    # visualizer.plot_anomaly_detection(data, predictions)
    
    # # 성능 지표 시각화 예시
    # metrics = {
    #     'accuracy': 0.85,
    #     'precision': 0.82,
    #     'recall': 0.88,
    #     'f1_score': 0.85
    # }
    # visualizer.plot_performance_metrics(metrics)
