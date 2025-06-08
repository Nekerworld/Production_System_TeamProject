"""
실시간 예측 페이지
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json
from typing import Dict, Any

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import predict_anomaly_probability
from src.utils.visualization import plot_prediction_results

# 페이지 설정
st.set_page_config(
    page_title="실시간 예측",
    page_icon="🔮",
    layout="wide"
)

def create_input_form():
    """입력 폼 생성"""
    st.subheader("데이터 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.number_input(
            "온도 (°C)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.1
        )
    
    with col2:
        current = st.number_input(
            "전류 (A)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
    
    return temperature, current

def generate_sequence_data(temperature: float, current: float, n_points: int = 10) -> pd.DataFrame:
    """시퀀스 데이터 생성"""
    now = datetime.now()
    dates = [now - timedelta(minutes=i) for i in range(n_points)]
    dates.reverse()
    
    # 실제 값 주변에 약간의 변동 추가
    temps = np.random.normal(temperature, 0.5, n_points)
    currents = np.random.normal(current, 0.1, n_points)
    
    return pd.DataFrame({
        'datetime': dates,
        'Temp': temps,
        'Current': currents
    })

def show_prediction_results(data: pd.DataFrame, predictions: Dict[str, Any]):
    """예측 결과 표시"""
    st.subheader("예측 결과")
    
    # 예측 결과 처리
    try:
        latest_prob = predictions['anomaly_percentage']
        status = "이상" if predictions['is_anomaly'] else "정상"
    except Exception as e:
        latest_prob = 0.0
        status = "정상"
        st.error(f"예측 결과 처리 중 오류 발생: {str(e)}")
    
    # 결과 표시
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="이상치 확률",
            value=f"{latest_prob:.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            label="상태",
            value=status,
            delta=None
        )
    
    # 시각화
    st.subheader("예측 결과 시각화")
    plot_prediction_results(
        data=data,
        predictions=predictions['predictions'],
        output_dir='results',
        save=False
    )
    
    # 상세 정보
    st.subheader("상세 정보")
    summary = predictions['data_summary']
    st.write({
        "총 데이터 포인트": summary['total_points'],
        "시퀀스 길이": summary['sequence_length'],
        "평균 온도": f"{summary['last_sequence']['avg_temperature']:.2f}°C",
        "평균 전류": f"{summary['last_sequence']['avg_current']:.2f}A",
        "시작 시간": summary['last_sequence']['start_time'],
        "종료 시간": summary['last_sequence']['end_time']
    })
    
    # 원시 데이터 확인
    if st.checkbox("원시 데이터 보기"):
        st.dataframe(data)
    
    # 결과 저장
    if st.button("결과 저장"):
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_data': {
                'temp': data['Temp'].tolist(),
                'current': data['Current'].tolist()
            },
            'anomaly_probability': float(latest_prob),
            'status': status,
            'confidence_level': predictions['confidence_level']
        }
        
        os.makedirs('results', exist_ok=True)
        with open(f'results/prediction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(result, f, indent=4)
        
        st.success("결과가 저장되었습니다.")

def show_prediction_details(data: pd.DataFrame, predictions: Dict[str, Any]) -> None:
    """예측 결과 상세 정보 표시"""
    st.subheader("예측 결과 상세")
    
    # 기본 정보 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "이상치 확률",
            f"{predictions['anomaly_percentage']:.1f}%",
            delta=None
        )
    with col2:
        st.metric(
            "상태",
            "이상" if predictions['is_anomaly'] else "정상",
            delta=None
        )
    with col3:
        st.metric(
            "신뢰도",
            predictions['confidence_level'],
            delta=None
        )
    
    # 데이터 요약 정보
    st.subheader("데이터 요약")
    summary = predictions['data_summary']
    st.write({
        "총 데이터 포인트": summary['total_points'],
        "시퀀스 길이": summary['sequence_length'],
        "평균 온도": f"{summary['last_sequence']['avg_temperature']:.2f}°C",
        "평균 전류": f"{summary['last_sequence']['avg_current']:.2f}A",
        "시작 시간": summary['last_sequence']['start_time'],
        "종료 시간": summary['last_sequence']['end_time']
    })
    
    # 원본 데이터 표시 옵션
    if st.checkbox("원본 데이터 보기"):
        st.dataframe(data)

def main():
    """메인 함수"""
    st.title("실시간 예측")
    
    # 입력 폼
    temperature, current = create_input_form()
    
    # 예측 실행 버튼
    if st.button("예측 실행"):
        # 시퀀스 데이터 생성
        data = generate_sequence_data(temperature, current)
        
        # 예측 수행
        predictions = predict_anomaly_probability(data)
        
        # 결과 표시
        show_prediction_results(data, predictions)
        
        # 상세 정보 표시
        show_prediction_details(data, predictions)

if __name__ == "__main__":
    main() 