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

def show_prediction_results(data: pd.DataFrame, predictions: np.ndarray):
    """예측 결과 표시"""
    st.subheader("예측 결과")
    
    # 예측 확률
    latest_prob = predictions[-1][0] * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="이상치 확률",
            value=f"{latest_prob:.1f}%",
            delta=None
        )
    
    with col2:
        status = "정상" if latest_prob < 50 else "이상"
        st.metric(
            label="상태",
            value=status,
            delta=None
        )
    
    with col3:
        confidence = "높음" if abs(latest_prob - 50) > 30 else "중간" if abs(latest_prob - 50) > 15 else "낮음"
        st.metric(
            label="신뢰도",
            value=confidence,
            delta=None
        )
    
    # 예측 결과 시각화
    plot_prediction_results(
        data=data,
        predictions=predictions,
        output_dir='results',
        save=False
    )

def show_prediction_details(data: pd.DataFrame, predictions: np.ndarray):
    """예측 상세 정보 표시"""
    st.subheader("예측 상세 정보")
    
    # 데이터 요약
    st.write("데이터 요약")
    summary = {
        "총 데이터 포인트": len(data),
        "시퀀스 길이": 10,
        "마지막 시퀀스 평균 온도": f"{data['Temp'].mean():.2f}°C",
        "마지막 시퀀스 평균 전류": f"{data['Current'].mean():.2f}A",
        "최근 예측 확률": f"{predictions[-1][0]*100:.1f}%"
    }
    
    for key, value in summary.items():
        st.write(f"- {key}: {value}")
    
    # 원시 데이터 표시
    if st.checkbox("원시 데이터 보기"):
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
        
        # 결과 저장
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': temperature,
            'current': current,
            'anomaly_probability': float(predictions[-1][0]),
            'status': "정상" if predictions[-1][0] < 0.5 else "이상"
        }
        
        # 결과 저장 버튼
        if st.button("결과 저장"):
            os.makedirs('results', exist_ok=True)
            with open('results/prediction_history.json', 'a') as f:
                json.dump(results, f)
                f.write('\n')
            st.success("결과가 저장되었습니다.")

if __name__ == "__main__":
    main() 