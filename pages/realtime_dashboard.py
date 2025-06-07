"""
실시간 대시보드 페이지
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import predict_anomaly_probability
from src.utils.visualization import create_visualizer, create_dashboard_widgets

# 페이지 설정
st.set_page_config(
    page_title="실시간 대시보드",
    page_icon="📊",
    layout="wide"
)

def generate_sample_data(n_points: int = 100) -> pd.DataFrame:
    """샘플 데이터 생성"""
    now = datetime.now()
    dates = [now - timedelta(minutes=i) for i in range(n_points)]
    dates.reverse()
    
    return pd.DataFrame({
        'datetime': dates,
        'Temp': np.random.normal(25, 2, n_points),
        'Current': np.random.normal(1, 0.2, n_points)
    })

def update_metrics():
    """메트릭 업데이트"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="현재 이상치 확률",
            value="15.2%",
            delta="2.1%"
        )
    
    with col2:
        st.metric(
            label="오늘 감지된 이상",
            value="3",
            delta="-1"
        )
    
    with col3:
        st.metric(
            label="모델 정확도",
            value="92.5%",
            delta="0.5%"
        )
    
    with col4:
        st.metric(
            label="평균 응답 시간",
            value="0.8초",
            delta="-0.2초"
        )

def plot_realtime_data(data: pd.DataFrame):
    """실시간 데이터 시각화"""
    fig = make_subplots(rows=2, cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('Temperature', 'Current'))
    
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
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Real-time Sensor Data"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_alerts():
    """알림 표시"""
    st.subheader("최근 알림")
    
    alerts = [
        {
            'time': '2024-01-01 10:00:00',
            'type': '경고',
            'message': '온도가 임계값을 초과했습니다.',
            'status': '확인 필요'
        },
        {
            'time': '2024-01-01 10:05:00',
            'type': '오류',
            'message': '전류 센서 오작동',
            'status': '긴급 조치 필요'
        }
    ]
    
    for alert in alerts:
        with st.expander(f"{alert['time']} - {alert['type']}"):
            st.write(f"메시지: {alert['message']}")
            st.write(f"상태: {alert['status']}")

def main():
    """메인 함수"""
    st.title("실시간 대시보드")
    
    # 메트릭 업데이트
    update_metrics()
    
    # 실시간 데이터 시각화
    st.subheader("실시간 센서 데이터")
    data = generate_sample_data()
    plot_realtime_data(data)
    
    # 알림 표시
    show_alerts()
    
    # 자동 새로고침
    st.empty()
    st.button("새로고침")

if __name__ == "__main__":
    main() 