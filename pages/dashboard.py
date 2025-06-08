"""
기본 대시보드 페이지
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가 (필요한 경우)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """대시보드 페이지 표시"""
    st.title("이상치 탐지 대시보드")
    
    # 상단 메트릭 카드
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
    
    # 실시간 데이터 차트
    st.subheader("실시간 센서 데이터")
    
    # 예시 데이터 생성
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1min')
    temp_data = np.random.normal(25, 2, 100)
    current_data = np.random.normal(1, 0.2, 100)
    
    # Plotly 차트 생성
    fig = make_subplots(rows=2, cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=('Temperature', 'Current'))
    
    fig.add_trace(
        go.Scatter(x=dates, y=temp_data,
                  name='Temperature', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=dates, y=current_data,
                  name='Current', line=dict(color='green')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Real-time Sensor Data"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 최근 알림
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

if __name__ == "__main__":
    main() 