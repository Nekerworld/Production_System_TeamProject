"""
이상치 탐지 시스템 메인 페이지
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
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 페이지 모듈 임포트
from pages import realtime_dashboard
from pages import data_analysis
from pages import model_management
from pages import realtime_prediction
from pages import historical_analysis

# 페이지 설정
st.set_page_config(
    page_title="이상치 탐지 시스템",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 설정
def setup_sidebar():
    """사이드바 설정"""
    st.sidebar.title("이상치 탐지 시스템")
    st.sidebar.markdown("---")
    
    # 시스템 상태
    if 'system_status' not in st.session_state:
        st.session_state.system_status = "정상"

    status_colors = {
        "정상": "green",
        "경고": "orange",
        "오류": "red"
    }
    status = st.session_state.system_status
    color = status_colors.get(status, "gray")
    
    st.sidebar.subheader("시스템 상태")
    st.sidebar.markdown(
        f'<div style="color: {color}; font-size: 20px; text-align: center; margin-bottom: 20px;">'
        f'{"🟢" if status == "정상" else "🟠" if status == "경고" else "🔴"} {status}</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")
    
    # 마지막 업데이트 시간
    st.sidebar.markdown(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def show_dashboard():
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

# 메인 함수
def main():
    """메인 함수"""
    # 사이드바 설정
    setup_sidebar()
    
    # 대시보드 표시
    show_dashboard()

if __name__ == "__main__":
    main()
