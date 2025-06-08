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
from pages import historical_analysis # 이력 데이터 분석 페이지
from pages import dashboard # 새로 추가된 대시보드 페이지 임포트

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
    
    # 시스템 상태 (realtime_dashboard에서 관리되는 session_state 사용)
    # 초기 상태가 설정되지 않은 경우를 대비
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
        f'{"🟢" if status == "정상" else "🟠" if status == "경고" else "🔴"} **{status}**</div>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown("---")

    # 메뉴 선택
    menu = st.sidebar.radio(
        "메뉴 선택",
        [
            "🏠 대시보드",
            "📊 실시간 대시보드",
            "📈 데이터 분석",
            "🔮 실시간 예측",
            "⚙️ 모델 관리",
            "🕰️ 이력 데이터 분석"
        ],
        key="main_menu_selection"
    )
    
    # 마지막 업데이트 시간
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return menu

# 메인 함수
def main():
    """메인 함수"""
    # 사이드바 설정 및 메뉴 선택
    menu = setup_sidebar()

    # 메뉴에 따른 페이지 표시
    if menu == "🏠 대시보드":
        dashboard.main()

    elif menu == "📊 실시간 대시보드":
        realtime_dashboard.main()

    elif menu == "📈 데이터 분석":
        data_analysis.main()

    elif menu == "🔮 실시간 예측":
        realtime_prediction.main()

    elif menu == "⚙️ 모델 관리":
        model_management.main()
    
    elif menu == "🕰️ 이력 데이터 분석":
        historical_analysis.main()

if __name__ == "__main__":
    main()
