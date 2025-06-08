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
import time
from plyer import notification
import threading
from queue import Queue
import json

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

# 세션 상태 초기화
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'last_check' not in st.session_state:
    st.session_state.last_check = datetime.now()
if 'system_status' not in st.session_state:
    st.session_state.system_status = "정상"

# 알림 큐
alert_queue = Queue()

def send_desktop_notification(title: str, message: str):
    """데스크톱 알림 전송"""
    try:
        notification.notify(
            title=title,
            message=message,
            app_icon=None,
            timeout=10,
        )
    except Exception as e:
        st.error(f"알림 전송 실패: {str(e)}")

def monitor_system(data: pd.DataFrame):
    """시스템 상태 모니터링"""
    # 온도 임계값 체크
    temp_threshold = 30
    current_threshold = 1.5
    
    latest_temp = data['Temp'].iloc[-1]
    latest_current = data['Current'].iloc[-1]
    
    if latest_temp > temp_threshold:
        alert = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': '경고',
            'message': f'온도가 임계값을 초과했습니다. (현재: {latest_temp:.1f}°C)',
            'status': '확인 필요'
        }
        alert_queue.put(alert)
        st.session_state.system_status = "경고"
        send_desktop_notification("온도 경고", f"온도가 {temp_threshold}°C를 초과했습니다.")
    
    if latest_current > current_threshold:
        alert = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': '오류',
            'message': f'전류가 임계값을 초과했습니다. (현재: {latest_current:.2f}A)',
            'status': '긴급 조치 필요'
        }
        alert_queue.put(alert)
        st.session_state.system_status = "오류"
        send_desktop_notification("전류 오류", f"전류가 {current_threshold}A를 초과했습니다.")

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

def update_metrics(data: pd.DataFrame):
    """메트릭 업데이트"""
    # 이상치 확률 계산
    try:
        anomaly_prob = predict_anomaly_probability(data)
        if isinstance(anomaly_prob, np.ndarray) and len(anomaly_prob) > 0:
            current_prob = float(anomaly_prob[-1][0]) * 100
        else:
            current_prob = 0.0
            st.warning("예측 결과를 처리할 수 없습니다.")
    except Exception as e:
        current_prob = 0.0
        st.error(f"예측 중 오류 발생: {str(e)}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="현재 이상치 확률",
            value=f"{current_prob:.1f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            label="오늘 감지된 이상",
            value=str(len(st.session_state.alerts)),
            delta=None
        )
    
    with col3:
        st.metric(
            label="모델 정확도",
            value="92.5%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="평균 응답 시간",
            value="0.8초",
            delta=None
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
    
    # 임계값 선 추가
    fig.add_hline(y=30, line_dash="dash", line_color="red",
                 annotation_text="Temperature Threshold", row=1, col=1)
    fig.add_hline(y=1.5, line_dash="dash", line_color="red",
                 annotation_text="Current Threshold", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Real-time Sensor Data"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_alerts():
    """알림 표시"""
    st.subheader("최근 알림")
    
    # 새로운 알림 처리
    while not alert_queue.empty():
        alert = alert_queue.get()
        st.session_state.alerts.insert(0, alert)
    
    # 알림 표시
    for alert in st.session_state.alerts:
        with st.expander(f"{alert['time']} - {alert['type']} ⚠️"):
            st.write(f"메시지: {alert['message']}")
            st.write(f"상태: {alert['status']}")
            if st.button("확인", key=f"confirm_{alert['time']}"):
                alert['status'] = "확인됨"
                st.experimental_rerun()

def show_system_status():
    """시스템 상태 표시"""
    status_colors = {
        "정상": "green",
        "경고": "orange",
        "오류": "red"
    }
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("시스템 상태")
    
    status = st.session_state.system_status
    color = status_colors.get(status, "gray")
    
    st.sidebar.markdown(
        f'<div style="color: {color}; font-size: 24px; text-align: center;">'
        f'{"🟢" if status == "정상" else "🟠" if status == "경고" else "🔴"} {status}</div>',
        unsafe_allow_html=True
    )

def main():
    """메인 함수"""
    st.title("실시간 대시보드")
    
    # 시스템 상태 표시
    show_system_status()
    
    # 실시간 데이터 생성 및 모니터링
    data = generate_sample_data()
    monitor_system(data)
    
    # 메트릭 업데이트
    update_metrics(data)
    
    # 실시간 데이터 시각화
    st.subheader("실시간 센서 데이터")
    plot_realtime_data(data)
    
    # 알림 표시
    show_alerts()
    
    # 자동 새로고침
    if st.button("새로고침"):
        st.experimental_rerun()
    
    # 5초마다 자동 새로고침
    time.sleep(5)
    st.experimental_rerun()

if __name__ == "__main__":
    main() 