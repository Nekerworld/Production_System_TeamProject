"""
과거 데이터 분석 페이지
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
from glob import glob

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_data_files, preprocess_data
from src.utils.visualization import plot_prediction_results

# 페이지 설정 (app.py에서 전역으로 설정되므로 여기서는 제거합니다.)
#     page_title="이력 데이터 분석",
#     page_icon="🕰️",
#     layout="wide"
# )

def load_historical_data(data_dir: str = 'data/장비이상 조기탐지/5공정_180sec') -> pd.DataFrame:
    """과거 데이터 로드"""
    csv_paths = [p for p in glob(os.path.join(data_dir, '*.csv')) if
                 'Error Lot list' not in os.path.basename(p)]
    error_df = pd.read_csv(os.path.join(data_dir, 'Error Lot list.csv'))
    
    # 데이터 로드 및 전처리
    dataframes = []
    for path in csv_paths:
        df = pd.read_csv(path)
        df['Time'] = (df['Time'].str.replace('오전', 'AM')
                              .str.replace('오후', 'PM'))
        df['Time'] = pd.to_datetime(df['Time'], format='%p %I:%M:%S.%f').dt.strftime('%H:%M:%S.%f')
        df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df['Index'] = df['Index'].astype(int)
        dataframes.append(df)
    
    # 데이터 결합
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # 이상치 마킹
    combined_df['is_anomaly'] = 0
    for _, row in error_df.iterrows():
        date = str(row.iloc[0]).strip()
        procs = set(row.iloc[1:].dropna().astype(int))
        if procs:
            mask = (combined_df['Date'] == date) & (combined_df['Process'].isin(procs))
            combined_df.loc[mask, 'is_anomaly'] = 1
    
    return combined_df

def show_data_summary(data: pd.DataFrame):
    """데이터 요약 정보 표시"""
    st.subheader("데이터 요약")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="총 데이터 포인트",
            value=f"{len(data):,}",
            delta=None
        )
    
    with col2:
        anomaly_count = data['is_anomaly'].sum()
        st.metric(
            label="이상치 수",
            value=f"{anomaly_count:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="평균 온도",
            value=f"{data['Temp'].mean():.1f}°C",
            delta=None
        )
    
    with col4:
        st.metric(
            label="평균 전류",
            value=f"{data['Current'].mean():.2f}A",
            delta=None
        )

def plot_temperature_distribution(data: pd.DataFrame):
    """온도 분포 시각화"""
    st.subheader("온도 분포")
    
    fig = go.Figure()
    
    # 정상 데이터
    normal_data = data[data['is_anomaly'] == 0]
    fig.add_trace(go.Histogram(
        x=normal_data['Temp'],
        name='정상',
        opacity=0.7,
        marker_color='blue'
    ))
    
    # 이상 데이터
    anomaly_data = data[data['is_anomaly'] == 1]
    fig.add_trace(go.Histogram(
        x=anomaly_data['Temp'],
        name='이상',
        opacity=0.7,
        marker_color='red'
    ))
    
    fig.update_layout(
        title='온도 분포 (정상 vs 이상)',
        xaxis_title='온도 (°C)',
        yaxis_title='빈도',
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_current_distribution(data: pd.DataFrame):
    """전류 분포 시각화"""
    st.subheader("전류 분포")
    
    fig = go.Figure()
    
    # 정상 데이터
    normal_data = data[data['is_anomaly'] == 0]
    fig.add_trace(go.Histogram(
        x=normal_data['Current'],
        name='정상',
        opacity=0.7,
        marker_color='blue'
    ))
    
    # 이상 데이터
    anomaly_data = data[data['is_anomaly'] == 1]
    fig.add_trace(go.Histogram(
        x=anomaly_data['Current'],
        name='이상',
        opacity=0.7,
        marker_color='red'
    ))
    
    fig.update_layout(
        title='전류 분포 (정상 vs 이상)',
        xaxis_title='전류 (A)',
        yaxis_title='빈도',
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_time_series(data: pd.DataFrame):
    """시계열 데이터 시각화"""
    st.subheader("시계열 데이터")
    
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
    
    # 이상치 표시
    anomaly_data = data[data['is_anomaly'] == 1]
    fig.add_trace(
        go.Scatter(x=anomaly_data['datetime'], y=anomaly_data['Temp'],
                  mode='markers', name='Anomaly',
                  marker=dict(color='red', size=10)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=anomaly_data['datetime'], y=anomaly_data['Current'],
                  mode='markers', name='Anomaly',
                  marker=dict(color='red', size=10)),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Historical Sensor Data"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_correlation_analysis(data: pd.DataFrame):
    """상관관계 분석"""
    st.subheader("상관관계 분석")
    
    # 상관계수 계산
    corr = data[['Temp', 'Current', 'is_anomaly']].corr()
    
    # 히트맵 생성
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title='변수 간 상관관계',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """메인 함수"""
    st.title("과거 데이터 분석")
    
    # 데이터 로드
    data = load_historical_data()
    
    # 데이터 요약
    show_data_summary(data)
    
    # 분포 분석
    col1, col2 = st.columns(2)
    with col1:
        plot_temperature_distribution(data)
    with col2:
        plot_current_distribution(data)
    
    # 시계열 데이터
    plot_time_series(data)
    
    # 상관관계 분석
    show_correlation_analysis(data)
    
    # 원시 데이터 확인
    if st.checkbox("원시 데이터 보기"):
        st.dataframe(data)

if __name__ == "__main__":
    main() 