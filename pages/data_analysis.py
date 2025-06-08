"""
데이터 분석 페이지
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_data_files, preprocess_data # 데이터 로딩 및 전처리 함수 임포트

def load_and_prepare_all_data(data_dir: str = 'data/장비이상 조기탐지/5공정_180sec') -> pd.DataFrame:
    """모든 데이터를 로드하고 전처리하여 단일 DataFrame으로 반환합니다."""
    st.info(f"데이터 디렉토리: {data_dir} 에서 데이터 로드 중...")
    dataframes, error_df = load_data_files(data_dir)
    
    if not dataframes:
        st.error("로드할 데이터 파일이 없습니다. 경로를 확인해주세요.")
        return pd.DataFrame()
    
    all_data = pd.concat(dataframes, ignore_index=True)
    st.success(f"총 {len(all_data)}개의 데이터 포인트 로드 완료.")
    
    st.info("데이터 전처리 중...")
    processed_data, _ = preprocess_data(all_data, error_df) # 여기서 scaler는 사용하지 않으므로 무시
    st.success("데이터 전처리 완료.")
    
    return processed_data

def show_data_summary(df: pd.DataFrame):
    """데이터 요약 통계를 표시합니다."""
    st.subheader("데이터 요약")
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("총 데이터 포인트", len(df))
        st.metric("이상치 개수", df['is_anomaly'].sum())
    with col2:
        st.metric("평균 온도", f"{df['Temp'].mean():.2f}°C")
        st.metric("평균 전류", f"{df['Current'].mean():.2f}A")
    
    st.write("### 통계량")
    st.dataframe(df.describe()[['Temp', 'Current', 'Process']])

def plot_distributions(df: pd.DataFrame):
    """주요 특징의 분포를 시각화합니다."""
    st.subheader("데이터 분포")
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        fig_temp = px.histogram(df, x='Temp', nbins=50, title='온도 분포')
        st.plotly_chart(fig_temp, use_container_width=True)
    with col2:
        fig_current = px.histogram(df, x='Current', nbins=50, title='전류 분포')
        st.plotly_chart(fig_current, use_container_width=True)

def plot_time_series(df: pd.DataFrame):
    """시계열 데이터를 시각화합니다."""
    st.subheader("시계열 데이터 (온도 & 전류)")
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
        
    # Plotly subplot 생성
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=('온도', '전류'))
    
    # 온도 데이터 추가
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Temp'], mode='lines', name='온도', 
                             line=dict(color='blue')),
                  row=1, col=1)
    
    # 이상치 표시 (온도)
    anomaly_temp = df[df['is_anomaly'] == 1]
    if not anomaly_temp.empty:
        fig.add_trace(go.Scatter(x=anomaly_temp['datetime'], y=anomaly_temp['Temp'], mode='markers',
                                 name='이상치 (온도)', marker=dict(color='red', size=8, symbol='x')),
                      row=1, col=1)

    # 전류 데이터 추가
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['Current'], mode='lines', name='전류',
                             line=dict(color='green')),
                  row=2, col=1)
    
    # 이상치 표시 (전류)
    anomaly_current = df[df['is_anomaly'] == 1]
    if not anomaly_current.empty:
        fig.add_trace(go.Scatter(x=anomaly_current['datetime'], y=anomaly_current['Current'], mode='markers',
                                 name='이상치 (전류)', marker=dict(color='red', size=8, symbol='x')),
                      row=2, col=1)
    
    fig.update_layout(height=700, title_text='시계열 데이터 및 이상치', hovermode='x unified')
    fig.update_xaxes(title_text="시간", row=2, col=1)
    fig.update_yaxes(title_text="온도 (°C)", row=1, col=1)
    fig.update_yaxes(title_text="전류 (A)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df: pd.DataFrame):
    """특징 간의 상관 관계를 시각화합니다."""
    st.subheader("상관 관계 분석")
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    
    # 상관 관계 계산
    # Process 컬럼은 범주형일 수 있으므로 상관 관계 계산에서 제외하거나 원-핫 인코딩 고려
    # 여기서는 숫자형 특징만 포함
    numeric_cols = ['Temp', 'Current', 'is_anomaly']
    correlation_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='Viridis',
                    title='특징 간 상관 관계')
    st.plotly_chart(fig, use_container_width=True)

def main():
    """메인 함수"""
    st.title("📈 데이터 분석")
    st.markdown("이 페이지에서는 시스템에 사용되는 데이터의 특성을 분석하고 시각화합니다.")
    
    # 데이터 로드 및 전처리
    all_processed_data = load_and_prepare_all_data()
    
    if not all_processed_data.empty:
        # 데이터 요약
        show_data_summary(all_processed_data)
        
        st.markdown("---")
        
        # 데이터 분포 시각화
        plot_distributions(all_processed_data)
        
        st.markdown("---")
        
        # 시계열 데이터 시각화
        plot_time_series(all_processed_data)
        
        st.markdown("---")
        
        # 상관 관계 분석
        plot_correlation_matrix(all_processed_data)
        
        st.markdown("---")
        
        # 원시 데이터 보기
        st.subheader("원시 데이터")
        if st.checkbox("전체 원시 데이터 보기"):
            st.dataframe(all_processed_data)
    else:
        st.warning("데이터를 로드하거나 전처리하는 데 문제가 발생하여 분석을 수행할 수 없습니다.")

if __name__ == "__main__":
    main() 