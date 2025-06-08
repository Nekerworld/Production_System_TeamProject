"""
모델 관리 페이지
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
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_data_files, preprocess_data
from src.model import train
from src.utils.visualization import plot_training_history, plot_confusion_matrix

# 로깅 설정
import logging

def load_model_info():
    """모델 정보 로드"""
    model_dir = 'models'
    model_info = {
        'last_trained': None,
        'accuracy': None,
        'loss': None,
        'confusion_matrix': None,
        'training_history': None
    }
    
    # 모델 파일 확인
    model_path = os.path.join(model_dir, 'prediction_model.h5')
    if os.path.exists(model_path):
        model = load_model(model_path)
        model_info['model'] = model
        
        # 모델 메타데이터 확인
        metadata_path = os.path.join(model_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                model_info.update(metadata)
    
    return model_info

def show_model_status(model_info):
    """모델 상태 표시"""
    st.subheader("모델 상태")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "학습됨" if model_info['model'] is not None else "미학습"
        st.metric(
            label="모델 상태",
            value=status,
            delta=None
        )
    
    with col2:
        last_trained = model_info['last_trained'] or "없음"
        st.metric(
            label="마지막 학습",
            value=last_trained,
            delta=None
        )
    
    with col3:
        accuracy = f"{model_info['accuracy']*100:.1f}%" if model_info['accuracy'] else "N/A"
        st.metric(
            label="정확도",
            value=accuracy,
            delta=None
        )
    
    with col4:
        loss = f"{model_info['loss']:.4f}" if model_info['loss'] else "N/A"
        st.metric(
            label="손실값",
            value=loss,
            delta=None
        )

def show_training_history(model_info):
    """학습 히스토리 표시"""
    st.subheader("학습 히스토리")
    
    if model_info['training_history']:
        plot_training_history(
            [model_info['training_history']],
            output_dir='results',
            save=False
        )
    else:
        st.info("학습 히스토리가 없습니다.")

def show_model_performance(model_info):
    """모델 성능 지표 표시"""
    st.subheader("모델 성능")
    
    if model_info['confusion_matrix'] is not None:
        plot_confusion_matrix(
            model_info['confusion_matrix'],
            output_dir='results',
            save=False
        )
    else:
        st.info("성능 지표가 없습니다.")

def show_model_architecture(model_info):
    """모델 구조 표시"""
    st.subheader("모델 구조")
    
    if model_info['model'] is not None:
        model = model_info['model']
        model.summary(print_fn=lambda x: st.text(x))
    else:
        st.info("모델이 없습니다.")

def train_new_model():
    """새 모델 학습"""
    st.subheader("모델 재학습")
    
    # 학습 파라미터 설정
    col1, col2 = st.columns(2)
    
    with col1:
        window_width = st.number_input(
            "윈도우 너비",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        
        epochs = st.number_input(
            "에포크 수",
            min_value=1,
            max_value=100,
            value=10,
            step=1
        )
    
    with col2:
        batch_size = st.number_input(
            "배치 크기",
            min_value=8,
            max_value=128,
            value=32,
            step=8
        )
        
        learning_rate = st.number_input(
            "학습률",
            min_value=0.0001,
            max_value=0.1,
            value=0.001,
            step=0.0001,
            format="%.4f"
        )
    
    # 학습 시작 버튼
    if st.button("학습 시작"):
        with st.spinner("모델 학습 중..."):
            try:
                # 모델 학습
                history, model, metrics = train_model(
                    window_width=window_width,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )
                
                # 학습 결과 저장
                model_info = {
                    'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'accuracy': metrics['accuracy'],
                    'loss': metrics['loss'],
                    'confusion_matrix': metrics['confusion_matrix'],
                    'training_history': history.history
                }
                
                # 메타데이터 저장
                os.makedirs('models', exist_ok=True)
                with open('models/model_metadata.json', 'w') as f:
                    json.dump(model_info, f)
                
                st.success("모델 학습이 완료되었습니다!")
                st.experimental_rerun()
                
            except Exception as e:
                st.error(f"학습 중 오류가 발생했습니다: {str(e)}")

def main():
    """메인 함수"""
    st.title("모델 관리")
    
    # 모델 정보 로드
    model_info = load_model_info()
    
    # 모델 상태 표시
    show_model_status(model_info)
    
    # 모델 구조 표시
    show_model_architecture(model_info)
    
    # 학습 히스토리 표시
    show_training_history(model_info)
    
    # 모델 성능 지표 표시
    show_model_performance(model_info)
    
    # 모델 재학습
    train_new_model()

if __name__ == "__main__":
    main() 