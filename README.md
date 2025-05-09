# Production_System_TeamProject
2025년 1학기 생산시스템 구축실무 과목의 팀 프로젝트에 대한 레포지토리입니다.

## ⚙️ 개발 환경 및 실행 환경 (Environment)

이 프로젝트는 CNC 가공 공정 데이터를 기반으로 한 딥러닝 품질 예측 모델을 구현하며, 다음과 같은 소프트웨어 및 하드웨어 환경에서 수행되었습니다.

### 📌 Python 환경

* **Python 버전**: `3.7.16`
* **가상환경 추천**: `venv` 또는 `conda` 환경 사용 권장
* [Tensorflow GPU 환경 구축 가이드](https://youtu.be/M4urbN0fPyM?list=FLymXUrZPMX6J6TBv0ytj4jA)

### 📦 주요 패키지 및 라이브러리

| 라이브러리                   | 버전             | 용도                   |
| ----------------------- | -------------- | -------------------- |
| `tensorflow`            | 2.6.0          | DNN 모델 구축 및 학습       |
| `keras-preprocessing`   | 1.1.2          | 딥러닝 전처리 유틸           |
| `numpy`                 | 1.21.5         | 수치 연산                |
| `pandas`                | 1.3.5          | 데이터프레임 처리            |
| `scikit-learn`          | 1.0.2          | 전처리, 모델 평가 등         |
| `scipy`                 | 1.7.3          | 수학/통계 기반 함수          |
| `matplotlib`, `seaborn` | 3.5.3 / 0.12.2 | 시각화                  |
| `imbalanced-learn`      | 0.8.1          | 클래스 불균형 처리 (SMOTE 등) |
| `xgboost`               | 1.6.2          | 비교용 머신러닝 모델          |

추가적으로 `tensorboard`, `protobuf`, `h5py`, `joblib`, `graphviz` 등도 함께 사용되었습니다.

### 💻 하드웨어 환경

* **CPU**: Intel(R) Xeon(R) Gold 6126 @ 2.60GHz
* **GPU**: NVIDIA GeForce RTX 2080 Ti
* **RAM**: 32GB

---

### 📎 실행 방법 (예시)

```bash
pip install -r requirements.txt
python train_model.py
```

※ `requirements.txt`는 `pip freeze > requirements.txt`로 직접 생성 가능하며, 프로젝트 공유 시 포함을 권장합니다.

---

### ✅ 기타 참고 사항

* 분석 대상 데이터는 `.csv` 형식이며 약 48개의 공정 변수(X, Y, Z, 스핀들 속도, 전압 등)를 포함합니다.
* 모델 학습 및 테스트는 Python 환경에서 전적으로 이루어지며, Jupyter Notebook이나 Python script 모두 지원 가능합니다.
