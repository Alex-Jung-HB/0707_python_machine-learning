# 0707_python_machine-learning
<지도학습(Supervised Learning)과 비지도학습(Unsupervised Learning)의 차이>
1. 기본 개념  
o 지도학습  
- 정답(Label)이 있는 데이터로 학습
- 입력(X)과 출력(Y)의 관계를 학습
   "선생님이 정답을 알려주며 가르치는" 방식
o 비지도학습  
- 정답이 없는 데이터로 학습  
- 데이터 자체의 패턴이나 구조를 발견  
   "스스로 규칙을 찾아내는" 방식  
2. 주요 차이점 비교  
o 학습 데이터  
- 지도학습: 레이블된 데이터를 사용하여 입력(X)과 정답(Y)이 쌍으로 제공되어, 모델이 입력과 출력 간의 관계를 학습가능  
- 비지도학습: 레이블이 없는 데이터만 사용하며 오직 입력 데이터(X)만 주어지고 모델이 스스로 데이터의 구조를 파악  
o 학습 목적  
- 지도학습: 주로 예측과 분류를 목적으로 사용하며 새로운 데이터가 들어왔을 때 정확한 출력값을 예측하도록 함.  
- 비지도학습: 데이터 내의 숨겨진 패턴을 발견하거나 유사한 데이터끼리 그룹화하도록 함.  
o 평가 방법  
- 지도학습: 정확도(Accuracy), F1 Score, Precision, Recall 등 정답과 비교하여 명확한 수치로 성능을 평가  
- 비지도학습: 실루엣 계수, 엘보우 방법, Davies-Bouldin Index 등을 사용하며, 평가가 상대적으로 주관적일 수 있음.  
o 계산 비용  
- 지도학습: 정답이 주어져 있어 학습 방향이 명확하므로 상대적으로 계산 비용이 낮음.  
- 비지도학습: 데이터의 모든 가능한 패턴을 탐색해야 하므로 일반적으로 더 높은 계산 비용 필요  
o 결과 해석  
- 지도학습: 출력이 명확하게 정의되어 있어 결과 해석이 직관적이고 명확함.  
- 비지도학습: 발견된 패턴이나 클러스터의 의미를 해석하는 것이 주관적이며, 도메인 지식이 필요한 경우가 많음.  
3. 대표적인 알고리즘  
o 지도학습  
- 분류: SVM, Decision Tree, Random Forest, Neural Network  
- 회귀: Linear Regression, Polynomial Regression, Ridge/Lasso  
o 비지도학습  
- 클러스터링: K-means, DBSCAN, Hierarchical Clustering  
- 차원 축소: PCA, t-SNE, Autoencoder  
- 연관 규칙: Apriori, FP-Growth  

<기계학습 시에 필요한 라이브러리 호출>  
1. 기본 호출 예시  
!pip install opencv-python tensorflow scikit-learn matplotlib pillow    
import cv2  
import numpy as np  
import tensorflow as tf  
from tensorflow.keras import layers, models  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score, classification_report  
import matplotlib.pyplot as plt  
import os from PIL import Image  

2. 라이브러리 요약
- 아래의 라이브러리들을 조합하면 컴퓨터 비전과 머신러닝 프로젝트의 전체 파이프라인을 구축할 수 있습니다.
  
A. OpenCV-Python
1) 개요  
용도: 컴퓨터 비전 및 이미지 처리  
주요 기능: 이미지/비디오 읽기/쓰기, 필터링, 객체 검출, 얼굴 인식, 에지 검출  
특징: C++로 작성되어 매우 빠른 성능, 실시간 처리 가능  
예시: cv2.imread(), cv2.Canny(), cv2.findContours()  
2) 라이브러리 구성  
o 핵심 모듈  
cv2.core: 기본 데이터 구조와 함수 (배열, 매트릭스 연산)  
cv2.imgproc: 이미지 처리 함수들  
cv2.imgcodecs: 이미지 파일 입출력  
cv2.videoio: 비디오 파일 및 카메라 입출력  
cv2.highgui: 창 관리, 이벤트 처리  
o 고급 모듈  
cv2.feature2d: 특징점 검출 및 기술자  
cv2.objdetect: 객체 검출 (HOG, Haar Cascade)  
cv2.calib3d: 카메라 캘리브레이션, 3D 재구성  
cv2.ml: 머신러닝 알고리즘  
cv2.photo: 사진 처리 (노이즈 제거, HDR)  
3) 주요 용어    
o 기본 용어    
Mat: OpenCV의 기본 이미지/행렬 데이터 구조  
Pixel: 이미지의 최소 단위 (화소)  
Channel: 색상 채널 (Gray: 1채널, BGR: 3채널)  
Depth: 각 픽셀의 비트 깊이 (8bit, 16bit, 32bit)  
o 이미지 처리 용어  
ROI (Region of Interest): 관심 영역  
Kernel/Filter: 컨볼루션 연산에 사용되는 작은 행렬  
Morphology: 형태학적 연산 (침식, 팽창)  
Threshold: 임계값 처리  
Contour: 윤곽선  
Edge: 가장자리  
o 특징 검출 용어  
Feature Point: 특징점 (코너, 가장자리)  
Descriptor: 특징 기술자  
Keypoint: 핵심 점  
SIFT/SURF/ORB: 특징 검출 알고리즘들  
4) 활용방법  
o 이미지 전처리  
o 에지 및 윤곽선 검출  
o 객체 검출  
o 비디오 처리  

B. TensorFlow  
1) 개요  
용도: 딥러닝 및 머신러닝 프레임워크  
주요 기능: 신경망 구축, 모델 훈련, GPU 가속 연산  
특징: Google이 개발, Keras API 내장, 분산 훈련 지원  
예시: CNN, RNN, Transformer 모델 구축  
2) 라이브러리 구성  
o 핵심 레이어  
tf.Tensor: 기본 데이터 구조 (다차원 배열)  
tf.Variable: 학습 가능한 변수  
tf.Operation: 연산 그래프의 노드  
tf.Graph: 연산 그래프 구조  
o High-Level API  
tf.keras: 고수준 신경망 API (권장)  
keras.layers: 신경망 레이어들  
keras.models: 모델 구성  
keras.optimizers: 최적화 알고리즘  
keras.losses: 손실 함수  
keras.metrics: 평가 지표  
o Mid-Level API  
tf.nn: 저수준 신경망 연산  
tf.math: 수학적 연산  
tf.linalg: 선형대수 연산  
tf.image: 이미지 처리  
tf.audio: 오디오 처리  
o Data Processing  
tf.data: 데이터 파이프라인 구성  
tf.io: 입출력 연산  
tf.feature_column: 특성 컬럼 처리  
o Distribution & Deployment  
tf.distribute: 분산 훈련  
tf.saved_model: 모델 저장/로드  
tf.lite: 모바일/임베디드 배포  
tf.js: 웹 브라우저 배포  
3) 주요 용어  
o 기본 개념  
Tensor: 다차원 배열 (스칼라, 벡터, 행렬의 일반화)  
Graph: 연산들의 그래프 구조  
Session: 그래프를 실행하는 환경 (TF 1.x에서 사용)  
Eager Execution: 즉시 실행 모드 (TF 2.x 기본)  
o 모델 관련 용어  
Layer: 신경망의 층  
Model: 여러 레이어로 구성된 신경망  
Sequential: 순차적 모델  
Functional API: 함수형 모델 API  
Subclassing: 클래스 상속을 통한 모델 정의  
o 훈련 관련 용어  
Forward Pass: 순전파  
Backward Pass: 역전파  
Gradient: 기울기  
Optimizer: 최적화 알고리즘  
Loss Function: 손실 함수  
Epoch: 전체 데이터를 한 번 훈련하는 단위  
Batch: 한 번에 처리하는 데이터 묶음  
o 고급 개념  
Checkpoint: 모델 체크포인트  
Callback: 훈련 중 실행되는 함수  
Regularization: 정규화  
Dropout: 드롭아웃  
Transfer Learning: 전이학습  
4) 활용방법  
이미지 분류: CNN을 사용한 객체 인식  
자연어 처리: RNN, LSTM, Transformer 모델  
추천 시스템: 협업 필터링, 딥러닝 기반 추천  
시계열 예측: LSTM, GRU 모델  
생성 모델: GAN, VAE  
강화학습: DQN, A3C 등  

C. Scikit-learn
1) 개요
용도: 전통적인 머신러닝 알고리즘
주요 기능: 분류, 회귀, 클러스터링, 차원 축소, 전처리
특징: 사용하기 쉬운 API, 풍부한 알고리즘, 잘 정리된 문서
예시: RandomForestClassifier, train_test_split, StandardScaler
2) 라이브러리 구성
3) 주요 용어
4) 활용방법

D. Matplotlib
1) 개요
용도: 데이터 시각화 및 그래프 생성
주요 기능: 선 그래프, 막대 그래프, 산점도, 히스토그램, 이미지 표시
특징: MATLAB 스타일 인터페이스, 고도로 커스터마이징 가능
예시: plt.plot(), plt.imshow(), plt.hist()
2) 라이브러리 구성
3) 주요 용어
4) 활용방법
   
E. Pillow (PIL)
1) 개요
용도: 이미지 처리 및 조작
주요 기능: 이미지 열기/저장, 크기 조정, 회전, 필터 적용, 포맷 변환
특징: Python Imaging Library(PIL)의 현대적 버전, 다양한 이미지 포맷 지원
예시: Image.open(), resize(), rotate()
2) 라이브러리 구성
3) 주요 용어
4) 활용방법
