# autoencoder.py
# 목적: 정상 플로우(benign)만 학습시키고, 재구성 오차(reconstruction error)로 이상 탐지 시도

import numpy as np
import pandas as pd
import joblib
from keras import layers, models


# 간단한 오토인코더 구조
def make_autoencoder(input_size):
    # 입력 레이어
    input_layer = layers.Input(shape=(input_size,))
    
    # 인코딩 부분 (차원 축소)
    h = layers.Dense(128, activation='relu')(input_layer)
    h = layers.Dense(64, activation='relu')(h)
    h = layers.Dense(32, activation='relu')(h)   # bottleneck
    
    # 디코딩 부분 (복원)
    h = layers.Dense(64, activation='relu')(h)
    h = layers.Dense(128, activation='relu')(h)
    
    # 출력 (원래 차원으로 복원)
    output_layer = layers.Dense(input_size, activation='linear')(h)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder


def train_autoencoder(X_normal, save_path="ae_model.h5"):
    """
    X_normal: 정상 데이터 (scaler로 미리 변환된 상태라고 가정)
    save_path: 학습된 모델 저장 경로
    """
    # 모델 빌드
    ae = make_autoencoder(X_normal.shape[1])
    
    # 학습
    history = ae.fit(
        X_normal, X_normal,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        verbose=2
    )
    
    # 모델 저장
    ae.save(save_path)
    
    return ae, history


# 사용 예시
# df = pd.read_csv("flows.csv")
# benign_data = df[df['label'] == 'benign']
# scaler = joblib.load("scaler.pkl")
# X_scaled = scaler.transform(benign_data.drop(columns=["label"]))
# train_autoencoder(X_scaled, "autoencoder_model.h5")
