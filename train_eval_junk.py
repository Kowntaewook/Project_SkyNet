# 목적 : flow.csv 불러와 baseline + DL 모델 학습/평가 (Purpose: Load flow.csv and baseline + DL model training/evaluation)
# 사용법 : python train_and_eval.py --flows flows.csv --out_dir results (How to use : python train_and_eval.py --flows flows.csv --out_dir results)
# Note : Code may bit messy made with little ChatGPT and ME (Kwontaewook)

import os
import argparse
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 아직 안 쓰는데 혹시 몰라서 남겨둠 (not used yet but just in case)
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import layers

# 데이터 로딩 및 기본 처리 (Data loading and basic processing)
def load_and_prepare(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일 없음: {csv_path}")

    df = pd.read_csv(csv_path)

    if 'label' not in df.columns:
        raise ValueError("CSV 파일에 'label' column이 반드시 필요합니다.")
    
    # 라벨 인코딩 단순히 String -> int (label encoding simply String -> int)
    le  = LabelEncoder()
    df['label_enc'] = le.fit_transform(df['label'].astype(str))

    # 숫자형만 남기고 나머진 0으로 (Keep only numeric others to 0)
    drop_cols = ['src', 'dst', 'pcap_file', 'label']
    feature_cols = [c for c in df.columns if c not in drop_cols + ['label_enc']]
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    y = df['label_enc'].values

    return X, y, le

# Random Forest baseline
def run_random_forest(X_train, X_test, y_train, y_test, out_dir):
    rf_model = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=42) # 100 보다 좀 높혀봄
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict(X_test)

    print("==== Random 결과 ====")
    print(classification_report(y_test, preds))
    # confusion matrix 생략 사유 : 귀찮음
    # print(confusion_matrix(y_test, preds))

    joblib.dump(rf_model, os.path.join(out_dir, 'rf_model.joblib'))
    return rf_model


# CNN 모델 정의 (CNN model definition)
def make_cnn(input_shape, num_classes):
    cnn_net = Sequential()
    cnn_net.add(layers.Input(shape=input_shape))
    cnn_net.add(layers.Reshape((input_shape[0], 1)))
    cnn_net.add(layers.Conv1D(64, kernel_size=3, activation="relu"))
    cnn_net.add(layers.MaxPool1D(2))
    cnn_net.add(layers.Conv1D(128, kernel_size=3, activation="relu"))
    cnn_net.add(layers.GlobalMaxPool1D())
    cnn_net.add(layers.Dense(64, activation="relu"))
    cnn_net.add(layers.Dense(num_classes, activation="softmax"))
    
    cnn_net.compile(optimizer="adam",
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"])
    return cnn_net

# LSTM 모델 정의 (LSTM model definition)
def make_lstm(input_shape, num_classes):
    lstm_model = Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((input_shape[0], 1)),
        layers.LSTM(64),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    lstm_model.compile(optimizer="adam",
                       loss="sparse_categorical_crossentropy",
                       metrics=["accuracy"])
    return lstm_model


# DL 학습 함수 (CNN + LSTM 둘다 돌림) (DL training function (both CNN + LSTM))
def train_dl_models(X_train, X_test,y_train, y_test, out_dir):
    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y_train))

    
    #CNN
    cnn_model = make_cnn(input_shape, num_classes)