# 🚀 SkyNet: 딥러닝 기반 악성 패킷 탐지 시스템


## Initial screen
![Image](https://github.com/user-attachments/assets/06ba6351-ddac-46b4-a2aa-818b2a9784a9)

## 📌 프로젝트 개요
네트워크 공격은 점점 정교해지고 있으며, 기존의 Rule-based 방식 IDS(침입 탐지 시스템)는 새로운 공격 유형에 취약합니다.  
본 프로젝트는 **머신러닝/딥러닝 기반 악성 패킷 탐지 시스템**을 구현하여, 다양한 공격 유형을 자동으로 탐지하고 분류하는 것을 목표로 합니다.

---

## 📊 데이터셋
본 프로젝트에서는 공개 데이터셋과 직접 수집한 데이터셋을 모두 활용했습니다.

- **공개 데이터셋**
  - CICIDS2017
  - UNSW-NB15
- **자체 수집 데이터셋**
  - 가상 환경(VM)에서 Port Scan, DoS, Brute-force 공격 패킷 캡처
- **전처리 과정**
  - pcap → CSV 변환
  - Feature Scaling (`StandardScaler`)
  - 라벨 인코딩 (`LabelEncoder`)

---

## 🧠 모델 구조
본 프로젝트는 머신러닝과 딥러닝을 비교합니다.

- **머신러닝 모델**: Random Forest  
- **딥러닝 모델**: CNN, LSTM  
- 입력: 네트워크 플로우 기반 Feature (Duration, Bytes, Flags 등)  
- 출력: 정상 / 악성 (공격 유형 분류 가능)

---

## ⚙️ 실행 방법

### 1. 데이터 전처리
```bash
python process_pcap.py --input flows.pcap --output flows.csv
