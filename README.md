# Multi-Dataset Aircraft Representation Learning (CLIP + U-Net + DiT + BERT)

> **PlanesNet / HRPlanes(v2) / FGVC Aircraft**를 통합한 항공기 컴퓨터비전 실험  
> Contrastive Language-Image Pre-Training(CLIP) + Rotation-oriented Augmentation + U-Net Transfer + DiT + BERT

---

## 📑 목차 (Table of Contents)

- [1. 프로젝트 개요](#1-프로젝트-개요)
- [2. 사용 데이터셋](#2-사용-데이터셋)
- [3. 개발 언어 및 기술 스택](#3-개발-언어-및-기술-스택)
- [4. 연구 / 실험 목표](#4-연구--실험-목표)
- [5. 방법론](#5-방법론)
  - [5.1 CLIP 기반 Contrastive Learning](#51-contrastive-language-image-pre-training-clip)
  - [5.2 Rotation-oriented Continuous Image Translation](#52-rotation-oriented-continuous-image-translation-ro-cit)
  - [5.3 U-Net 전이 학습](#53-u-net-전이-학습-transfer-learning)
  - [5.4 DiT + BERT](#54-dit--bert)
- [6. 실험 설정](#6-실험-설정)
- [7. 시각화 (6개 플롯)](#7-시각화-6개-플롯)
- [8. 프로젝트 구조 예시](#8-프로젝트-구조-예시)
- [9. 실행 방법 (Colab + Web)](#9-실행-방법-colab--web)
- [10. 기대 결과 및 한계](#10-기대-결과-및-한계)
- [11. TODO](#11-todo)

---

## 1. 프로젝트 개요

이 프로젝트는 항공기(aircraft)를 대상으로 한 **멀티 데이터셋 컴퓨터비전 실험**이다.  
서로 다른 시점/도메인의 항공기 이미지 데이터셋(위성 이미지, 고해상도 항공 촬영, 일반 사진)을 통합하여,

- **Contrastive Language-Image Pre-Training(CLIP)** 기반의 표현 학습(representation learning)을 수행하고,
- **Rotation-oriented Continuous Image Translation** 기법으로 회전 불변성을 강화하며,
- **U-Net 전이 학습, DiT(Vision Transformer 기반 Diffusion Transformer), BERT 텍스트 인코더**를 결합한 여러 모델 구성을 비교한다.

> 최종적으로는 각 모델의 성능과 특성을 여섯 가지 그래프(Boxplot, Heatmap, Line plot, Scatter plot, Histogram, Bar plot)로 시각화한다.

---

## 2. 사용 데이터셋

### 2.1 Planes in Satellite Imagery (PlanesNet, Kaggle)

- **도메인**: 위성 이미지 (top-down 시점)
- **태스크**: 이진 분류 (plane / no-plane)
- **구성**:
  - 수만 장 규모의 작은 패치 이미지 (예: 20×20 픽셀 수준)
  - 각 이미지에 항공기 존재 여부 레이블 포함
- **특징**:
  - 배경(건물, 도로, 들판 등) 위에 아주 작게 찍힌 항공기
  - 회전, 크기 변동이 심해 **회전 불변 표현 학습** 테스트에 적합

### 2.2 HRPlanes / HRPlanesv2

- **도메인**: 고해상도 위성/항공 이미지
- **태스크**: 객체 검출 (bounding box, 일부 마스크 변환 가능)
- **구성 (v2 기준)**:
  - 수천 장 규모의 고해상도 이미지
  - 수만 개 수준의 항공기 박스(annotation)
- **특징**:
  - 활주로, 공항 주변 등에서 다양한 크기와 방향의 항공기가 등장
  - **회전된(orientation이 다양한) 항공기**가 많아,  
    Rotation-oriented Continuous Image Translation 실험에 핵심 데이터셋으로 사용

### 2.3 FGVC Aircraft (Kaggle Mirror)

- **도메인**: 일반 RGB 이미지 (spotting, 항공기 촬영 이미지)
- **태스크**: 세밀 분류(Fine-grained classification)
- **구성**:
  - 약 10,000장 규모의 항공기 이미지
  - 다양한 민간/군용 항공기 기종(class) 레이블 제공
- **특징**:
  - 기종 간 외형 차이가 미묘하여,  
    **CLIP + BERT 기반 텍스트 조건 분류**, **fine-grained representation** 비교에 적합

---

## 3. 개발 언어 및 기술 스택

### 3.1 개발 언어

- **Python 3.x**
  - 메인 실험 및 학습 코드
  - 데이터 전처리, 모델 학습, 평가, 시각화 전부 Python 기반

### 3.2 딥러닝 / ML 스택

- **PyTorch**, `torchvision`
  - CLIP/DiT/U-Net 모델 구현 및 학습
- **Hugging Face Transformers**
  - CLIP, ViT, DiT, BERT 등 사전학습 모델 활용
- **scikit-learn**
  - 평가 지표(accuracy, F1 등), 임베딩 시각화(t-SNE, UMAP 전처리)용
- **NumPy, Pandas**
  - 데이터 로딩, 전처리, 통계량 계산

### 3.3 시각화

- **Matplotlib, Seaborn**
  - 6개 플롯(boxplot, heatmap, line plot, scatter plot, histogram, bar plot) 생성
  - 결과를 이미지 파일로 저장 후 README/보고서에서 참조

### 3.4 Web / 대시보드 (기말과제 참고용)

기말과제 제출 시, **웹 대시보드 화면을 보면서 수기로 내용을 정리**할 수 있도록 간단한 웹 UI를 제공한다.

- **Backend (선택)**
  - 간단한 경우: 별도 백엔드 없이 Streamlit 단일 앱으로 처리
  - 확장형 구성이 필요할 경우:
    - **FastAPI** 기반 inference API 서버
    - 학습된 모델 로딩 및 `/predict`, `/metrics` 엔드포인트 제공

- **Web UI**
  - **Streamlit** 기반 대시보드 (기본 가정)
    - 주요 기능:
      - 예시 이미지 업로드 및 모델 예측 결과 확인
      - 데이터셋/모델별 성능 요약표 표시
      - 저장된 그래프(6개 플롯) 표시
    - 명령어 예시:
      ```bash
      streamlit run web/app.py
      ```

  > 이 Web UI는 **실험 결과를 시각적으로 확인**하고,  
  > 기말 과제 제출 시 웹 화면을 보면서 **보고서/수기 답안 작성**에 참고하기 위한 용도로 설계되었다.

---

## 4. 연구 / 실험 목표

1. **멀티 도메인 항공기 이미지**(위성, 고해상도, 일반 사진)를 하나의 프레임워크에서 학습 가능한지 검증
2. **CLIP 기반 Contrastive Learning**을 활용하여,  
   - 데이터셋 간 공통적인 “항공기” 표현을 학습하고  
   - zero-shot / few-shot 분류 성능을 관찰
3. **Rotation-oriented Continuous Image Translation**으로  
   - 특히 위성 이미지(PlanesNet, HRPlanes)의 **회전 불변성(rotation-invariance)**을 향상
4. **U-Net 전이 학습, DiT, BERT**를 조합한 다양한 모델 구성 비교
5. 모델 성능과 특성을 **6가지 시각화(boxplot, heatmap, line, scatter, histogram, bar)**로 정리

---

## 5. 방법론

### 5.1 Contrastive Language-Image Pre-Training (CLIP)

- 기본 프레임워크는 **이미지 인코더(vision encoder)**와 **텍스트 인코더(text encoder)**를 동시에 학습하는 CLIP 구조를 따른다.
- 이미지–텍스트 쌍 (예:  
  - `"satellite image of an airplane"`  
  - `"high-resolution image of an airport with multiple planes"`  
  - `"side view of a commercial aircraft"`  
  ) 을 구성하고,
- **InfoNCE / contrastive loss**를 사용하여  
  - 같은 의미의 이미지–텍스트 임베딩은 가깝게,  
  - 다른 의미는 멀어지도록 학습한다.

이미지 인코더는 **CLIP ViT 계열 + DiT 백본**,  
텍스트 인코더는 **BERT 계열**로 확장하여 실험한다.

### 5.2 Rotation-oriented Continuous Image Translation (Ro-CIT)

- PlanesNet, HRPlanes처럼 **회전된 항공기**가 많이 등장하는 데이터셋에 대해,
- 단순 랜덤 회전이 아니라,  
  **연속적인 각도 변화 + 이미지 품질 유지**를 목표로 하는 rotation-oriented augmentation을 적용한다.
- 주요 아이디어:
  - 항공기의 방향(heading)을 연속적인 변수로 보고,
  - 여러 각도에서의 이미지를 생성/변환하여,
  - 모델이 “기체의 방향과 상관없이 항공기 특징”을 학습하도록 유도한다.

### 5.3 U-Net 전이 학습 (Transfer Learning)

- HRPlanes에서 제공되는 bounding box를 기반으로 간단한 마스크를 생성하거나,
- 일부 데이터에 대해 manual/heuristic segmentation mask를 만들어,
- **U-Net 기반 segmentation 네트워크**를 보조 태스크(auxiliary task)로 학습한다.
- U-Net의 encoder 부분에 **CLIP/DiT에서 학습된 feature**를 전이시켜,
  - 객체 위치 및 윤곽에 대한 구조 정보를 추가로 학습하고,
  - 분류/검출 태스크의 표현력을 높이는 것이 목적이다.

### 5.4 DiT + BERT

- **DiT(Vision Transformer 기반 Diffusion Transformer)**:
  - 이미지 인코더로서 CLIP backbone과 비교
  - 위성/항공 이미지의 전역 문맥(global context) 캡처 성능을 테스트
- **BERT**:
  - 텍스트 인코더로 사용하여,  
    비행기 기종명, 설명 문장, 데이터셋별 context 텍스트 임베딩을 제공
  - CLIP 구조에서 텍스트 측 임베딩 품질이 성능에 미치는 영향을 분석

### 5.5 기술 향상 포인트 (ROI, CLIP, CNN, U-Net 전이, DiT, BERT)

- **ROI 최적화**: 단순 정확도 개선뿐 아니라 `추가 가치(정확도↑) - 추가 비용(연산/서빙)`을 계산하여 모델 선택. 고정비 대비 GPU/CPU 시간, 배치 크기, 양자화(8/4-bit) 옵션을 비교.
- **CLIP 강화**: 텍스트 프롬프트 엔지니어링(도메인 서술 템플릿 5~10개 앙상블), 이미지 측 rotation/색상 증강, temperature 학습, projector fine-tune로 제로샷 성능 및 전이 성능 동시 개선.
- **CNN 기준선 개선**: SE/CBAM 채널 어텐션 삽입, mixed precision + cosine LR 스케줄, label smoothing을 적용해 작은 모델도 견고하게 유지.
- **U-Net 전이 학습**: CLIP/DiT encoder를 동결하거나 상위 블록만 미세조정한 하이브리드 U-Net으로 보조 segmentation 학습 → detection/분류 피쳐 품질 향상(멀티태스킹 regularization).
- **DiT 세부 튜닝**: patch size 16/32 비교, 시간 스텝을 줄인 경량 DiT 변형, gradient checkpointing/flash attention으로 메모리 절감 후 더 깊은 스택 실험.
- **BERT 측면**: 도메인 용어 추가 vocab(airport/aircraft taxonomy)와 prompt prefix(“satellite view of…”)로 텍스트 임베딩을 정제, CLS pooling과 mean pooling 비교로 downstream 분류 성능 검증.
- **평가/모니터링**: PlanesNet/HRPlanes/FGVC/RSICD/RSITMD, RemoteCLIP 공개 수치와 자체 CNN/CNIP 실험을 동일 메트릭(Acc/F1/latency)과 비용 기준으로 표준화해 ROI 대시보드에 반영.

---

## 6. 실험 설정

### 6.1 환경 (Colab)

- Python 3.x (Google Colab)
- 주요 라이브러리:
  - `torch`, `torchvision`
  - `transformers`
  - `tqdm`, `numpy`, `pandas`
  - `matplotlib`, `seaborn`
  - `scikit-learn`

### 6.2 공통 파이프라인

1. **데이터 다운로드**
   - Kaggle 또는 원본 페이지에서 PlanesNet / HRPlanes(v2) / FGVC Aircraft 다운로드
2. **전처리 & 통합**
   - 공통 이미지 크기로 resize (예: 224×224 또는 256×256)
   - train / val / test split
   - 텍스트 레이블 생성 (예: `"satellite image of a plane"`, `"no plane"`, `"Boeing 737"`, `"Airbus A320"` 등)
3. **학습**
   - CLIP loss 기반 contrastive training
   - rotation-oriented augmentation 활성화
   - U-Net 보조 태스크, DiT/CLIP 백본 비교, BERT 텍스트 인코더 비교
4. **평가**
   - 주요 지표:
     - Accuracy, F1-score
     - Per-class 성능
     - 데이터셋/모델별 비교

---

## 7. 시각화 (6개 플롯)

다음 6가지 그래프를 통해 결과를 시각화한다.  
(실제 그래프 파일명은 예시이며, `results/plots/` 디렉터리에 저장하는 것을 가정한다.)

1. **Boxplot – 모델/데이터셋별 성능 분포**
   - 파일 예시: `results/plots/boxplot_metrics.png`
   - 내용:
     - 각 모델(CLIP only, CLIP+U-Net, CLIP+DiT, CLIP+DiT+U-Net, etc.)의
       - per-class accuracy 또는 per-image loss 분포를 boxplot으로 표현
     - 데이터셋(PlanesNet / HRPlanes / FGVC) 간 분포 차이 비교

2. **Heatmap – Confusion Matrix / Correlation**
   - 파일 예시: `results/plots/heatmap_confusion.png`
   - 내용:
     - FGVC Aircraft의 기종별 confusion matrix heatmap
     - 또는 데이터셋별 feature correlation matrix
   - 해석:
     - 어떤 기종/클래스가 서로 많이 헷갈리는지,
     - 데이터셋 간 표현 공간의 상관관계를 확인

3. **Line Plot – 학습 곡선 (Training / Validation)**
   - 파일 예시: `results/plots/lineplot_training_curves.png`
   - 내용:
     - epoch에 따른 train / val loss, accuracy 변화를 line plot으로 표현
   - 해석:
     - 수렴 여부, overfitting 여부, augmentation/백본 변경에 따른 학습 안정성 비교

4. **Scatter Plot – 임베딩 시각화 (t-SNE / UMAP)**
   - 파일 예시: `results/plots/scatter_embeddings.png`
   - 내용:
     - CLIP/DiT 이미지 임베딩을 t-SNE 또는 UMAP으로 2D 투영
     - 점 색깔:
       - 데이터셋별(PlanesNet/HRPlanes/FGVC) 또는 클래스별(plane/no-plane, 기종 등)
   - 해석:
     - 데이터셋 간 도메인 차이,
     - 항공기와 비-항공기, 기종별 클러스터링 패턴 확인

5. **Histogram – 스코어 / 확률 분포**
   - 파일 예시: `results/plots/histogram_scores.png`
   - 내용:
     - 예측 확률(confidence), logits, 혹은 항공기 존재 점수의 분포를 histogram으로 표현
     - plane / no-plane 또는 정답 / 오답에 대해 분리된 분포를 비교
   - 해석:
     - 모델의 calibration 정도,  
       확신이 너무 강하거나/약한 구간 파악

6. **Bar Plot – 데이터셋/클래스별 샘플 수 및 성능**
   - 파일 예시: `results/plots/barplot_dataset_stats.png`
   - 내용:
     - x축: 데이터셋 또는 클래스
     - y축: 샘플 수, 평균 accuracy, F1 등
   - 해석:
     - 클래스 불균형 정도와 성능 간의 관계,
     - 데이터셋마다 난이도 차이 확인

---

## 8. 프로젝트 구조 예시

```bash
project-root/
├─ data/
│  ├─ planesnet/
│  ├─ hrplanes/
│  └─ fgvc_aircraft/
├─ src/
│  ├─ datasets/
│  │  ├─ planesnet_dataset.py
│  │  ├─ hrplanes_dataset.py
│  │  └─ fgvc_dataset.py
│  ├─ models/
│  │  ├─ clip_backbones.py      # CLIP, DiT, U-Net encoder 등
│  │  ├─ unet_decoder.py
│  │  └─ classifier_heads.py
│  ├─ train_clip.py
│  ├─ train_unet_aux.py
│  ├─ evaluate.py
│  └─ visualize.py              # 6가지 플롯 생성
├─ notebooks/
│  ├─ 01_eda_and_preprocessing.ipynb
│  ├─ 02_train_clip_multidataset.ipynb
│  ├─ 03_unet_transfer_and_dit.ipynb
│  └─ 04_visualization_plots.ipynb
├─ web/
│  └─ app.py                    # Streamlit 대시보드 (또는 FastAPI/HTML)
├─ results/
│  ├─ logs/
│  └─ plots/
├─ requirements.txt
└─ README.md
