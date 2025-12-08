# 다중 데이터셋 항공기 표현 학습 – 기술 보고서

## 1. 요약
PlanesNet + HRPlanes + FGVC Aircraft를 동시에 학습시키는 통합 파이프라인과 개발 과정에서 발생한 오류/난제 및 해결 방법, 그리고 추가 epoch/시각화가 성능 추적에 미친 영향을 정리하였다. 핵심 학습 스크립트(`src/train_multidataset.py`)는 CLIP 기반 표현 학습, 회전 지향 증강(Ro-CIT), UNet 보조 분할, DiT 개선, BERT 텍스트 프롬프트를 한 흐름으로 묶는다.

## 2. 시스템 개요
- **데이터셋**: PlanesNet(위성 이진 분류), HRPlanes(경계 상자를 분할 마스크로 변환), FGVC Aircraft(세밀 분류).
- **모델 구조**: CLIP 스타일 CNN 백본 → DiT → 데이터셋별 분류 헤드, HRPlanes용 UNet 디코더.
- **손실 구성**: 분류(전체), 분할(HRPlanes), 이미지-텍스트 CLIP 대조 손실.
- **디바이스 분배**: `--vision-device`에서 비전 네트워크, `--text-device`에서 BERT 임베딩 캐시.

## 3. 이슈 및 해결 내역
| 이슈 | 원인 | 해결 |
| --- | --- | --- |
| CUDA Assertion | CUDA 미포함 파이토치 | CPU 실행 옵션 안내 (`--vision-device cpu`). |
| CLIP 손실 차원 불일치 | 이미지 512-d vs. BERT 768-d | 백본 투영 차원을 768로 확장, DiT/헤드에 동일 차원 사용. |
| HRPlanes 배치 실패 | 메타데이터에 길이가 다른 박스 리스트 포함 | 메타데이터에서 `boxes` 제거, 마스크 텐서만 유지. |
| 분할 손실 크기 오류 | Decoder 출력 32×32, 마스크 256×256 | BCE 손실 전 `F.interpolate`로 업샘플. |
| logs/plots 업로드 실패 | `.gitignore`의 `results/` 규칙 | `results/*`로 변경 후 `!results/logs/**`, `!results/plots/**` 예외 추가. |

## 4. Epoch 별 학습 경향
- 초기 1~3 epoch에서 손실이 급감하고 검증 정확도가 상승, 특히 PlanesNet은 3 epoch 근처가 최적.
- 4 epoch 이상에서는 과적합 징후(검증 손실 상승)가 나타나 조기 종료 권장.
- 다중 데이터셋 학습에서도 4~6 epoch 수준으로 설정 시 시간 대비 성능이 적절함.

## 5. 시각화 파이프라인
`src/visualize.py` 실행만으로 `results/plots/`에 6종 그래프가 생성된다.
1. **막대 그래프**: PlanesNet 클래스 분포.
2. **선 그래프**: 각 데이터셋 train/val 정확도 추이.
3. **상자 그림**: 데이터셋별 학습 loss 분포.
4. **히트맵**: 검증 정확도 상관 행렬.
5. **산점도**: train vs. val 정확도.
6. **히스토그램**: CLIP 손실 분포.

## 6. 결론 및 향후 계획
- GPU 지원 PyTorch 환경이 갖춰지면 보다 장기 학습에 바로 적용 가능.
- 워크플로우: `python src/train_multidataset.py …` → `python src/visualize.py` 만으로 로그/플롯 관리.
- 향후 과제: 실제 CLIP 가중치 로드, UNet 디코더 추가 튜닝, 조기 종료 자동화 모듈.
