# evensplit_resnet8

generalist를 균등분할(Dirichlet 아님)하고 ResNet-8을 **scratch**로 학습시키는 실험 계열.
목적: 편향된 generalist가 specialist처럼 보이는 confound를 제거하고, 사전학습이 전문성 신호를
가리는 것도 없앤 상태에서 q_i(빈도 기반) 신호와 파이프라인 성능을 검증.

## 고정 세팅 (모든 run 공통)

| 항목 | 값 |
|---|---|
| 데이터셋 | CIFAR-10 (학습 50,000장, 테스트 10,000장) |
| 파티션 | `specialist_mix`, generalist는 **균등분할**(`--generalist_split equal`) |
| specialist 수 / 담당 클래스 | 3명, cat(3) / deer(4) / truck(9) 고정 |
| generalist_access_ratio | 0.15 |
| proxy_ratio | 0.1 |
| 모델 | ResNet-8, **ImageNet 사전학습 없이 scratch** |
| 입력 크기 | 32×32 |
| 옵티마이저 | SGD, lr **0.0025**, momentum **0.9** |
| batch size | 16 |
| **epoch** | **100** |
| 스케줄러 | 없음 |
| augmentation | RandomResizedCrop(scale 0.8~1.0) + 좌우반전 (기본 켜짐) |
| 로컬 학습 라운드 | 1 (클라이언트 간 통신 없음) |
| 평가 시 참조 배치(T0) | 500, 5회 무작위 추첨 반복 |

## 지금까지 완료한 run (`logs_2026-09-29/`)

| setting 라벨 | generalist 수 | purity | seed |
|---|---|---|---|
| p0.85_s90 | 10 | 0.85 | 90 |
| p0.65_s90 | 10 | 0.65 | 90 |
| p0.5_s90 | 10 | 0.5 | 90 |
| p0.85_s91 | 10 | 0.85 | 91 (독립 재현) |
| p0.85_s92 | 10 | 0.85 | 92 (독립 재현) |

재현: `bash experiments/evensplit_resnet8/run_experiment.sh <generalist수> <purity> [seed]` (저장소 루트에서 실행)

## 지표 설명

### tableA — specialist client별 분류 지표 (client × 담당 클래스 단위)

| 컬럼 | 의미 |
|---|---|
| `precision` | client가 그 클래스라고 말했을 때 실제로 맞을 확률 |
| `recall_sensitivity` | 진짜 그 클래스인 것 중 client가 맞힌 비율 (재현율 = 민감도, 같은 지표) |
| `specificity` | 진짜 그 클래스가 아닌 것 중 client가 "아니다"라고 맞힌 비율 (= 1 − 오경보율) |
| `balanced_acc` | (recall + specificity) / 2. 클래스 불균형에 덜 휘둘리는 단일 요약 지표 |
| `fpr` | 오경보율. 그 클래스가 아닌데 그 클래스라고 말한 비율 |
| `auroc` | client의 확률 점수 자체가 "진짜 그 클래스인지"를 얼마나 잘 구별하는지(임계값 무관) |
| `vote_frequency` | 그 클래스로 답한 전체 비율 (=q_i 원값) |

### tableB — 세팅별 집계기 성능

| 컬럼 | 의미 |
|---|---|
| `n_rescue` / `rescue_ratio_of_total` / `rescue_ratio_of_applicable` | rescue 대상(= specialist 클래스 & generalist 다수결 틀림 & specialist 맞음) 개수와, 전체 테스트/해당 클래스 쿼리 대비 비율 |
| `genmaj_acc_specialist_classes` / `_other_classes` | generalist 10명의 다수결만으로 그 클래스군을 맞히는 정확도 (그룹 사각지대 확인용) |
| `swamped_fraction` | (진단용, EM/Dawid-Skene 비교 시절 지표 — 현재 채택 파이프라인에는 해당 없음, §아래 주의 참고) |
| `CWTM_Acc_all` / `_Acc_rescue` | 견고집계(절사평균)만 썼을 때 전체/구제 정확도 |
| `pipeline_as_written_Acc_all` / `_Acc_rescue` | q_i 원래 수식 그대로(온도조절 없음) |
| `pipeline_tempering0.3_Acc_all` / `_Acc_rescue` | q_i + 온도조절(임시방편, target_maxprob=0.3) — **현재 채택 중인 파이프라인** |
| `dawid_skene_*`, `oracle_ceiling_*` | 폐기한 EM 방향의 비교값(참고용으로만 남김) |

## 주의

- `swamped_fraction`과 `dawid_skene`/`oracle_ceiling` 열은 **폐기된 EM(Dawid-Skene) 방향**을 검증하던 과정에서 생긴 값이다. 실제로 채택 중인 q_i+온도조절 파이프라인에서는 "다수결에 밀려서 지는" 비율이 0%였고(별도 분석, `analysis_where_specialist_loses.py`), 병목은 4단계 통계 유의성 검정(REGIME C 기권)이었다 — 자세한 내용은 대화 기록 참고, 별도 문서화 필요.
- 학습 하이퍼파라미터 중 momentum(0.9), batch(16), augmentation 종류는 RFI 논문에 명시되지 않아 **가정한 값**이다(논문 확인값은 SGD, lr 0.0025, epoch 100뿐).
