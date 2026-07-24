# VMem revisit failure 결과와 연구 방향

작성일: 2026-07-24

## 0. 무엇을 실제로 테스트했는가

### 공통 실행 조건

실제 GPU generation은 저장소의 `vmem` Conda 환경과 VMem checkpoint를 그대로
사용해 Gradio 없이 실행했다.

- GPU: NVIDIA H100 PCIe 80GB
- generator checkpoint: `liguang0115/vmem/vmem_weights.pth`
- point-map predictor: `liguang0115/cut3r/cut3r_512_dpt_4_64.pth`
- config: `configs/inference/inference.yaml`
- resolution: 576×576
- context/target 수: `K=4`, `M=4`
- diffusion inference: checked-in config의 50 steps
- 기본 scene/seed: `test_samples/living_room.jpg`, seed 42
- 추가 재현 scene/seed: `test_samples/open_door.jpg`, seed 42/43/44
- 이동 거리: command당 0.1
- interpolation output: command당 1 frame
- intervention 적용 시점: 기본적으로 outbound에는 적용하지 않고 revisit부터 적용

각 run은 입력 이미지 한 장을 identity camera pose에 놓고 시작했다. UI 버튼 대신
`Navigator`의 동일한 `move_forward`, `move_backward`, `turn_left`, `turn_right`
경로를 headless runner에서 호출했다. Pipeline, generator, checkpoint는 앱에서
사용하는 것과 같고, 실험 모드에서는 context 선택과 conditioning source만
계측하거나 통제했다.

### 실행한 57개 run의 정확한 구성

| Group | Run 수 | Scene/seed | Trajectory 및 비교 조건 |
| --- | ---: | --- | --- |
| Distractor dose | 4 | living_room/42 | `forward 1 → backward 1`; wrong latent+CLIP slot 수 0/1/2/4 |
| Component ablation | 4 | living_room/42 | 같은 exact return에서 첫 slot의 latent, CLIP, latent+CLIP, pose+intrinsics를 각각 교체 |
| Novel yaw | 9 | living_room/42 | `forward 1 → backward 1 → yaw 10/20/30°`; 각 yaw에서 correct/wrong/correct+wrong |
| Partial overlap | 3 | living_room/42 | `forward 1 → backward 1 → yaw 45°`; correct/wrong/correct+wrong |
| Context baseline | 9 | living_room/42 | 긴 exact return과 novel yaw에서 surfel/recent/initial-only/none 비교 |
| Rotation accumulation | 4 | living_room/42 | 같은 최종 yaw 90°를 `90×1`, `45×2`, `30×3`, `15×6`으로 생성 |
| Revisit gap | 3 | living_room/42 | exact return 전 gap cycle을 0/2/6회 넣어 총 generation call 2/6/14 비교 |
| Contamination follow-up | 3 | living_room/42 | wrong revisit 후 intervention을 끄고 `yaw 15° → yaw -15°`; correct/wrong/correct+wrong |
| Reproducibility | 18 | 2 scenes × 3 seeds | 각 scene/seed에서 exact return correct/wrong/correct+wrong |
| **합계** | **57** |  | 완료 57, 실패 0 |

Context baseline 9개는 다음과 같이 구성했다.

1. `forward×4 → backward×4` exact return에서 `surfel`, `recent`,
   `initial_only`를 각각 실행했다.
2. 같은 surfel outbound를 사용하고 revisit 구간에서만 spatial memory를 끄는
   `none` 조건을 실행했다.
3. 짧은 `forward → backward → yaw 20°`에서 `recent`, `initial_only`를
   비교했다.
4. 긴 `forward×4 → backward×4 → yaw 20°`에서 `surfel`, `recent`,
   `initial_only`를 비교했다.

Rotation test는 최종 camera rotation을 모두 90도로 맞추면서 generation call 수만
1/2/3/6으로 바꿨다. Gap test는 최종 exact pose를 유지하면서 중간에
`yaw 15° → yaw -15°` pair를 각각 0/2/6회 넣었다. 따라서 최종 pose 차이가
아니라 autoregressive 호출 누적 효과를 비교한다.

### Memory intervention의 정확한 의미

- `correct`: surfel routing이 선택한 index의 pose, latent, CLIP embedding,
  intrinsics를 그대로 사용한다.
- `none`: revisit에서 spatial routing을 사용하지 않고 recent context로 바꾼다.
- `wrong`: route pose와 tensor shape는 유지하되 선택 밖의 다른 stored frame에서
  conditioning content를 가져온다.
- `correct_plus_wrong`: K=4 slots 안에 correct source와 wrong source를 함께 둔다.

Intervention 전 route index와 intervention 후 latent/CLIP/image source index는
별도로 기록했다. Wrong memory 실험은 checkpoint나 surfel state를 손상시킨 것이
아니며, 같은 seed·trajectory·route에서 생성기가 잘못된 content conditioning에
얼마나 민감한지 보기 위한 controlled intervention이다.

Component test에서는 다른 source를 어느 부분에 적용할지 분리했다.

- `latent`: VAE latent만 다른 frame 것으로 교체
- `clip`: CLIP embedding만 교체
- `latent_clip`: 둘 다 교체
- `pose_intrinsics`: camera pose와 intrinsics만 교체

Dose test의 slot 순서는 surfel relevance로 정렬된 context 순서다. 따라서
`wrong_slot_count=1`은 무작위 한 slot이 아니라 가장 앞선 high-impact slot 하나를
교체한 조건이다.

### 각 run에서 저장하고 비교한 것

각 run은 다음 자료를 `experiments/results/overnight/<run_id>/`에 저장했다.

- 실제 frame과 전체 MP4
- 실제 camera pose와 command/generation call 수
- intervention 전 후보와 최종 route index
- latent, CLIP, pose, context image의 실제 source index
- surfel relevance, pose distance, selection reason, fallback 여부
- initial/final PSNR, MAE, MSE
- 직전 frame과의 변화량
- runtime과 CUDA peak memory
- JSON/JSONL trace, summary CSV
- frame 및 selected-memory contact sheet
- manual failure label manifest

Exact return의 PSNR/MAE는 동일한 camera pose인 초기 입력과 최종 생성 frame을
비교했다. Novel-yaw와 partial-overlap에는 ground truth가 없으므로 correct run의
output과 intervention output 사이 차이만 계산했다. 이 값은 memory sensitivity이지
novel-view accuracy가 아니다.

자동 validity check는 NaN/Inf, black frame, saturation, 급격한 pixel 변화,
near-duplicate를 찾았다. Rotation의 구조 붕괴는 이 heuristic이 잡지 못해 contact
sheet를 직접 확인하고 이 문서와 `failure_analysis.md`에 기록했다. 각 run에
`manual_labels.csv` template은 생성됐지만 frame별 수동 label은 아직 채우지 않았다.

## 0.1 논문 재현 테스트는 무엇을 기반으로 했는가

### 근거 자료

실험 설계는 다음 세 자료를 대조해 만들었다.

1. 저장소의 논문 PDF `2506.18903v3 (2).pdf`
2. 공식 저장소의 `README.md`, 공개 checkpoint와
   `configs/inference/inference.yaml`
3. 실제 실행 경로인 `app.py`, `navigation.py`, `modeling/pipeline.py`

논문에서 직접 사용한 근거는 다음과 같다.

- PDF 7쪽 Sec. 4.1: SEVA backbone, `K+M=21`, 경량 모델 `K=4/M=4`,
  RealEstate10K와 Tanks-and-Temples, FID/LPIPS/PSNR/SSIM/Rdist/Tdist 정의
- PDF 7쪽 Sec. 4.2: RealEstate10K camera trajectory를 10-frame 간격으로
  subsample하고, short는 다섯 번째 생성 이미지인 50th frame, long은 200 frame
  이상 떨어진 마지막 이미지를 평가
- PDF 7–8쪽 Sec. 4.3: initial-to-final trajectory를 생성한 뒤 같은 경로를
  역순으로 돌아오는 cycle trajectory
- PDF 8쪽 Sec. 4.3: RealEstate10K은 return trajectory에서 10 frame마다,
  Tanks-and-Temples advanced scene 6개는 첫 50 frames를 사용해 매 frame 평가
- PDF 8쪽 Sec. 4.4와 Table 4: temporal, camera distance, FOV, surfel VMem
  retrieval 비교
- PDF 11쪽 Appendix A: 경량 LoRA model의 `K=4/M=4`, inference CFG 3,
  point-map scale `σ=0.03`, surfel radius `α=0.2`
- PDF 11쪽 Appendix C: 이전 depth와 user camera를 고정하며 CUT3R point map을
  autoregressive하게 업데이트하는 방식

### 이번에 재현한 부분

이번 로컬 테스트는 논문의 공식 수치를 재산출한 full benchmark reproduction이
아니다. 정확한 표현은 **논문의 cycle/retrieval protocol을 기반으로 현재 공개
VMem code path를 재현한 failure-oriented local reproduction**이다.

재현한 요소는 다음과 같다.

1. 공개된 경량 VMem checkpoint와 `K=4/M=4` generator를 사용했다.
2. 입력 이미지 한 장에서 시작해 camera command를 autoregressive하게 생성했다.
3. 논문 Sec. 4.3의 핵심인 “나간 경로를 역순으로 따라 initial pose로 복귀”하는
   cycle trajectory를 최소 forward/backward 형태로 구현했다.
4. 논문 Table 4의 temporal retrieval을 `recent`, surfel retrieval을 `surfel`
   baseline으로 비교했다.
5. exact pose에서 초기 frame과 revisit frame의 PSNR을 분리해 측정했다.
6. surfel 후보, top-k route, fallback을 기록해 실제 VMem retrieval 경로였는지
   확인했다.
7. 논문이 언급한 cycle trajectory의 한계를 보완하기 위해 novel yaw,
   partial overlap, rotation-call accumulation, gap, conflicting memory를 추가했다.

즉 논문의 “cycle을 통해 revisit consistency를 본다”는 평가 논리를 출발점으로
삼았고, 공개 앱의 같은 generator/retrieval/memory code를 사용했다. 그 위에 논문
평가가 분리하지 않은 retrieval, integration, contamination failure를 진단하는
intervention과 logging을 추가했다.

### 공식 논문 재현과 다른 부분

다음 이유로 이번 수치를 논문 Table 1–4의 재현값이라고 부르면 안 된다.

- 공식 RealEstate10K test sequence와 ground-truth camera trajectory가 로컬
  저장소에 없었다.
- Tanks-and-Temples advanced scene 6개와 첫 50-frame trajectory를 실행하지 않았다.
- 로컬 입력은 `living_room.jpg`, `open_door.jpg` 두 장이며 camera trajectory도
  synthetic forward/back/yaw command다.
- 공식 return trajectory의 모든 평가 지점이 아니라 최소 exact-return endpoint를
  중심으로 비교했다.
- LPIPS, SSIM, FID와 DUSt3R 기반 Rdist/Tdist를 아직 계산하지 않았다.
- 다른 논문 baseline인 LookOut, GenWarp, MotionCtrl, ViewCrafter, SEVA를 다시
  실행하지 않았다.
- checked-in config는 50 inference steps와 `K=4/M=4`를 유지하지만 `cfg=2.0`이다.
  논문 Appendix A의 CFG는 3이다.
- 현재 config에는 `surfel.shrink_factor=0.05`,
  `radius_scale=0.5`가 있으며 논문의 `σ=0.03`, `α=0.2`와 이름 및 값이
  그대로 대응한다고 확인되지 않았다.
- 서버 안정화 과정에서 CUT3R memory-write 입력을 최근 최대 8 frames로 제한했다.
  논문은 retrieved past views와 newly generated views를 함께 사용한다고 설명한다.
- forward/backward navigation은 현재 NMS를 끄는 코드 경로가 있고, rotation은
  기본 NMS를 사용한다.
- surfel retrieval 실패 시 recent/latest fallback이 존재한다. 결과 집계에서는
  fallback 횟수를 기록하고 fallback-free subset도 별도로 계산했다.

따라서 현재 결과는 공개 VMem 구현에서 가설을 선별하는 evidence이고, 논문과
직접 비교 가능한 benchmark number는 아니다.

### 로컬 실험 재실행 방법

57-run matrix는 다음 명령으로 동일하게 계획하고, 완료된 run은 자동으로 건너뛴다.

```bash
conda activate vmem
python scripts/run_revisit_matrix.py \
  --groups dose components novel partial context rotation gap contamination repro \
  --scenes test_samples/living_room.jpg test_samples/open_door.jpg \
  --seeds 42 43 44 \
  --output-dir experiments/results/overnight
```

집계는 다음과 같이 재생성한다.

```bash
python scripts/analyze_revisit_matrix.py \
  --results-dir experiments/results/overnight
```

실행 당시 `--inference-steps` override를 주지 않았으므로
`configs/inference/inference.yaml`의 50 steps가 사용됐다. 각 run의 최종 설정은
자체 `run_config.json`에 저장돼 있어 문서보다 그 파일이 실행 사실의 최종
근거다.

### 공식 논문 수치까지 재현하려면 추가로 필요한 것

1. 논문과 동일한 RealEstate10K test split 및 camera trajectory를 확보한다.
2. 10-frame 간격으로 trajectory를 subsample한다.
3. short는 50th frame, long은 200th frame 이상인 마지막 view를 평가한다.
4. initial-to-final 뒤 reverse trajectory를 붙이고 return에서 10 frame마다
   평가한다.
5. Tanks-and-Temples advanced 6 scenes는 첫 50 frames로 cycle을 만들고
   subsampling 없이 매 frame 평가한다.
6. 논문과 동일하게 CFG 3, point-map `σ=0.03`, surfel radius `α=0.2`의 실제
   code mapping을 확인한 별도 official-reproduction config를 만든다.
7. LPIPS/PSNR/SSIM/FID와 DUSt3R pose 기반 Rdist/Tdist를 구현한다.
8. VMem K=4/K=17과 temporal/camera-distance/FOV retrieval, 공개 가능한 다른
   baseline을 같은 split에서 비교한다.
9. 현재 recent-8 CUT3R write 제한과 navigation NMS 변경을 원 논문 경로로
   되돌린 별도 branch/config에서 실행한다.
10. 논문 Table 1–4 수치와 평균, sample 수, 실패율을 함께 비교한다.

## 0.2 장기 영상 교정 실험

최초 57-run이 대부분 2–4 frames인 micro-ablation이었다는 문제를 확인한 뒤,
별도의 long-form local cycle을 실제 실행했다. 이 결과는 이전 짧은 test보다 연구
방향 판단에 우선한다.

| Trajectory | Context | Frames | Return paired PSNR 평균 | Final exact PSNR |
| --- | --- | ---: | ---: | ---: |
| 90° yaw-and-return | surfel | 127 | 22.63 dB | 19.65 dB |
| 90° yaw-and-return | recent | 127 | 14.29 dB | 9.88 dB |
| multi-segment exact reverse | surfel | 433 | 18.97 dB | 25.25 dB |
| multi-segment exact reverse | recent | 433 | 12.30 dB | 6.62 dB |

127-frame cycle은 10도 yaw를 9회 실행한 뒤 정확히 역방향으로 돌아왔다.
433-frame cycle은 세 translation segment와 두 90도 turn으로 216-frame outbound를
만들고 모든 camera command를 역순·역부호로 실행했다.

평균 paired PSNR은 return의 모든 frame을 같은 outbound camera pose의 frame과
비교한 값이다. Endpoint 하나만 높아도 중간 revisit가 망가질 수 있기 때문에
endpoint PSNR과 분리했다.

실제 영상에서는 다음 현상이 확인됐다.

- 두 context mode 모두 novel region exploration 중 구조적으로 붕괴했다.
- Surfel도 붕괴를 방지하지는 못했다.
- Surfel은 과거 view가 다시 보이는 pose에 도달할 때 원래 scene을 반복적으로
  복구했다.
- Recent는 drift한 frame만 context로 사용하면서 원래 scene을 잃었고, exact
  initial pose로 돌아와도 복구하지 못했다.
- 433-frame recent는 명백하게 다른 실내 scene으로 drift했지만 자동 validity
  heuristic은 failure를 0건으로 판정했다.

따라서 현재 증거는 “memory를 켜면 exploration이 더 빨리 무너진다”보다 반대에
가깝다. Backbone collapse는 두 mode에서 발생하지만, surfel memory는 exact/known
view recovery에 실질적으로 기여한다.

이 결과에 따라 reliability 연구의 초점도 수정한다. Primary failure는
**autoregressive backbone collapse**이고, spatial memory는 이를 완전히 막지 못하지만
revisit recovery를 제공한다. Reliability-aware read/write의 목적은 VMem을
대체하는 것이 아니라 다음 두 문제를 줄이는 것이어야 한다.

1. 이미 붕괴한 generated frame을 trusted spatial memory로 승격하지 않는다.
2. 복귀 시 clean past view와 corrupted descendant가 함께 검색될 때 clean memory를
   우선한다.

장기 영상과 분석은 `experiments/results/long_cycle/`에 있고 Git에서는 제외한다.
이 실험도 Oxford 단일 이미지와 synthetic camera command를 사용했으므로
paper-length local diagnostic이지 공식 benchmark reproduction은 아니다.

## 1. 현재 결과가 말해주는 것

이번 실험의 가장 중요한 결론은 다음과 같다.

> Spatial relevance와 memory reliability는 서로 다른 문제다.

VMem의 surfel routing은 긴 exact revisit에서 유용했다. Movement 4회 후 복귀하는
조건에서 surfel은 `30.40 dB`, recent baseline은 `20.94 dB`, revisit에서 spatial
memory를 끈 조건은 `18.81 dB`였다. 따라서 spatial memory를 제거하고 recent frame만
사용하는 것은 적절한 해결책이 아니다.

반면 pose와 FOV 관점에서 선택 가능한 후보라도 그 안의 내용이 잘못되어 있으면
결과가 크게 망가졌다.

- 2 scenes × 3 seeds에서 correct memory는 평균 `31.04±0.51 dB`였다.
- wrong memory는 `14.07±4.63 dB`로 평균 `16.97 dB` 하락했다.
- correct memory와 wrong distractor를 함께 준 조건도 `14.05±4.61 dB`였다.
- fallback이 없었던 네 scene/seed pair만 분리해도 하락 폭은 `14.59 dB` 이상이었다.
- 가장 관련 높은 latent slot 하나만 잘못되어도 효과가 포화됐다.
- CLIP embedding만 바꾼 경우 영향이 거의 없었지만 latent만 바꾸면 전체
  latent+CLIP corruption과 같은 수준으로 악화됐다.
- pose/intrinsics 불일치는 latent corruption보다 더 크게 악화됐다.

즉 현재 모델은 spatially relevant한 후보를 찾을 수는 있지만, 후보의 latent가
신뢰할 만한지 판별하거나 conflicting memory를 거부하는 능력은 약하다. 특히
correct memory가 함께 있어도 wrong latent 하나의 영향을 막지 못했다.

추가로 같은 최종 90도 pose를 만들 때 generation call 수가 많을수록 결과가
악화됐다. `90×1`을 기준으로 `45×2`, `30×3`, `15×6`의 최종 output은 각각
`12.31`, `10.62`, `7.66 dB`였고, 수동 검토에서도 호출 수에 따라 구조 붕괴가
심해졌다. 이는 memory 문제와 별개인 backbone autoregressive drift가 존재함을
보여준다.

한 번 잘못 생성된 revisit frame을 memory에 기록한 뒤 intervention을 끄더라도
후속 frame이 그 frame과 descendant를 다시 선택했다. Wrong 조건은 오염 직후
`22.20 dB`, 정상 generation 두 번 후에도 `22.14 dB`였다. 따라서 read 단계의
잘못된 memory 사용뿐 아니라 write 단계의 오염 전파도 별도의 문제다.

## 2. 현재 결과로 주장할 수 없는 것

이번 결과는 controlled stress test이며 자연 발생 오류의 빈도를 측정한 것은 아니다.
Route pose는 유지하면서 선택된 memory의 latent 또는 conditioning 일부를 의도적으로
교체했다. 따라서 현재 단계에서 다음과 같이 과도하게 주장하면 안 된다.

- VMem이 공식 benchmark에서 전반적으로 실패한다.
- wrong memory가 실제 trajectory에서 현재 intervention과 같은 빈도로 발생한다.
- novel-yaw output 중 어느 context mode가 더 정확하다.
- exact revisit PSNR이 높으므로 3D 공간 전체가 일관적이다.
- proposed reliability mechanism이 이미 기존 VMem보다 낫다.

Novel-yaw와 partial-overlap에서는 intervention에 대한 민감도만 확인했다. 해당
view의 ground truth가 없으므로 출력 사이 PSNR은 정확도 순위가 아니다. 공식
RealEstate10K/Tanks-and-Temples 설정과 LPIPS/SSIM, DUSt3R 기반 rotation/translation
metric 검증도 아직 필요하다.

## 3. 가장 유망한 연구 질문

가장 논문 가능성이 높은 질문은 다음과 같다.

> 생성 기반 spatial memory에서 relevance와 reliability를 분리하고, 신뢰할 수 없는
> memory를 read와 write 양쪽에서 거부하면 revisit consistency와 장기 rollout을
> 개선할 수 있는가?

핵심은 더 복잡한 retrieval을 만드는 것이 아니라 다음 세 결정을 분리하는 것이다.

1. **Where to retrieve:** 현재 pose/FOV와 공간적으로 관련된 후보는 무엇인가?
2. **Whether to trust:** 관련 후보 중 실제 생성 conditioning으로 사용할 만큼
   신뢰할 수 있는 것은 무엇인가?
3. **Whether to write:** 새로 생성한 frame을 장기 memory에 기록해도 안전한가?

VMem은 첫 번째 질문에는 강점이 있다. 이번 결과는 두 번째와 세 번째 질문이
명시적으로 다뤄져야 한다는 근거를 제공한다.

## 4. 우선순위 1: Reliability-aware memory read

### 가설

Pose/FOV relevance와 별도의 reliability score를 사용하면 correct memory는 유지하면서
wrong 또는 conflicting latent의 영향을 줄일 수 있다.

### 최소 방법

기존 surfel relevance로 top-k 후보를 얻은 뒤 각 후보에 reliability score
`q_i`를 부여한다. 최종 conditioning weight는 단순 relevance가 아니라 다음처럼
분리한다.

```text
weight_i = spatial_relevance_i × reliability_i
```

신뢰도가 threshold보다 낮으면 해당 slot을 제거하거나 recent/initial context로
fallback한다. 모든 후보가 낮으면 spatial memory를 억지로 사용하지 않고
`reject`를 명시적으로 선택한다.

### 먼저 검사할 reliability signal

복잡한 학습 모델보다 계산 가능한 신호부터 ablation한다.

- **Cross-memory agreement:** 비슷한 pose/FOV 후보들의 latent 또는 decoded appearance가
  서로 일치하는가?
- **Temporal consistency:** 후보를 사용한 one-step prediction이 직전 정상 frame과
  비정상적으로 단절되는가?
- **Cycle/reprojection consistency:** 후보의 geometry를 target pose로 옮겼다가
  되돌렸을 때 appearance/feature가 유지되는가?
- **Uncertainty under context dropout:** 후보 slot을 하나씩 제외했을 때 output이
  크게 변하는가?
- **Generation-quality cue:** black/saturation뿐 아니라 구조 붕괴, repeated texture,
  depth inconsistency를 탐지할 수 있는가?
- **Memory provenance:** 실제 입력에서 시작한 trusted frame인지, 생성된 frame인지,
  몇 단계의 autoregressive descendant인지 기록한다.

현재 component ablation상 CLIP-only corruption은 영향이 거의 없었으므로 CLIP
similarity만으로 reliability를 정의하는 것은 우선순위가 낮다. Latent consistency,
geometry consistency, provenance가 더 중요한 후보이다.

### 필수 baseline

- original VMem surfel relevance
- recent
- initial-only
- no spatial memory
- relevance + fixed rejection threshold
- reliability only
- relevance × reliability
- oracle rejection

Oracle은 어떤 후보가 intervention으로 손상됐는지 알고 이를 제거하는 상한선이다.
Oracle조차 개선이 없다면 reliability estimator보다 integration 구조를 먼저
바꿔야 한다.

## 5. 우선순위 2: Robust integration과 conflicting-memory rejection

### 가설

Correct memory가 있어도 wrong latent 하나가 결과를 지배하는 이유는 후보 선택뿐
아니라 multi-context integration이 outlier에 취약하기 때문이다.

### 연구할 방법

- slot별 hard rejection
- reliability 기반 soft gating
- top-k latent의 robust aggregation
- 후보 간 agreement가 낮을 때 single trusted memory만 사용
- correct와 distractor가 충돌하면 conditioning strength를 낮추는 adaptive guidance
- spatial memory와 recent context 사이의 uncertainty-aware mixture

### 핵심 평가

`correct_plus_wrong`이 가장 중요한 조건이다. 단순히 wrong-only 성능을 높이는
것보다 다음 두 조건을 동시에 만족해야 한다.

1. Correct-only 성능을 거의 손상시키지 않는다.
2. Correct+wrong 결과를 correct-only에 가깝게 복구한다.

Wrong slot 수 1개에서 이미 효과가 포화됐으므로 평균적인 noise robustness보다
단일 high-impact outlier를 억제하는 설계가 필요하다.

## 6. 우선순위 3: Memory write validation과 contamination 방지

### 가설

생성된 모든 frame을 같은 신뢰도로 memory에 즉시 기록하는 구조가 일회성 오류를
self-propagating contamination으로 바꾼다.

### 최소 방법

Memory를 다음 두 계층으로 분리한다.

- **Trusted memory:** 실제 입력, 검증된 keyframe, 높은 consistency를 반복 확인한 frame
- **Provisional memory:** 새로 생성되어 아직 검증되지 않은 frame

Provisional frame은 즉시 영구 memory로 승격하지 않는다. 이후 관측이나 cycle
consistency로 확인될 때만 trusted memory로 commit한다. 낮은 품질 또는 높은
uncertainty frame은 quarantine하거나 짧은 temporal buffer에만 둔다.

### 비교할 write policy

- write-all: 현재 방식
- input-only write
- periodic keyframe write
- reliability threshold write
- delayed commit
- descendant-depth 제한
- oracle clean-frame write

### 주요 metric

- contamination이 유지되는 generation step 수
- 오염된 frame 또는 descendant가 선택되는 비율
- 첫 failure 이후 정상 성능으로 회복하는 데 필요한 step 수
- correct trajectory에서 memory coverage가 감소하는 정도
- extra latency와 memory overhead

Read gating과 write gating을 반드시 분리 ablation해야 한다. Read만 고쳐도 이미
기록된 오염 memory가 남을 수 있고, write만 고쳐도 현재 call에서 선택된 wrong
memory의 영향은 막지 못한다.

## 7. 우선순위 4: Backbone drift와 memory failure의 공동 분석

Rotation accumulation 결과는 같은 최종 pose라도 호출 횟수가 늘면 backbone이
붕괴할 수 있음을 보여준다. 반면 exact revisit gap 14 calls에서도 초기 memory를
다시 찾으면 약 `30.44 dB`로 복구됐다. 따라서 memory는 backbone drift를 일부
복구하지만 novel view에서는 복구 여부를 아직 알 수 없다.

연구 방향은 backbone 전체를 교체하는 것보다 먼저 다음을 검사하는 것이다.

- 같은 pose 경로에서 generation call 수만 변화시키는 controlled schedule
- keyframe 간 큰 motion과 작은 motion 누적 비교
- drift가 감지될 때 trusted memory로 re-anchor
- rotation/translation별 별도 failure boundary
- memory read 전과 후의 구조 consistency 변화

이 방향은 reliability-aware memory의 보조 축으로 두는 것이 좋다. Backbone
교체 자체를 주제로 삼으면 VMem-specific contribution이 약해질 수 있다.

## 8. 제안하는 단계별 연구 계획

### Phase A: Natural failure mining

Controlled corruption이 아니라 정상 rollout에서 unreliable memory가 자연스럽게
생기는 조건을 찾는다.

- 공식 trajectory와 추가 long-horizon trajectory 실행
- scene/seed/motion scale 확대
- 모든 generated frame에 provenance와 descendant depth 기록
- manual label과 geometry/quality heuristic을 함께 저장
- 첫 구조 붕괴 시점, memory write 시점, 이후 retrieval 시점을 연결

목표는 “wrong memory가 실제로 얼마나 자주 생기는가”와 “어떤 observable signal이
failure 전에 나타나는가”를 측정하는 것이다.

### Phase B: Oracle experiments

모델을 학습하기 전에 개입 상한선을 확인한다.

- oracle wrong-memory removal
- oracle clean-memory-only routing
- oracle write rejection
- read-only oracle, write-only oracle, read+write oracle

Oracle에서 개선 폭이 충분해야 learned reliability estimator를 개발할 가치가 있다.

### Phase C: Training-free reliability baseline

Cross-memory agreement, provenance, geometry consistency를 결합한 단순 score를
구현한다. Threshold는 한 scene에서 맞추고 다른 scene/seed에서 검증해 test leakage를
피한다.

### Phase D: Learned reliability estimator

Training-free signal이 failure와 상관관계를 보일 때만 작은 reliability head를
학습한다. 입력 후보는 spatial relevance, pose distance, FOV overlap, latent
agreement, geometry consistency, provenance, descendant depth이다. Backbone과
생성 architecture는 고정하고 reliability prediction만 학습한다.

### Phase E: Official evaluation

다음 축을 분리해 보고한다.

- rollout validity
- retrieval correctness
- reliability detection
- exact revisit consistency
- novel-yaw/partial-overlap consistency
- contamination propagation
- official pose/appearance metric

단일 revisit score로 합치지 않는다.

## 9. 가장 중요한 실험 표

향후 핵심 결과 표는 다음 구조가 적합하다.

| Method | Correct-only | Wrong-only | Correct+wrong | Natural long rollout | Contamination recovery | Reject AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent |  |  |  |  |  | N/A |
| Original VMem |  |  |  |  |  | N/A |
| VMem + read gate |  |  |  |  |  |  |
| VMem + write gate |  |  |  |  |  |  |
| VMem + read/write gate |  |  |  |  |  |  |
| Oracle |  |  |  |  |  | 1.0 |

Correct-only 성능, corrupted-memory robustness, 자연 trajectory 성능을 모두 보여야
한다. Stress test만 좋아지고 clean 성능이 하락하는 방법은 채택하지 않는다.

## 10. 성공 및 중단 기준

### 계속할 기준

- natural rollout에서도 low-quality memory가 후속 generation을 오염시킨다.
- failure 전에 계산 가능한 reliability signal이 나타난다.
- oracle rejection이 original VMem보다 의미 있게 개선된다.
- learned 또는 training-free gate가 correct-only 성능을 유지한다.
- correct+wrong 조건에서 correct output에 가까운 결과를 회복한다.
- read/write gating이 scene과 seed가 바뀌어도 유지된다.

### 방향을 바꿀 기준

- natural trajectory에서는 unreliable memory가 거의 선택되지 않는다.
- oracle로 wrong memory를 제거해도 novel-view consistency가 개선되지 않는다.
- failure가 memory 조건과 무관하게 backbone collapse로 먼저 발생한다.
- reliability estimator가 pose/FOV 또는 scene identity만 학습한다.
- recent baseline이 공식 trajectory의 모든 조건에서 더 낫다.

## 11. 권장 연구 주제

현재 증거에 가장 잘 맞는 주제는 다음과 같다.

> **Reliability-Aware Read and Write for Generative Spatial Memory**

핵심 contribution 후보는 세 가지다.

1. Relevance와 reliability를 분리하는 generative spatial-memory failure formulation
2. Conflicting-memory와 contamination을 측정하는 revisit stress benchmark
3. Unreliable memory를 read/write 양쪽에서 제한하는 reliability-aware mechanism

첫 구현은 큰 architecture 변경보다 **oracle rejection → training-free read gate →
write quarantine** 순서가 적절하다. 이 순서라면 각 단계에서 연구 가설이 실제로
성립하는지 확인하면서 불필요한 모델 개발을 줄일 수 있다.
