# Spatial-memory revisit failure experiment plan

작성일: 2026-07-23

## 1. 목적과 작업 원칙

이 실험의 목적은 새로운 memory 방법을 제안하는 것이 아니라, 현재 VMem 구현에서 revisit failure를 최소 조건으로 재현하고 다음 네 유형을 분리하는 것이다.

1. backbone autoregressive rollout failure
2. memory retrieval/routing failure
3. 올바른 memory가 선택됐지만 생성에 반영되지 않는 integration failure
4. 잘못되거나 충돌하는 memory가 이후 생성을 오염시키는 contamination failure

중점 가설은 **spatially relevant한 memory가 항상 reliable한 것은 아니다**이다. 실험 코드는 모델 architecture와 checkpoint를 변경하지 않고 context routing과 입력 memory에만 통제된 intervention을 적용한다.

모든 생성물은 저장소 안 `experiments/results/`에 기록하고 Git에서 제외한다. 저장소 밖 파일, 시스템 환경, CUDA/driver, 다른 사용자의 프로세스는 변경하지 않는다. 기존 `vmem` Conda 환경만 사용한다.

## 2. 공식 실행 방법과 checkpoint

README의 공식 데모 실행은 다음과 같다.

```bash
conda activate vmem
huggingface-cli login
python app.py
```

코드가 런타임에 받는 checkpoint는 다음 두 개다.

- generator: `liguang0115/vmem`, `vmem_weights.pth`
- point-map predictor: `liguang0115/cut3r`, `cut3r_512_dpt_4_64.pth`

추가로 VAE와 OpenCLIP ViT-H-14 pretrained weight를 사용한다. 이 서버에는 필요한 weight가 Hugging Face cache에 준비되어 있지만 cache와 checkpoint는 수정하거나 Git에 포함하지 않는다.

## 3. 논문의 공식 evaluation과 revisit 설정

논문 PDF `2506.18903v3 (2).pdf`에서 확인한 설정은 다음과 같다.

### Generator 설정

- backbone: SEVA
- 원래 SEVA: context `K=17`, target `M=4`, 총 21 views
- 경량 VMem: LoRA로 fine-tune한 context `K=4`, target `M=4`
- main experiment는 경량 `K=4`, `M=4` 사용
- appendix inference 설정: CFG 3, point-map scaling factor 0.03, surfel radius의 alpha 0.2

### Short/long rollout

- RealEstate10K ground-truth camera trajectory를 10-frame 간격으로 subsample한다.
- short-term은 다섯 번째 생성 이미지, 즉 시작점에서 50 frames 떨어진 지점을 평가한다.
- long-term은 시작점에서 200 frames 이상 떨어진 마지막 이미지를 평가한다.

### Cycle trajectory revisit

- 원래 trajectory의 initial-to-final 경로를 생성한 뒤 같은 경로를 역순으로 따라 시작 pose로 복귀한다.
- RealEstate10K은 모든 test sequence의 return trajectory에서 10 frames마다 평가한다.
- Tanks-and-Temples는 advanced scene 6개에서 첫 50 frames로 cycle을 만들고 temporal subsampling 없이 매 frame 평가한다.
- 논문도 이 cycle이 단순하고 occlusion이 제한적이며, 기존 지표가 진정한 multi-view consistency보다 저수준 texture similarity를 주로 측정한다는 한계를 명시한다.

### 공식 비교 방법과 지표

비교 모델은 LookOut, GenWarp, MotionCtrl, ViewCrafter, SEVA이다. Retrieval ablation은 다음 네 가지다.

- temporal: 최근 K views
- camera distance: target과 camera pose가 가까운 K views
- field of view: target과 FOV overlap이 큰 K views
- VMem: surfel-indexed retrieval

지표는 다음과 같다.

- 전체 분포 품질: FID
- image consistency: LPIPS, PSNR, SSIM
- camera consistency: DUSt3R로 생성 view pose를 추정한 뒤 Rdist, Tdist

Rdist/Tdist는 첫 frame 기준 relative pose를 사용하고 translation은 가장 먼 frame 기준으로 정규화한다.

## 4. 현재 코드와 논문 설정의 차이

현재 코드는 논문의 큰 구조인 surfel render, view-index voting, NMS, CUT3R memory update를 유지하지만 다음 차이가 있다.

- `configs/inference/inference.yaml`은 `K=4`, `M=4`, `CFG=2`, surfel shrink factor 0.05를 사용한다. 논문 appendix의 CFG 3, point-map scale 0.03과 다르다.
- 첫 호출은 8-frame tensor shape를 맞추기 위해 `1 context + 7 padded targets`, 이후는 `4 context + 4 targets`로 실행된다. padding target은 state에 저장하지 않는다.
- 서버 안정화 수정으로 memory write의 CUT3R 입력을 최근 `context_num_frames + target_num_frames`, 즉 최대 8 frames로 제한했다.
- 논문은 retrieved past views와 newly generated views를 함께 CUT3R에 넣는다고 설명하므로, 현재 recent-8 reconstruction window는 장기 memory write에서 논문과 다를 수 있다.
- surfel retrieval 실패 시 recent frames 또는 latest frame fallback을 사용한다. 이 generation은 정상 VMem retrieval 결과와 분리해야 한다.
- rotation command는 기본 NMS를 사용하지만 forward/backward command는 현재 `use_non_maximum_suppression=False`로 호출된다.
- 기존 `scripts/revisit_test.py`는 yaw/forward/box return과 PSNR/MAE/MSE, context trace를 제공하지만 context baseline, controlled wrong memory, contact sheet, run-level CSV는 제공하지 않는다.
- 공식 dataset과 ground-truth trajectory가 저장소에 포함되어 있지 않다. 로컬 test image dry run은 공식 FID/LPIPS/Rdist/Tdist 재현이 아니라 failure instrumentation 검증 및 가설 탐색이다.

서버·테스트용 변경의 상세 배경은 `IMPLEMENTATION_NOTES.md`를 기준으로 한다.

## 5. 현재 context retrieval과 fallback 구조

`VMemPipeline.get_context_info()`의 현재 흐름은 다음과 같다.

1. 첫 frame만 있으면 index 0을 context로 사용한다.
2. target pose 묶음의 average pose에서 surfel을 렌더링한다.
3. 보이는 surfel의 `cos / (1 + depth)`를 해당 surfel이 기억하는 timestep에 누적한다.
4. relevance 후보를 camera geodesic distance로 정렬하고 NMS로 최대 K개를 선택한다.
5. 후보가 없으면 recent frames, 그래도 없으면 latest frame으로 fallback한다.
6. 선택 index로 pose, latent, CLIP embedding, intrinsics를 함께 읽어 generator condition을 만든다.

현재 trace는 최종 index와 선택 이유만 저장한다. 실험에서는 pre/post-intervention 후보, relevance, pose distance, route index와 실제 content source index를 분리해 저장한다.

## 6. 구현할 실험

### Context mode

- `surfel`: 현재 VMem retrieval
- `recent`: 최근 K frames를 시간 역순이 아닌 chronological order로 사용
- `initial_only`: 초기 frame을 고정 context로 사용한다. 모델의 고정 8-view shape를 유지해야 하는 후속 call에서는 초기 frame을 K slots에 반복한다.

### Trajectory

- `exact_revisit`: 짧은 forward-and-back 또는 turn-and-return으로 exact initial pose 복귀
- `novel_angle_revisit`: 같은 위치로 돌아온 뒤 yaw 10/20/30도 offset
- `partial_overlap`: 같은 위치에서 더 큰 yaw offset으로 초기 view와 일부만 overlap
- `rotation_accumulation`: 최종 yaw 90도를 `90×1`, `45×2`, `30×3`, `15×6`으로 만들어 command 수와 실제 generated frame 수 비교
- `revisit_gap`: 같은 최종 pose로 복귀하되 short/medium/long에 따라 중간 generation call 수를 변경

### Memory intervention

- `correct`: 선택된 route index와 같은 latent/embedding 사용
- `none`: spatial routing을 사용하지 않고 recent context로 대체
- `wrong`: plausible한 선택 pose/K는 유지하되 latent/CLIP embedding을 올바른 selection 밖의 least-relevant available frame에서 가져온다. 모델 state 자체는 변조하지 않는다.
- `correct_plus_wrong`: 고정 K slots 안에 correct source와 wrong source를 함께 배치한다. route pose는 유지하고 content source index를 별도로 기록한다.

`wrong`과 `correct_plus_wrong`은 tensor shape나 surfel state를 직접 손상시키지 않는 최소 intervention이다. “깨진 frame을 memory state에 영구 삽입”하는 contamination 실험은 이후 별도 단계로 남긴다. 현재 단계에서 이를 강제로 구현하면 routing failure와 state corruption을 분리하기 어렵다.

Intervention은 outbound exploration 동안 비활성화하고 revisit phase부터 적용할 수 있게 한다. 이렇게 해야 rollout 자체의 차이와 revisit conditioning의 차이를 분리할 수 있다.

## 7. 저장 및 평가 구조

각 run은 `experiments/results/<run_id>/` 아래에 다음을 저장한다.

```text
frames/
contexts/
contacts/
run_config.json
trajectory.json
memory_trace.json
memory_trace.jsonl
summary.json
summary.csv
manual_labels.csv
revisit.mp4
frame_contact_sheet.jpg
selected_memory_contact_sheet.jpg
```

기록 항목:

- scene, seed, context mode, trajectory, yaw/movement/interpolation/gap 설정
- command 수, generation call 수, 실제 생성 frame 수
- target pose, candidate indices, relevance, pose distance
- intervention 전후 route index, 실제 content source index
- fallback 여부와 context selection reason
- context image
- wall-clock 실행 시간과 CUDA peak allocated memory
- final pose error와 initial/final pixel metrics
- rollout validity heuristic: black, non-finite, saturation, abrupt frame change, low-variance/repetition 후보

자동 heuristic은 failure의 확정 판정이 아니다. `manual_labels.csv`에 failure step, 구조 붕괴, 반복 texture, contamination 여부를 사람이 기록할 수 있게 한다.

평가는 하나의 score로 합치지 않고 다음 네 묶음으로 저장한다.

1. rollout validity
2. retrieval correctness
3. revisit consistency: exact, novel yaw, partial overlap, gap
4. memory sensitivity: correct/none/wrong/correct+wrong

## 8. 수정 예정 파일

- `experiment_plan.md`: 본 계획
- `modeling/pipeline.py`: 실험용 context mode, intervention, 후보/relevance/pose-distance trace
- `scripts/run_revisit_experiment.py`: headless CLI runner
- `scripts/revisit_experiment_utils.py`: trajectory, metric, contact sheet, manifest helper
- `README.md`: headless 실행법과 결과 구조
- `.gitignore`: `experiments/results/`와 PDF 제외 규칙 유지
- `IMPLEMENTATION_NOTES.md`: 새 instrumentation이 논문 core가 아님을 추가 기록
- `failure_analysis.md`: 실제 dry run에서 관찰한 사실만 기록

핵심 model architecture, checkpoint, CUT3R source는 수정하지 않는다.

## 9. Dry-run 순서와 명령

먼저 CLI/trajectory를 GPU 없이 정적으로 검증한다.

```bash
conda run --no-capture-output -n vmem \
  python scripts/run_revisit_experiment.py --list-trajectories

conda run --no-capture-output -n vmem \
  python scripts/run_revisit_experiment.py \
  --scene test_samples/living_room.jpg \
  --context-mode surfel \
  --trajectory exact_revisit \
  --seed 42 \
  --movement-steps 1 \
  --interp-frames 1 \
  --memory-intervention correct \
  --plan-only \
  --output-dir experiments/results/plan_only
```

그 다음 한 scene, 한 seed, 짧은 trajectory로 실제 generation을 실행한다.

```bash
conda run --no-capture-output -n vmem \
  python scripts/run_revisit_experiment.py \
  --scene test_samples/living_room.jpg \
  --context-mode surfel \
  --trajectory exact_revisit \
  --seed 42 \
  --movement-steps 1 \
  --interp-frames 1 \
  --memory-intervention correct \
  --output-dir experiments/results/dry_run/surfel_correct
```

첫 GPU run이 정상 종료된 뒤에만 같은 scene/seed에서 `recent`, `initial_only` 및 `none`, `wrong`, `correct_plus_wrong`을 순차 실행한다. 공식 50-step sampler 설정을 기본으로 유지하며, 실행하지 못한 조건은 결과 문서와 PR에 명시한다.

## 10. 판단 기준

다음 패턴은 spatial-memory reliability 연구를 계속할 근거다.

- correct 대비 wrong memory에서 revisit 결과가 크게 변함
- correct와 wrong을 함께 넣었을 때 correct-only보다 악화
- surfel relevance와 pose distance는 그럴듯하지만 content source에 민감
- exact pose는 유지되지만 novel yaw/partial overlap에서 급락
- surfel retrieval은 정상인데 output이 selected context와 불일치

다음 패턴이면 memory보다 backbone 또는 평가 설계를 먼저 바꾼다.

- 모든 memory 조건에서 같은 step에 동일 붕괴
- wrong memory가 output에 거의 영향을 주지 않음
- recent baseline이 모든 조건에서 일관되게 해결
- 한 scene/seed에서만 재현되고 다른 최소 scene에서 사라짐
