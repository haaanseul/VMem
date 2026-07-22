# VMem 구현 메모: 논문 기준과 현재 작업 트리의 차이

마지막 확인: 2026-07-23

## 이 문서의 목적

이 저장소는 VMem 논문 구현을 기반으로 하지만, 기존 서버가 정상 동작하지 않아 다른 서버에서 실행하고 재방문 실패를 분석할 수 있도록 배포·안정성·테스트 관련 수정이 추가되어 있다. 이후 코드를 변경할 때는 아래 세 층을 구분한다.

1. **논문의 핵심 알고리즘**: 가능한 한 유지해야 하는 기준선
2. **서버 실행을 위한 변경**: 환경 및 장애 회피 목적
3. **재방문 테스트를 위한 변경**: 실패 재현, 계측, 평가 목적

단순히 현재 코드가 동작한다는 이유로 논문 알고리즘과 동일하다고 가정하지 않는다. 특히 context retrieval과 surfel memory write 경로를 수정할 때는 아래 차이를 다시 확인한다.

## 논문 기준 동작

논문 `VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory`의 핵심 흐름은 다음과 같다.

1. 단일 입력 이미지에서 시작해 사용자 카메라 경로를 따라 여러 뷰를 autoregressive하게 생성한다.
2. 각 surfel은 위치, 법선, 반지름과 함께 그 표면을 관찰한 과거 view index 집합을 저장한다.
3. 새 target camera 묶음의 평균 pose에서 surfel을 렌더링하고, 화면에 많이 나타나는 view index를 집계한다.
4. 중복 pose를 줄이는 NMS를 적용한 뒤 top-K 과거 view를 생성 context로 사용한다.
5. 새 뷰를 생성한 후, 검색된 과거 뷰와 새 뷰를 CUT3R로 함께 정렬해 point map을 얻는다.
6. point map을 surfel로 변환하고 기존 surfel과 병합하면서 새 frame index를 기록한다.
7. 논문의 경량 설정은 context `K=4`, 한 번에 생성하는 target `M=4`를 사용한다.

현재 구현도 surfel 기반 검색, view index 투표, NMS, CUT3R 기반 memory write라는 큰 구조는 유지한다.

## 현재 구현에서 추가·변경된 부분

### 1. 서버 및 Gradio 실행 안정화

- `vmem_runtime.log`와 `vmem_crash.log`를 추가해 시작, 예외, navigation 요청을 기록한다.
- Hugging Face Spaces 전용 `@spaces.GPU` 의존을 제거하고 일반 CUDA 서버에서 직접 실행한다.
- Gradio session state에 CUDA tensor를 보관하지 않고 CPU tensor를 사용한다.
- 하나의 전역 모델을 여러 요청이 동시에 변경하지 않도록 navigation lock과 `max_threads=1`을 사용한다.
- 성공한 요청 뒤 CUDA cache를 비우며, CUDA 오류가 발생하면 navigator state를 초기화한다.
- 서버 주소, 포트, share 여부, FPS, interpolation frame 수와 이동 크기를 환경 변수로 조절할 수 있다.
- 기본 바인딩은 `0.0.0.0:7860`, public Gradio share link는 기본 비활성이다.
- point-cloud/surfel 시각화는 headless 서버 충돌을 피하기 위해 기본 비활성이다.

이 변경들은 논문의 생성 방법을 개선하기 위한 것이 아니라 서버에서 안정적으로 실행하기 위한 것이다.

### 2. pipeline 방어 코드와 state 정합성 수정

- latent, encoder embedding, PIL frame, camera pose state를 명확히 분리해 저장한다.
- 선택된 context frame, 선택 이유, target index, surfel 수를 `debug_context_history`에 기록한다.
- surfel이 비어 있거나 위치·깊이·법선·반지름에 NaN/Inf가 있으면 제외한다.
- surfel 검색 결과가 없으면 최근 프레임 또는 마지막 프레임으로 fallback한다. 이는 논문의 정상 retrieval 경로에는 없는 장애 회피 동작이다.
- surfel-to-timestep index를 전체 영상의 절대 frame index에 맞추도록 수정했다.
- target trajectory를 4-frame chunk로 나누고, 모델의 고정 입력 크기를 맞추기 위해 endpoint pose를 padding한다. padding으로 생성된 가짜 target은 state에 저장하지 않는다.
- 첫 생성은 모델의 8-frame 입력 구조 때문에 `1 context + 7 targets` 형태로 padding될 수 있지만 실제 요청 프레임만 결과와 memory에 남긴다.
- 매 이동에서 실제로 생성된 pose와 frame 수를 기록해 UI pose와 undo 범위를 맞춘다.
- depth 시각화에서 NaN/Inf와 상수 depth를 안전하게 처리한다.

### 3. 논문과 결과 비교 시 특히 주의할 차이

- 논문은 memory write 시 **검색된 과거 view와 새 view를 함께** CUT3R에 넣는 방식을 설명한다.
- 현재 서버용 코드는 계산량과 state 불일치를 줄이기 위해 `context_num_frames + target_num_frames`, 즉 최근 최대 8개 frame만 scene reconstruction 입력으로 제한한다.
- 따라서 장기 실행에서 현재 memory write 입력은 논문이 설명한 retrieved context와 정확히 같지 않을 수 있다. 장기 재방문 품질을 해석하거나 개선할 때 가장 먼저 검토할 지점이다.
- 회전은 기본 surfel NMS retrieval을 사용하지만, 전진·후진 navigation은 `use_non_maximum_suppression=False`로 호출된다. 이 경우에도 surfel 후보는 사용하지만 pose 다양성 선택 방식은 논문 기본 경로와 다르다.
- surfel retrieval 실패 시 recent/latest fallback이 사용되면 해당 generation은 사실상 논문의 VMem retrieval 결과가 아니다. `memory_trace.json`의 `reason` 값을 함께 확인해야 한다.

### 4. 재방문 실패 분석 도구

`scripts/revisit_test.py`는 Gradio 없이 다음 항목을 기록한다.

- yaw-return, wide-yaw-return, forward-back, box-return trajectory
- 초기 pose 대비 최종 rotation/translation 오차
- 초기 이미지와 재방문 이미지 사이의 MSE, MAE, PSNR
- 각 generation의 context 선택 이유와 frame index
- 생성 frame, 영상, camera pose, memory trace, JSON report

관련 suite script는 원래 UI 버튼과 비슷한 이동, 넓은 회전, 반복 재방문과 360도 회전을 재현하기 위해 추가됐다. 이 테스트는 논문 공식 evaluation을 그대로 복제한 것이 아니라 현재 구현의 장기 일관성 실패를 빠르게 찾기 위한 regression 도구다.

새 `scripts/run_revisit_experiment.py`는 위 regression 도구와 별도로 다음 실험 계측을 추가한다.

- `surfel`, `recent`, `initial_only` context baseline
- exact/novel-angle/partial-overlap revisit, rotation accumulation, revisit gap
- `correct`, `none`, `wrong`, `correct_plus_wrong` memory intervention
- surfel 후보 relevance와 pose distance, routing index와 실제 content source index 분리 기록
- frame/context contact sheet, JSON/JSONL, CSV, manual-label manifest 저장

이 기능은 논문의 새로운 memory architecture가 아니다. `wrong` 계열 intervention은 surfel이나 checkpoint를 손상시키지 않고 선택된 route pose/K는 유지한 채 latent와 CLIP embedding의 source만 다른 저장 frame으로 교체한다. 따라서 routing과 generation integration의 민감도를 분리하기 위한 실험 장치로만 해석해야 한다. 기본 앱 경로에서는 실험 모드가 비활성이고 기존 `surfel`/`correct` 동작을 유지한다.

2026-07-23 최소 `living_room`, seed 42, forward 1회/return 1회 dry run에서는 correct memory가 30.44 dB였지만 strict wrong과 correct+wrong이 각각 22.20 dB, 22.13 dB였다. `recent`는 이 짧은 run에서 surfel과 같은 context를 골라 30.44 dB였고 `initial_only`는 30.54 dB였다. 모든 조건은 exact pose 복귀와 자동 rollout validity flag 0개를 기록했다. 이 결과는 한 장면·한 seed의 instrumentation 검증이며 일반화된 논문 성능 결론이 아니다. 상세 수치는 `failure_analysis.md`에만 기록한다.

### 5. 배포 및 의존성 변경

- CUT3R 소스를 `extern/CUT3R`에 포함해 다른 서버에서도 같은 코드로 실행할 수 있게 했다.
- PyTorch와 CUDA extension인 `curope`를 일반 Python 패키지와 분리해 설치하도록 변경했다.
- Python 3.10에서 dependency resolver가 오래된 source release로 backtracking하는 문제를 줄이기 위해 주요 패키지와 transitive dependency 버전을 고정했다.
- 모델 가중치는 Git에 넣지 않고 Hugging Face에서 받아 로컬 cache에 저장한다.
- PDF, runtime log, 생성 영상, 재방문 결과, checkpoint와 로컬 환경은 Git에서 제외한다.

## 2026-07-22 현재 실행 상태

- GPU: NVIDIA H100 PCIe 80GB
- Conda environment: `vmem`
- Python: 3.10.20
- PyTorch: 2.7.0+cu126
- Torchvision: 0.22.0+cu126
- Gradio: 6.15.2
- 필수 VMem, CUT3R, VAE, CLIP 가중치 다운로드 완료
- 서버: tmux session `vmem-server`, `0.0.0.0:7860`
- HTTP root와 Gradio config endpoint에서 200 응답 확인

주의: 문서의 과거 로컬 환경 설명과 `requirements.txt`의 CUDA 12.4 index, 현재 설치된 CUDA 12.6 wheel이 서로 다르다. 현재 서버는 동작하지만 새 서버 재현 시에는 설치된 torch build를 다시 확인해야 한다.

## 이후 작업 시 확인 순서

1. 변경이 논문 핵심 알고리즘, 서버 안정화, 테스트 계측 중 어디에 속하는지 먼저 표시한다.
2. context 선택을 바꾸면 `memory_trace.json`에서 `surfel_relevance`와 fallback 비율을 비교한다.
3. memory write를 바꾸면 retrieved context와 reconstruction 입력 frame이 일치하는지 확인한다.
4. 짧은 yaw-return만 보고 결론내리지 말고 wide-yaw, 반복 재방문, 360도 회전을 함께 실행한다.
5. 품질 변화와 camera pose 오차를 분리해서 본다. 최종 pose가 같아도 생성 이미지가 달라질 수 있다.
6. 논문과 다른 실험 설정이나 fallback이 적용된 결과는 공식 VMem 결과처럼 표현하지 않는다.
