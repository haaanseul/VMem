# VMem 실험 영상 안내

작성일: 2026-07-24

## 결론부터

장기 stress-test MP4 4개는 `experiments/results/long_cycle/` 아래에 있다.
논문 공개 demo code가 현재 수정 때문에 깨졌는지 확인하는 MP4 3개는
`experiments/results/paper_demo_ab/` 아래에 있다. 모두 실제 GPU로 생성했다.

기존 `dry_run`, `overnight`, `long_cycle_smoke`의 MP4 63개는 삭제했다. 이 영상들은
3–15 frames의 실행 검증용 영상이라 장기 동작을 눈으로 판단하는 데 적합하지 않았다.
다만 각 실험의 개별 frame, JSON/JSONL log, CSV, summary와 contact sheet는 남겨
두었으므로 기존 수치와 분석 근거는 유지된다.

## 무엇을 다시 만들었는가

처음 만든 57-run matrix는 memory intervention과 logging이 제대로 작동하는지
최소 비용으로 검사한 실험이었다. 전체 256 frames, 평균 4.49 frames/run이었으며
44/57개가 입력을 포함해 4 frames 이하였다. 이것은 논문의 장기 cycle 영상을
재현한 것이 아니었다.

이 문제를 교정하기 위해 다음 두 장기 trajectory를 새로 구현하고 `surfel`과
`recent`로 각각 다시 생성했다.

### 1. `yaw_cycle`

```text
yaw +10° × 9 → yaw -10° × 9
```

- 90도까지 회전한 뒤 같은 camera pose를 역순으로 복귀
- command당 7 interpolation frames
- 외부 command 18회
- 실제 generation call 36회
- 입력을 포함해 127 frames
- 영상 길이 10.58초

### 2. `long_reverse_cycle`

```text
forward × 6
→ yaw +5° × 18
→ forward × 6
→ yaw -5° × 18
→ forward × 6
→ 위 명령 전체를 역순·역부호로 실행
```

- translation과 rotation을 섞은 54-command outbound path
- 동일한 54 commands를 정확히 역연산해 복귀
- command당 4 interpolation frames
- 실제 generation call 108회
- outbound 216 frames, return 216 frames, 입력 포함 총 433 frames
- 영상 길이 36.08초

Return의 모든 frame은 outbound에서 camera pose가 가장 가까운 frame과 비교했다.
최대 pose matching 오차는 yaw cycle이 회전 `0.0194°`/이동 `0`, long cycle이
회전 `0.0201°`/이동 `1.49e-8`이었다. 따라서 같은 pose를 실질적으로 다시
방문하는 영상이다.

## 남겨둔 영상 4개

### A. `long_reverse_cycle_surfel/revisit.mp4`

- 위치:
  `experiments/results/long_cycle/long_reverse_cycle_surfel/revisit.mp4`
- 433 frames, 36.08초
- VMem의 surfel-indexed spatial retrieval 사용
- return 전체 paired PSNR 평균: `18.97 dB`
- final exact-pose PSNR: `25.25 dB`
- fallback: 0회

가장 먼저 확인할 영상이다. Novel region을 탐색하는 중간에는 구조가 심하게
무너지지만, return에서 과거 view와 가까운 pose에 도달하면 원래 Oxford 회랑을
반복적으로 복구한다. VMem이 backbone collapse를 막지는 못하지만 known-view
revisit recovery에는 도움을 준다는 근거다.

### B. `long_reverse_cycle_recent/revisit.mp4`

- 위치:
  `experiments/results/long_cycle/long_reverse_cycle_recent/revisit.mp4`
- 433 frames, 36.08초
- spatial memory 대신 가장 최근 frame만 context로 사용
- return 전체 paired PSNR 평균: `12.30 dB`
- final exact-pose PSNR: `6.62 dB`
- fallback: 0회

A와 camera path, scene, seed가 같은 직접 비교 영상이다. 대략 frame 34 이후
원래 회랑을 잃고 다른 밝은 실내 장면으로 drift하며, exact initial pose로
돌아와도 복구하지 못한다. A와 나란히 보는 것이 가장 중요하다.

### C. `yaw_cycle_surfel/revisit.mp4`

- 위치:
  `experiments/results/long_cycle/yaw_cycle_surfel/revisit.mp4`
- 127 frames, 10.58초
- surfel retrieval 사용
- return 전체 paired PSNR 평균: `22.63 dB`
- final exact-pose PSNR: `19.65 dB`
- fallback: 1회

이동 없이 회전만 누적했을 때의 단순화된 실험이다. 중간에 큰 검은 구조가
생기지만 return 후반에 원래 장면을 상당 부분 복구한다. Long cycle보다 짧아서
failure onset과 recovery를 빠르게 확인하기 좋다.

### D. `yaw_cycle_recent/revisit.mp4`

- 위치:
  `experiments/results/long_cycle/yaw_cycle_recent/revisit.mp4`
- 127 frames, 10.58초
- recent context 사용
- return 전체 paired PSNR 평균: `14.29 dB`
- final exact-pose PSNR: `9.88 dB`
- fallback: 0회

C와 동일한 회전 경로다. 같은 구조 붕괴가 시작된 뒤 spatial memory가 없어서
원래 장면으로 제대로 복구하지 못한다.

## 추천 확인 순서

1. 433-frame Surfel 영상 A
2. 433-frame Recent 영상 B
3. 두 영상의
   `cycle_pair_contact_sheet.jpg`를 비교
4. 더 짧게 failure/recovery를 보려면 127-frame C와 D 확인

각 run directory의 파일 의미는 다음과 같다.

- `revisit.mp4`: 전체 rollout 영상
- `frame_contact_sheet.jpg`: 전체 진행을 일정 간격으로 요약
- `cycle_pair_contact_sheet.jpg`: 같은 pose의 outbound/return frame을 좌우로 비교
- `selected_memory_contact_sheet.jpg`: 생성에 사용된 memory image 요약
- `cycle_pairs.csv`/`.json`: 모든 return frame의 pose match와 PSNR/MAE
- `trajectory.json`: planned command와 실제 camera pose
- `memory_trace.json`/`.jsonl`: 후보, 선택 context, fallback과 intervention 기록
- `summary.json`: run 설정, 실행 시간, frame/call 수와 최종 metric

## `dry run`은 무엇인가

이 작업에서 `dry run`이라는 표현이 두 종류로 섞여 사용됐다.

### `--plan-only`: 진짜 무실행 확인

`scripts/run_revisit_experiment.py --plan-only`는 model/checkpoint를 로드하지 않고,
GPU generation도 하지 않는다. CLI 설정과 생성할 camera command를 JSON으로
확인하는 일반적인 의미의 dry run이다. 영상은 생성하지 않는다.

### `experiments/results/dry_run`: 최소 GPU 실행

이 directory의 이름에 있는 `dry_run`은 실제로는 **GPU를 사용하는 최소 smoke
test**였다. `forward 1회 → backward 1회`로 입력을 포함해 3 frames만 생성해 다음을
확인했다.

- checkpoint와 pipeline이 실행되는가
- `surfel`, `recent`, `initial_only` mode가 동작하는가
- `correct`, `wrong`, `correct_plus_wrong` intervention이 적용되는가
- MP4, frame, pose, memory trace와 summary가 저장되는가

즉 이것은 환경·계측 검증용 실제 생성이지만, 영상 내용이나 장기 일관성을 평가하기
위한 실험은 아니다. 혼동을 피하기 위해 앞으로는 문서에서 이를
`minimal GPU smoke test`라고 부른다.

`long_cycle_smoke`도 같은 목적의 9-frame 실제 GPU 실행이었다. 장기 suite를
시작하기 전에 새 trajectory가 끝까지 실행되고 exact pose로 돌아오는지만 확인했다.

## 삭제한 영상

2026-07-24에 다음 MP4만 삭제했다.

- `experiments/results/dry_run/**/revisit.mp4`: 5개
- `experiments/results/overnight/**/revisit.mp4`: 57개
- `experiments/results/long_cycle_smoke/**/revisit.mp4`: 1개

총 63개, 약 3.56 MiB다. 이 파일들은 Git에 포함되지 않은 재생성 가능한 출력이다.
해당 정리 직후에는 위 장기 영상 4개만 남겼다. 이후 논문 데모 재현 점검을 위해
다음 3개를 추가했다.

- `paper_demo_ab/official_short/revisit.mp4`: 공식 원본이 완료한 9-frame run
- `paper_demo_ab/current_full_history/revisit.mp4`: crash-safe code + 공식 write
- `paper_demo_ab/current_recent8/revisit.mp4`: crash-safe code + 서버 기본 write

공식 원본과 pixel-identical한 9-frame 현재-code 중복 MP4는 삭제했다. 자세한
조건과 결과는 `PAPER_DEMO_REPRODUCTION.md`에 있다. 실험 directory와 분석 자료는
삭제하지 않았다.

## 논문 영상 재현 여부

남겨둔 433-frame 영상은 논문 Fig. 5의 270–401-frame sequence 및 Fig. 6의
400-frame 이상 cycle과 길이가 비슷하고, 나간 경로를 역순으로 돌아오는 Sec. 4.3의
개념을 따른다.

하지만 공식 재현 영상은 아니다.

- RealEstate10K/Tanks-and-Temples sequence가 아닌 Oxford 단일 이미지 사용
- dataset ground-truth camera가 아닌 synthetic command trajectory 사용
- 공식 test split과 10-frame subsampling 미사용
- LPIPS/SSIM/FID와 DUSt3R Rdist/Tdist 미계산
- checked-in CFG 2와 현재 서버용 memory-write 경로 사용

따라서 4개 영상은 `paper-length local cycle diagnostic`으로만 해석한다.
