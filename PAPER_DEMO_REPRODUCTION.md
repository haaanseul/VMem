# VMem 논문 공개 데모 재현 점검

작성일: 2026-07-24

## 결론

현재 확인된 범위에서는 사용자가 수정한 서버 코드 때문에 공개 code path의 짧은
출력이 깨졌다고 보기 어렵다. 다만 아래 9/25-frame 테스트는 공식 홈페이지의
5.5초 Oxford 영상을 재현한 것이 아니다.

- 공식 원본과 현재 crash-safe 코드의 9-frame 결과는 pixel 단위로 완전히 같았다.
- 현재 코드에서 공식식 full-history memory write와 서버용 recent-8 write를
  25 frames까지 비교했을 때 둘 다 장면을 유지하고 exact initial pose를 복구했다.
- 반면 공식 공개 코드를 현재 환경에서 그대로 실행하면 VAE repository ID 문제로
  시작하지 못하고, ID만 호환되게 고쳐도 13-frame reconstruction에서
  `Octree.subdivide()` recursion crash가 발생했다.

즉 현재의 VAE ID 변경과 Octree/CUT3R 방어 코드는 데모를 망가뜨린 변경이 아니라
공개 원본을 현재 서버에서 실행 가능하게 만든 호환성·안정성 수정이다.

이전에 확인한 127/433-frame 붕괴는 공식 홈페이지의 Oxford path를 재현한 결과가 아니라,
generation을 의도적으로 오래 누적한 stress test의 결과다.

## 정정: 공식 홈페이지 영상은 165 frames다

공식 프로젝트 페이지의 기본 Oxford 비교 MP4를 직접 검사했다.

| 항목 | With VMem | Without VMem |
| --- | ---: | ---: |
| 파일 | `video_2_pair/final_video_1_compressed.mp4` | `video_2_pair/final_video_2_compressed.mp4` |
| 해상도 | 768×576 | 768×576 |
| FPS | 30 | 30 |
| 재생 시간 | 5.5초 | 5.5초 |
| decoded frames | 165 | 165 |

따라서 사용자가 말한 4–5초대 공식 영상은 실제로 공개되어 있다. 저장소의
`assets/demo_teaser.gif` 전체는 59.63초짜리 편집 영상이며, 첫 Oxford 비교 구간에
이 5초대 with/without sequence가 들어간다.

중요한 점은 5.5초가 생성이 짧다는 뜻이 아니라는 것이다. 논문과 공개 code는 한
번에 target `M=4` frames를 생성한다. 홈페이지 영상의 165 frames는
`1 + 41 × 4`와 정확히 일치하므로, initial frame을 포함한 약 41회 autoregressive
generation 결과일 가능성이 높다. 이 부분은 공개 pose log가 없어 frame count에
근거한 추론이다. 같은 165 frames를 공개 app의 기본 13 FPS로 저장하면 약
12.69초이며, 홈페이지는 30 FPS로 압축 재생한다.

논문에는 이보다 긴 qualitative 결과가 나온다.

- Fig. 5: in-the-wild 입력에서 선택 frame이 270–401까지인 long sequence
- Sec. 4.3/Fig. 6: 출발 경로를 역순으로 돌아오는 400-frame 이상 cycle
- RealEstate10K: 10-frame interval, 50th-frame short-term 및 200-frame 이상
  long-term 평가
- Tanks-and-Temples: 첫 50 frames로 cycle을 만들고 모든 return frame 평가

그러나 논문, 프로젝트 페이지, 공개 GitHub 중 어디에도 Oxford 홈페이지 영상의
camera pose sequence, UI button 순서, seed, raw generation FPS는 제공되지 않는다.
홈페이지에는 완성된 MP4만 있고, 공개 app은 사용자가 탐색을 마친 뒤의 pose만
저장할 수 있다.

또한 홈페이지 MP4는 768×576/30 FPS인 반면 공개 app config는
576×576/13 FPS다. 따라서 이 홈페이지 asset을 공개 app이 그대로 export한
동일 설정 영상이라고 단정할 수 없다.

## 무엇을 기준으로 삼았는가

다음 공식 source를 확인했다.

1. 논문 프로젝트 페이지가 연결한 GitHub `runjiali-rl/vmem`
2. 프로젝트 페이지가 연결한 Hugging Face Space `liguang0115/vmem`
3. 저장소의 논문 PDF `2506.18903v3 (2).pdf`

실행 기준 source는 공식 GitHub commit
`39291e4f272f6b4f270691d930926ab5930f942e`다. Hugging Face Space의 core
pipeline도 누적된 전체 `pil_frames`를 CUT3R reconstruction에 전달한다.

공식 저장소에는 웹사이트 teaser인 `assets/demo_teaser.gif`는 있지만, 이를 만든
camera command/trajectory JSON은 없다. App은 사용자가 버튼을 누른 뒤에야 camera
path를 저장할 수 있을 뿐, 논문 teaser의 기존 path를 제공하지 않는다.

따라서 홈페이지 MP4나 teaser GIF를 동일 seed·동일 trajectory로 pixel 재현하는
것은 공개 자료만으로 불가능하다. 아래 테스트는 공식 example image와 공식 app의
실제 Navigator/Pipeline/button code path가 현재 수정으로 달라졌는지 검사한
compatibility audit이다.

## 재현 조건

- scene: `test_samples/oxford.jpg`
- seed: 42
- resolution: 576×576
- context/target: `K=4`, `M=4`
- diffusion steps: 50
- CFG: 2.0
- navigation interpolation: button당 4 frames
- step size: 0.1
- FPS: 13
- checkpoint: `liguang0115/vmem/vmem_weights.pth`
- CUT3R: `liguang0115/cut3r/cut3r_512_dpt_4_64.pth`
- visualization: 파일 I/O에만 영향을 주므로 비활성

공식 UI의 `20° Veer` button은 `app.py`에서
`Navigator.turn_left(abs(y_angle // 2))`를 호출한다. 따라서 button label은
20°지만 실제 camera rotation은 click당 10°다. 이번 비교도 이 동작을 그대로
사용했다.

## 테스트 1: 공식 원본 그대로 실행

공식 source snapshot을 별도 directory에서 import해 현재 code와 섞이지 않도록
했다.

첫 model load에서 다음 hard-coded VAE ID가 더 이상 유효하지 않아 404가 발생했다.

```text
stabilityai/stable-diffusion-2-1-base
```

현재 code가 사용하는 다음 호환 repository로 ID만 교체했다.

```text
sd2-community/stable-diffusion-2-1
```

VAE architecture와 weight family를 바꾸기 위한 실험 수정이 아니라, 삭제된
repository 경로를 현재 접근 가능한 community mirror로 바꾼 것이다.

## 테스트 2: 공개 source의 9-frame compatibility path

Trajectory:

```text
20° Veer left button 1회 (실제 +10°)
→ 20° Veer right button 1회 (실제 -10°)
```

결과:

| 항목 | 공식 원본 + VAE ID 호환 |
| --- | ---: |
| frames | 9 |
| commands | 2 |
| runtime | 57.76초 |
| GPU peak allocated | 15,922.97 MiB |
| final rotation/translation error | 0° / 0 |
| initial-to-final PSNR | 25.2434 dB |

현재 crash-safe code에 공식 `full_history` memory write를 사용해 같은 run을 실행했다.
두 run의 9개 PNG hash가 frame별로 모두 같았고, 두 MP4의 FFmpeg PSNR은
`inf`였다. 즉 생성 pixel이 완전히 동일했다.

공식 pipeline은 첫 click에서 실제 새 frame 4개 외에 initial frame까지 포함한
5개를 Navigator에 반환한다. 현재 code는 새 frame 4개만 반환하도록 고쳤다.
이 차이는 Gradio video state의 initial-frame 중복을 없앤 것이며, pipeline에
저장된 9 frames와 생성 pixel에는 차이가 없었다.

## 테스트 3: 공개 source의 25-frame compatibility path 시도

Trajectory:

```text
20° Veer left × 3 (실제 +30°)
→ 20° Veer right × 3 (실제 -30°)
```

공식 원본은 13 images를 누적 CUT3R reconstruction에 전달한 단계에서
`Octree.subdivide()`가 같은 point set을 끝없이 세분화해 Python recursion limit을
초과했다.

```text
RecursionError: maximum recursion depth exceeded
```

따라서 공식 공개 code는 이 서버에서 25-frame demo를 완료하지 못했다. 현재
`utils/util.py`의 degenerate split/최소 extent 방어가 필요한 이유가 실제로
재현됐다.

## 테스트 4: 현재 crash-safe code의 25-frame A/B

공식 원본 crash를 막는 현재의 index, state, Octree 방어는 유지하고
CUT3R memory-write 입력만 비교했다.

| 조건 | Frames | Final PSNR | Runtime | Surfels |
| --- | ---: | ---: | ---: | ---: |
| 서버 기본 `recent_window` | 25 | 25.1872 dB | 146.58초 | 1,601 |
| 공식식 `full_history` | 25 | 25.2225 dB | 152.16초 | 1,201 |

두 조건 모두 exact pose로 돌아왔고 contact sheet에서 원래 Oxford 회랑을 정상적으로
복구했다. 두 전체 영상 사이 FFmpeg PSNR은 평균 `29.5264 dB`, 최저
`24.1643 dB`였다. Write 방식에 따른 차이는 있지만, 둘 중 하나만 구조적으로
붕괴하지는 않았다.

이 짧은 공식-style path에서는 recent-8 변경이 사용자가 본 심한 붕괴의 원인이
아니다.

## 현재 수정 중 품질과 관계있는 것

### 생성 pixel을 바꾸지 않은 것으로 확인된 수정

- Hugging Face Spaces `@spaces.GPU` 제거
- Gradio state를 CPU에 보관
- navigation lock과 `max_threads=1`
- runtime/crash log
- public share 비활성
- visualization 비활성
- 첫 click의 중복 initial frame 반환 제거

### 원본 crash/state bug를 고치는 수정

- 폐기된 VAE repository ID 교체
- padded target을 실제 trajectory state에 넣지 않음
- 실제 생성 pose와 UI pose 일치
- surfel absolute timestep index 정합성
- empty/NaN surfel과 degenerate Octree split 방어

공개 source 9-frame run과 현재 full-history 9-frame run이 pixel-identical하므로,
이 수정들이 같은 짧은 code path의 generation을 바꾸었다는 증거는 없다. 이
결과만으로 165-frame 홈페이지 path의 품질 동등성을 주장하지 않는다.

### 실제 algorithm behavior가 다른 수정

공식 원본:

```text
construct_and_store_scene(self.pil_frames)
```

현재 서버 기본:

```text
construct_and_store_scene(self.pil_frames[-8:])
```

이 변경은 긴 누적 reconstruction의 비용과 crash 가능성을 낮추기 위해 추가됐다.
이번 25-frame A/B에서는 final PSNR 차이가 0.04 dB 미만이었지만, 긴 trajectory의
surfel geometry에는 영향을 줄 수 있다.

이를 분리해 검사할 수 있도록 다음 두 mode를 지원한다.

- `recent_window`: 서버 안정성 기본값
- `full_history`: 논문 공개 demo의 memory-write 방식

## 어떤 영상을 보면 되는가

### 공개 source code로 실제 완료한 compatibility 영상

`experiments/results/paper_demo_ab/official_short/revisit.mp4`

- 9 frames
- 공식 GitHub pipeline/Navigator 사용
- VAE ID만 호환
- 홈페이지 165-frame MP4의 재현물이 아님

### 현재 code, 공식 full-history 방식

`experiments/results/paper_demo_ab/current_full_history/revisit.mp4`

- 25 frames
- 공식 원본의 Octree crash를 막는 현재 방어 코드 유지
- memory write는 공식 방식

### 현재 서버 기본 방식

`experiments/results/paper_demo_ab/current_recent8/revisit.mp4`

- 25 frames
- 동일 scene/seed/button path
- memory write만 최근 8 frames

9-frame 현재 full-history MP4는 공식 원본 영상과 pixel-identical이므로 중복
영상으로 보존하지 않는다.

## 공개 app 방식 서버 실행

공식 memory-write 동작으로 Gradio를 실행하려면 다음을 사용한다.

```bash
conda activate vmem
VMEM_SCENE_RECONSTRUCTION_MODE=full_history \
VMEM_NAVIGATION_INTERPOLATION_FRAMES=4 \
VMEM_NAVIGATION_STEP_SIZE=0.1 \
GRADIO_SHARE=0 \
python app.py
```

`full_history`는 command가 누적될수록 CUT3R 입력과 실행 시간이 커진다. 공식
원본처럼 Octree recursion crash가 나지는 않도록 현재 방어 코드는 유지되지만,
긴 session에서는 `recent_window`보다 느리다.

평가 완료 후에는 Gradio를 실행하지 않은 상태로 유지한다.
