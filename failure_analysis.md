# Spatial-memory revisit dry-run failure analysis

작성일: 2026-07-23

이 문서는 실제로 완료한 GPU dry run만 기록한다. 결과 영상과 frame은 `experiments/results/` 아래에 있으며 Git에는 포함하지 않는다.

## 실행 조건

- scene: `test_samples/living_room.jpg`
- seed: 42
- trajectory: `exact_revisit`
- commands: `forward:1`, `backward:1`
- movement step: 0.1
- interpolation frames: 1
- context mode: `surfel` intervention 비교, `recent`/`initial_only` baseline 비교
- intervention phase: revisit only
- sampler: checked-in config의 50 steps
- 비교 intervention: `correct`, `wrong`, `correct_plus_wrong`

다섯 run 모두 외부 command 2개, generation call 2개, initial 포함 frame 3개였다. 최종 camera pose의 rotation error는 0도, translation error는 약 `1.49e-9`였다.

## 실제 결과

| Intervention | Revisit content sources | Initial 대비 PSNR | Initial 대비 MAE | 직전 outbound frame 대비 MAE | Fallback | Validity flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `correct` | `[0, 1, 1, 1]` | 30.4379 dB | 0.018480 | 0.036504 | 0 | 0 |
| `wrong` | `[1, 0, 0, 0]` | 22.1998 dB | 0.040964 | 0.004236 | 0 | 0 |
| `correct_plus_wrong` | `[1, 1, 0, 1]` | 22.1325 dB | 0.041483 | 0.004677 | 0 | 0 |

모든 revisit call의 intervention 전 route index는 `[0, 1, 1, 1]`이었고 selection reason은 `surfel_relevance`였다. 즉 route와 camera conditioning은 유지한 채 latent/CLIP content source만 바꿨다. `wrong`은 각 slot에 route index와 다른 source를 사용했고, `correct_plus_wrong`은 절반의 slot만 교체했다.

correct 대비 wrong은 PSNR이 8.24 dB 감소하고 MAE가 약 2.22배 증가했다. correct+wrong도 8.31 dB 감소하고 MAE가 약 2.24배 증가했다. contact sheet에서 두 intervention의 복귀 frame은 초기 frame보다 직전 outbound frame에 가까워 보였고, 수치상으로도 직전 frame MAE가 각각 0.00424와 0.00468로 매우 작았다. correct run의 같은 값은 0.03650이었다.

실행 시간은 correct 52.00초, strict wrong 51.63초, strict correct+wrong 51.64초였다. 세 run의 CUDA peak allocated memory는 각각 약 15,923 MiB였다.

### Context baseline

| Context mode | Revisit content sources | Initial 대비 PSNR | Initial 대비 MAE | Selection reason | Validity flags |
| --- | --- | ---: | ---: | --- | ---: |
| `surfel` | `[0, 1, 1, 1]` | 30.4379 dB | 0.018480 | `surfel_relevance` | 0 |
| `recent` | `[0, 1, 1, 1]` | 30.4379 dB | 0.018480 | `temporal_recent` | 0 |
| `initial_only` | `[0, 0, 0, 0]` | 30.5403 dB | 0.018111 | `initial_only` | 0 |

이 두-step trajectory에서는 surfel과 recent가 같은 stored frame을 선택하므로 결과도 동일했다. `initial_only`가 0.10 dB 높았지만 한 장면·한 seed의 미세한 차이이며 우위로 해석하지 않는다. context baseline의 차이를 보려면 후보가 K보다 많아지는 긴 trajectory가 필요하다. `none` intervention은 이 상태에서 recent route와 동일한 `[0,1,1,1]`이 되므로 중복 GPU run은 실행하지 않았다.

## Overnight follow-up matrix

동일한 checked-in 50-step sampler로 `experiments/results/overnight/` 아래 57개 고유 GPU run을 완료했다. `scripts/analyze_revisit_matrix.py`가 cross-condition metric과 평균을 `analysis.json`/`analysis.md`로 재계산한다.

### Distractor dose-response

| Wrong latent+CLIP slots | Exact revisit PSNR | MAE |
| ---: | ---: | ---: |
| 0 | 30.4379 dB | 0.018480 |
| 1 | 22.1502 dB | 0.041435 |
| 2 | 22.1311 dB | 0.041487 |
| 4 | 22.1998 dB | 0.040964 |

가장 관련 높은 첫 slot 하나만 바꿔도 약 8.29 dB 하락했고 2개나 4개에서 더 악화되지 않았다. 이 조건에서는 단일 wrong slot에서 효과가 이미 포화됐다.

### Conditioning component ablation

| 한 slot에 적용한 corruption | Exact revisit PSNR | MAE |
| --- | ---: | ---: |
| latent only | 22.1501 dB | 0.041433 |
| CLIP only | 30.4341 dB | 0.018491 |
| latent + CLIP | 22.1502 dB | 0.041435 |
| pose + intrinsics only | 20.5775 dB | 0.051407 |

CLIP-only는 correct와 사실상 같았지만 latent-only는 latent+CLIP과 같았다. 현재 stress test의 content sensitivity는 CLIP semantic embedding보다 latent에 의해 설명된다. Pose/intrinsics 불일치는 더 크게 악화됐다.

### 2 scenes × 3 seeds 재현성

| Scene | Seed | Correct | Wrong | Correct + wrong |
| --- | ---: | ---: | ---: | ---: |
| living_room | 42 | 30.44 | 22.20 | 22.13 |
| living_room | 43 | 30.59 | 13.41 | 13.40 |
| living_room | 44 | 30.57 | 18.12 | 18.08 |
| open_door | 42 | 31.52 | 10.16 | 10.15 |
| open_door | 43 | 31.62 | 11.14 | 11.13 |
| open_door | 44 | 31.52 | 9.41 | 9.39 |

전체 6 scene/seed pair 평균은 correct `31.04±0.51` dB, wrong `14.07±4.63` dB, correct+wrong `14.05±4.61` dB였다. 평균 하락은 각각 16.97 dB와 17.00 dB다.

`open_door` seed 42와 44는 세 intervention run에서 retrieval fallback이 1회씩 있었다. 세 조건 모두 fallback 0인 네 pair(`living_room` seed 42/43/44, `open_door` seed 43)만 분리해도 correct `30.81±0.47` dB, wrong `16.22±4.27` dB, correct+wrong `16.19±4.25` dB로 각각 14.59/14.62 dB 하락했다. 따라서 효과는 한 scene/seed나 fallback에만 의존하지 않지만 크기의 seed 분산은 크다.

### K보다 긴 exact trajectory의 context 비교

movement 4회 후 같은 경로로 4회 복귀했다.

| Context | Exact revisit PSNR | MAE | Final route |
| --- | ---: | ---: | --- |
| surfel | 30.4031 dB | 0.018485 | `[0, 7, 1, 2]` |
| revisit에서만 `none`/recent | 18.8080 dB | 0.06751 | recent 강제 |
| 전체 trajectory recent | 20.9390 dB | 0.051371 | `[4, 5, 6, 7]` |
| initial-only | 30.4907 dB | 0.01822 | `[0, 0, 0, 0]` |

짧은 run과 달리 surfel과 recent route가 실제로 분리됐다. 동일 surfel outbound 후 revisit에서만 spatial memory를 끈 `none` 조건이 11.60 dB 악화되어 spatial retrieval의 exact-return 기여가 확인됐다. 한편 initial-only도 surfel과 비슷하므로 exact revisit만으로 novel-view spatial consistency를 증명할 수는 없다.

### Novel yaw 10/20/30도

동일 target pose의 correct output을 기준으로 memory intervention output을 비교했다.

| Yaw | Correct vs wrong | Correct vs correct+wrong |
| ---: | ---: | ---: |
| 10° | 22.20 dB | 22.27 dB |
| 20° | 21.72 dB | 19.40 dB |
| 30° | 22.54 dB | 22.68 dB |

모든 yaw에서 intervention이 novel output을 크게 바꿨다. movement-4/yaw-20 조건에서는 surfel 최종 output 대비 recent가 18.67 dB, initial-only가 22.27 dB였다. 최종 route는 surfel `[0,4,2,1]`, recent `[5,6,7,8]`, initial-only `[0,0,0,0]`이었다. novel view ground truth가 없으므로 이 수치는 sensitivity이며 어떤 context가 더 정확한지를 뜻하지 않는다.

45도 partial-overlap에서도 correct output 대비 wrong은 24.41 dB, correct+wrong은 22.72 dB로 달라졌다. overlap이 줄어도 mixed distractor sensitivity가 남았지만, 역시 ground truth 정확도 비교는 아니다.

### Rotation accumulation과 revisit gap

모든 rotation schedule의 최종 camera pose는 정확히 90도였다. `90×1` 최종 output을 기준으로 `45×2`, `30×3`, `15×6`은 각각 12.31, 10.62, 7.66 dB였다. contact sheet 수동 검토에서 45×2의 마지막 frame, 30×3의 후반부, 15×6의 frame 3 이후에 벽·가구가 합쳐지거나 사라지는 구조 붕괴가 확인됐다. 자동 validity heuristic은 이를 하나도 flag하지 못했으므로 수동 manifest가 필요하다.

반면 exact return gap은 2/6/14 generation call에서 각각 30.44/30.50/30.44 dB였고 fallback과 자동 validity flag가 없었다. 중간 rollout 호출 수가 늘어도 surfel이 초기 pose memory를 다시 찾으면 exact view는 복구됐다.

### Persistent contamination

wrong intervention을 exact revisit 한 번에만 적용한 뒤 intervention을 끄고 정상 surfel retrieval로 yaw 15도와 return을 생성했다.

| Intervention | 오염 직후 exact revisit | 정상 follow-up 2회 후 exact revisit |
| --- | ---: | ---: |
| correct | 30.44 dB | 30.33 dB |
| wrong | 22.20 dB | 22.14 dB |
| correct+wrong | 22.13 dB | 22.08 dB |

wrong/mixed degradation이 intervention 종료 후에도 거의 그대로 남았다. final 정상 retrieval route는 세 조건 모두 `[2,3,3,3]`이었으며, index 2는 오염된 revisit frame이고 index 3은 그 이후 생성된 yaw frame이다. 잘못 생성된 frame이 memory write에 들어가 후속 retrieval/generation을 오염시키는 self-propagating contamination 후보다.

## Failure 분류

### Backbone autoregressive rollout failure

짧은 exact revisit에서는 선행하지 않았지만 rotation accumulation에서 관찰됐다. 같은 최종 90도 pose라도 호출 수가 1→2→3→6으로 늘수록 output이 급격히 달라졌고 수동 검토에서 구조 붕괴가 심해졌다. 반면 exact revisit gap은 14 calls까지 정상 복구돼, backbone drift와 memory-assisted exact recovery를 분리할 수 있었다.

### Retrieval/routing failure

초기 controlled comparison에서는 관찰되지 않았다. pre-intervention route가 동일하고 fallback도 없었다. 긴 exact trajectory에서는 surfel route가 recent보다 9.46 dB, revisit에서 spatial memory를 끈 조건보다 11.60 dB 높아 routing의 실질적인 이점이 확인됐다. 다만 일부 `open_door`와 rotation run에는 fallback이 있었으므로 해당 결과는 별도로 표시했다.

### Integration failure / memory sensitivity

관찰됐다. 동일 route/seed에서 한 latent slot만 바꿔도 결과가 크게 달라졌고 CLIP-only는 영향이 거의 없었다. 2 scenes × 3 seeds 및 fallback-free subset에서도 반복됐다.

### Contamination failure

관찰됐다. correct source가 일부 남아 있는 `correct_plus_wrong`도 correct-only를 방어하지 못했다. intervention을 끈 뒤 두 번 정상 생성해도 degradation이 유지되고 final retrieval이 오염된 frame과 그 descendant를 선택해 persistent contamination 후보가 됐다.

## 현재 판단

실행한 matrix는 다음 연구 판단 기준을 만족한다.

- wrong memory 하나로 exact revisit 결과가 크게 변했다.
- correct memory가 남아 있어도 conflicting content slot이 결과를 방해했다.
- 한 latent slot만으로 효과가 포화되고 여러 scene/seed에서 반복됐다.
- surfel retrieval은 긴 exact revisit에서 recent/none보다 우수했다.
- 같은 최종 rotation pose라도 생성 호출 수가 늘면 구조가 붕괴했다.
- 오염된 frame이 후속 정상 generation과 retrieval에 남았다.

따라서 “spatially relevant한 route가 항상 reliable한 content를 보장하지 않는다”는 가설은 계속 탐색할 근거가 강해졌다. 동시에 surfel memory 자체는 긴 exact revisit에서 유용하므로 단순 제거보다 reliability 판별이나 rejection mechanism이 더 맞는 연구 방향이다. 다만 intervention은 route pose와 content latent를 의도적으로 불일치시킨 controlled stress test이므로 자연 발생하는 VMem retrieval 오류의 빈도를 측정한 결과는 아니다.

## 아직 검증하지 못한 항목

- LPIPS/SSIM/FID 및 DUSt3R 기반 Rdist/Tdist 공식 평가
- 공식 RealEstate10K/Tanks-and-Temples trajectory와 ground truth novel-view 평가
- 자연 발생한 low-quality memory를 자동 탐지한 intervention이 아닌 failure 빈도
- 더 많은 scene/seed와 다른 movement scale에서의 통계
- latent reliability score 또는 rejection 기준 자체의 설계와 검증

현재 결과만으로 VMem 전체나 논문의 공식 benchmark에 대한 결론을 내리지 않는다.
