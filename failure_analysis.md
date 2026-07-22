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

## Failure 분류

### Backbone autoregressive rollout failure

이 최소 조건에서는 선행하지 않았다. 정상 memory run은 짧은 exact revisit을 유지했고 black/NaN/saturation/abrupt-change heuristic에 걸린 frame이 없었다. 다만 한 scene과 두 generation call만으로 장기 backbone 안정성을 결론낼 수 없다.

### Retrieval/routing failure

관찰되지 않았다. 세 조건의 pre-intervention route가 동일했고 surfel retrieval이 성공했으며 fallback도 없었다. 여기서 발생한 차이는 retrieval 후보나 route pose를 바꾼 결과가 아니다.

### Integration failure / memory sensitivity

관찰됐다. 동일한 route pose와 seed에서 content source만 바꾸자 복귀 결과가 크게 달라졌다. 특히 결과가 직전 outbound frame과 거의 같아진 점은 모델이 conflicting content를 무시하지 못하고 생성에 반영했다는 후보 증거다.

### Contamination failure

단일 revisit generation 안의 distractor 간섭은 관찰됐다. correct source가 일부 남아 있는 `correct_plus_wrong`도 correct-only보다 크게 악화됐다. 그러나 이 intervention은 memory state를 영구 변조하지 않았고 이후 generation을 더 실행하지 않았으므로, 장기 memory contamination으로 부르지는 않는다.

## 현재 판단

이 dry run은 다음 연구 판단 기준 두 가지를 만족한다.

- wrong memory 하나로 exact revisit 결과가 크게 변했다.
- correct memory가 남아 있어도 conflicting content slot이 결과를 방해했다.

따라서 “spatially relevant한 route가 항상 reliable한 content를 보장하지 않는다”는 가설은 추가 실험 가치가 있다. 다만 이번 intervention은 route pose와 content latent/embedding을 의도적으로 불일치시킨 controlled stress test이므로, 자연 발생하는 VMem retrieval 오류의 빈도를 측정한 결과는 아니다.

## 아직 검증하지 못한 항목

- `none` intervention의 별도 실제 GPU run 및 K보다 많은 memory가 있는 구별 가능한 baseline 비교
- yaw 10/20/30도 novel-view와 partial-overlap revisit
- 90×1, 45×2, 30×3, 15×6 rotation accumulation
- medium/long revisit gap
- 여러 scene과 여러 seed에서의 재현성
- LPIPS/SSIM/FID 및 DUSt3R 기반 Rdist/Tdist 공식 평가
- broken-frame 삽입 후 후속 generation에 남는 장기 contamination
- `correct_plus_wrong`의 distractor 비율별 dose-response

현재 결과만으로 VMem 전체나 논문의 공식 benchmark에 대한 결론을 내리지 않는다.
