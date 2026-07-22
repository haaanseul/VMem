<div align="center">
<img src="assets/title_logo.png" width="200" alt="VMem Logo"/>
<h1>VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory</h1>

<p align="center">ICCV 2025 ⭐ <strong>highlight</strong> ⭐</p>


<a href="https://v-mem.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project%20Page-gray.svg"></a>
<a href="http://arxiv.org/abs/2506.18903"><img src="https://img.shields.io/badge/%F0%9F%93%84%20arXiv-2506.18903-B31B1B.svg"></a>
<a href="https://huggingface.co/liguang0115/vmem"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a>
<a href="https://huggingface.co/spaces/liguang0115/vmem"><img src="https://img.shields.io/badge/%F0%9F%9A%80%20Gradio%20Demo-Huggingface-orange"></a>

[Runjia Li](https://runjiali-rl.github.io/), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), [Tomas Jakab](https://www.robots.ox.ac.uk/~tomj/)
<br>
<br>
[University of Oxford](https://www.robots.ox.ac.uk/~vgg/)
</div>

<p align="center">
  <img src="assets/demo_teaser.gif" width="100%" alt="Teaser" style="border-radius:10px;"/>
</p>

<!-- <p align="center" border-radius="10px">
  <img src="assets/benchmark.png" width="100%" alt="teaser_page1"/>
</p> -->

# Overview

`VMem` is a plug-and-play memory mechanism of image-set models for consistent scene generation.
Existing methods either rely on inpainting with explicit geometry estimation, which suffers from inaccuracies, or use limited context windows in video-based approaches, leading to poor long-term coherence. To overcome these issues, we introduce Surfel Memory of Views (VMem), which anchors past views to surface elements (surfels) they observed. This enables conditioning novel view generation on the most relevant past views rather than just the most recent ones, enhancing long-term scene consistency while reducing computational cost.


# :wrench: Installation

```bash
conda create -n vmem python=3.10
conda activate vmem
pip install -r requirements.txt
```


# :rocket: Usage

You need to properly authenticate with Hugging Face to download our model weights. Once set up, our code will handle it automatically at your first run. You can authenticate by running

```bash
# This will prompt you to enter your Hugging Face credentials.
huggingface-cli login
```

Once authenticated, go to our model card [here](https://huggingface.co/liguang0115/vmem) and enter your information for access.

We provide a demo for you to interact with `VMem`. Simply run

```bash
python app.py
```

## Headless revisit experiments

Spatial-memory revisit failures can be tested without Gradio. Start with a
plan-only check, which does not load the model or use the GPU:

```bash
conda activate vmem
python scripts/run_revisit_experiment.py \
  --scene test_samples/living_room.jpg \
  --context-mode surfel \
  --trajectory exact_revisit \
  --seed 42 \
  --memory-intervention correct \
  --plan-only \
  --output-dir experiments/results/plan_only
```

Remove `--plan-only` for generation. The runner supports `surfel`, `recent`,
and `initial_only` context modes; exact/novel-angle/partial-overlap, rotation
accumulation, and revisit-gap trajectories; and `correct`, `none`, `wrong`, or
`correct_plus_wrong` memory interventions. See all options with:

```bash
python scripts/run_revisit_experiment.py --help
```

Each run writes frames, an MP4, actual camera poses, strict JSON/JSONL memory
traces, a summary CSV, context images, contact sheets, and a manual-label
manifest below its `--output-dir`. Large outputs under `experiments/results/`
are ignored by Git. The intervention is applied only during the revisit phase
by default so outbound rollout and revisit conditioning can be separated.

The paper/code comparison and experiment definitions are in
[experiment_plan.md](experiment_plan.md). Only observations from completed
local runs are recorded in [failure_analysis.md](failure_analysis.md).

For moving this working tree to another server through GitHub, see [DEPLOYMENT.md](DEPLOYMENT.md).


## :heart: Acknowledgement
This work is built on top of [CUT3R](https://github.com/CUT3R/CUT3R), [DUSt3R](https://github.com/naver/dust3r) and [Stable Virtual Camera](https://github.com/stability-ai/stable-virtual-camera). We thank them for their great works.





# :books: Citing

If you find this repository useful, please consider giving a star :star: and citation.

```
@article{li2025vmem,
  title={VMem: Consistent Interactive Video Scene Generation with Surfel-Indexed View Memory},
  author={Li, Runjia and Torr, Philip and Vedaldi, Andrea and Jakab, Tomas},
  journal={arXiv preprint arXiv:2506.18903},
  year={2025}
}
```
