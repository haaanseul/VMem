# Deploying VMem on another server

This repository is prepared so code, configs, scripts, bundled CUT3R source, assets, and test samples can be pushed to GitHub. Runtime outputs, logs, Python caches, local environments, and model checkpoint files are intentionally ignored.

## Clone

```bash
git clone <your-github-url> VMem
cd VMem
```

## Environment

Use Python 3.10.

```bash
conda env create -f environment.yml
conda activate vmem
```

Alternatively, create the environment manually:

```bash
conda create -n vmem python=3.10
conda activate vmem
```

For the original CUDA/PyTorch setup:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/nightly/cu124 torch==2.7.0 torchvision==0.22.0
pip install -r requirements_no_torch_curope.txt
pip install --no-build-isolation -e ./extern/CUT3R/src/croco/models/curope
```

If PyTorch and the CUT3R `curope` extension are installed separately on the target server, install only the remaining Python packages:

```bash
pip install -r requirements_no_torch_curope.txt
```

The local development environment used for this working tree was:

```text
conda env: vmem
python: 3.10.20
torch: 2.7.0+cu118
torch CUDA build: 11.8
```

The checked-in `requirements.txt` currently installs `torch==2.7.0` from the PyTorch CUDA 12.4 index. If your target server is built around CUDA 11.8, install the matching PyTorch wheel first and then use `requirements_no_torch_curope.txt` for the rest of the packages.

## Hugging Face access

The model weights are downloaded from Hugging Face at runtime. Authenticate on the target server before running:

```bash
huggingface-cli login
```

Then request or confirm access to the VMem model card:

```text
https://huggingface.co/liguang0115/vmem
```

## Run the Gradio app

```bash
python app.py
```

If the server is remote, bind Gradio to an externally reachable host/port according to your server policy.

## Run revisit tests

```bash
python scripts/revisit_test.py --image test_samples/oxford.jpg --out revisit_outputs/oxford_revisit
```

Suite scripts are also available:

```bash
bash scripts/run_failure_suite.sh
bash scripts/run_originalish_suite.sh
bash scripts/run_oxford_revisit_original.sh
```

Generated files are written under `revisit_outputs/`, which is ignored by git.

## What should not be committed

Do not commit generated videos, traces, logs, downloaded model weights, or local virtual environments. If you need to preserve large generated artifacts, upload them to external storage or use Git LFS intentionally.
