# calibration-UCQ

Calibration and uncertainty quantification experiments for frozen EEG foundation models on EEGMAT.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Each invocation runs one model only. EEGMAT is downloaded to `data/eegmat/1.0.0` if it is missing.

```bash
python3 run_experiment.py --model reve --seed 0
python3 run_experiment.py --model cbramod --seed 0
```

REVE is hosted in gated Hugging Face repositories. Request access on Hugging Face first, then authenticate with one of:

```bash
huggingface-cli login
export HF_TOKEN=hf_...
python3 run_experiment.py --model reve --seed 0 --hf-token hf_...
```

Useful debug run:

```bash
python3 run_experiment.py --model reve --seed 0 --max-subjects 6 --epochs 5 --ensemble-size 2
```

Outputs are one JSON file per run under `outputs/<model>/`.

## Dataset Task

This implementation treats MAT as the PhysioNet EEGMAT dataset. Labels are:

- `0`: background EEG before mental arithmetic
- `1`: EEG during mental arithmetic

Subjects are split disjointly as 60% train, 20% calibration, and 20% test using `--seed`.
