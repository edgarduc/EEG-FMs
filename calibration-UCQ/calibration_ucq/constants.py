from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

EEGMAT_VERSION = "1.0.0"
EEGMAT_DIR = DATA_DIR / "eegmat" / EEGMAT_VERSION
EEGMAT_BASE_URL = f"https://physionet.org/files/eegmat/{EEGMAT_VERSION}"

CBRAMOD_REPO_URL = "https://github.com/wjq-learning/CBraMod.git"
CBRAMOD_HF_REPO = "weighting666/CBraMod"
CBRAMOD_WEIGHTS = "pretrained_weights.pth"

LABRAM_REPO_URL = "https://github.com/935963004/LaBraM.git"
LABRAM_WEIGHTS_URL = "https://github.com/935963004/LaBraM/raw/main/checkpoints/labram-base.pth"
LABRAM_WEIGHTS = "labram-base.pth"

REVE_MODEL_ID = "brain-bzh/reve-base"
REVE_POSITION_MODEL_ID = "brain-bzh/reve-positions"
