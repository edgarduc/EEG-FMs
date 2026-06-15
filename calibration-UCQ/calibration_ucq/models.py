from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .constants import (
    CBRAMOD_HF_REPO,
    CBRAMOD_REPO_URL,
    CBRAMOD_WEIGHTS,
    DATA_DIR,
    REVE_MODEL_ID,
    REVE_POSITION_MODEL_ID,
)


@dataclass(frozen=True)
class ExtractorInfo:
    name: str
    target_sfreq: float
    embedding_dim: int | None = None


class FrozenFeatureExtractor(nn.Module):
    info: ExtractorInfo

    def extract(self, eeg: torch.Tensor, channel_names: list[str]) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, eeg: torch.Tensor, channel_names: list[str]) -> torch.Tensor:
        self.eval()
        return self.extract(eeg, channel_names)


class REVEFeatureExtractor(FrozenFeatureExtractor):
    def __init__(
        self,
        *,
        model_id: str = REVE_MODEL_ID,
        position_model_id: str = REVE_POSITION_MODEL_ID,
        pooling: str = "flatten",
        device: torch.device,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("transformers is required for the REVE adapter. Install requirements.txt.") from exc

        self.info = ExtractorInfo(name="reve", target_sfreq=200.0)
        self.pooling = pooling
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
        self.position_model = AutoModel.from_pretrained(position_model_id, trust_remote_code=True).to(device)
        self.model.eval()
        self.position_model.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def extract(self, eeg: torch.Tensor, channel_names: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        eeg = eeg.to(device)
        positions = self.position_model(channel_names)
        if not torch.is_tensor(positions):
            positions = torch.as_tensor(positions)
        positions = positions.to(device=device, dtype=eeg.dtype).unsqueeze(0).expand(eeg.shape[0], -1, -1)
        output = self.model(eeg, positions)
        return _pool_output(output, self.pooling)


class CBraModFeatureExtractor(FrozenFeatureExtractor):
    def __init__(
        self,
        *,
        source_dir: Path | None = None,
        weights_path: Path | None = None,
        pooling: str = "flatten",
        channels: int = 22,
        segments: int = 4,
        points_per_patch: int = 200,
        device: torch.device,
    ):
        super().__init__()
        self.info = ExtractorInfo(name="cbramod", target_sfreq=200.0)
        self.pooling = pooling
        self.channels = channels
        self.segments = segments
        self.points_per_patch = points_per_patch
        self.chunk_points = segments * points_per_patch

        source_dir = source_dir or DATA_DIR / "model_sources" / "CBraMod"
        weights_path = weights_path or DATA_DIR / "model_weights" / "cbramod" / CBRAMOD_WEIGHTS
        _ensure_cbramod_source(source_dir)
        _ensure_cbramod_weights(weights_path)

        sys.path.insert(0, str(source_dir))
        try:
            from models.cbramod import CBraMod
        except ImportError as exc:
            raise RuntimeError(f"Could not import CBraMod from {source_dir}.") from exc

        self.model = CBraMod().to(device)
        state = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state)
        self.model.proj_out = nn.Identity()
        self.model.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def extract(self, eeg: torch.Tensor, channel_names: list[str]) -> torch.Tensor:
        del channel_names
        device = next(self.model.parameters()).device
        eeg = _match_channel_count(eeg.to(device), self.channels)
        chunks = _chunk_or_pad(eeg, self.chunk_points)
        embeddings = []
        for chunk in chunks:
            cbramod_input = chunk.reshape(chunk.shape[0], self.channels, self.segments, self.points_per_patch)
            embeddings.append(_pool_output(self.model(cbramod_input), self.pooling))
        return torch.cat(embeddings, dim=1)


def build_feature_extractor(
    model_name: str,
    *,
    device: torch.device,
    pooling: str = "flatten",
    cbramod_source_dir: Path | None = None,
    cbramod_weights_path: Path | None = None,
) -> FrozenFeatureExtractor:
    normalized = model_name.lower()
    if normalized == "reve":
        return REVEFeatureExtractor(pooling=pooling, device=device)
    if normalized == "cbramod":
        return CBraModFeatureExtractor(
            source_dir=cbramod_source_dir,
            weights_path=cbramod_weights_path,
            pooling=pooling,
            device=device,
        )
    raise ValueError(f"Unsupported model {model_name!r}. Choose one of: reve, cbramod.")


def _pool_output(output: Any, pooling: str) -> torch.Tensor:
    tensor = _output_to_tensor(output)
    if tensor.ndim == 2:
        return tensor
    if pooling == "mean":
        return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1]).mean(dim=1)
    if pooling == "flatten":
        return tensor.reshape(tensor.shape[0], -1)
    raise ValueError(f"Unsupported embedding pooling mode: {pooling}")


def _output_to_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("last_hidden_state", "pooler_output", "logits"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    for attr in ("last_hidden_state", "pooler_output", "logits"):
        value = getattr(output, attr, None)
        if torch.is_tensor(value):
            return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value
    raise TypeError(f"Cannot convert model output of type {type(output).__name__} to a tensor.")


def _match_channel_count(eeg: torch.Tensor, channels: int) -> torch.Tensor:
    if eeg.shape[1] == channels:
        return eeg
    if eeg.shape[1] > channels:
        return eeg[:, :channels]
    pad = channels - eeg.shape[1]
    return F.pad(eeg, (0, 0, 0, pad))


def _chunk_or_pad(eeg: torch.Tensor, chunk_points: int) -> list[torch.Tensor]:
    chunks = []
    for start in range(0, eeg.shape[-1], chunk_points):
        chunk = eeg[:, :, start : start + chunk_points]
        if chunk.shape[-1] < chunk_points:
            chunk = F.pad(chunk, (0, chunk_points - chunk.shape[-1]))
        chunks.append(chunk)
    return chunks


def _ensure_cbramod_source(source_dir: Path) -> None:
    if (source_dir / "models" / "cbramod.py").exists():
        return
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", CBRAMOD_REPO_URL, str(source_dir)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not clone CBraMod. Check network access or pass --cbramod-source-dir "
            "pointing to a local clone of https://github.com/wjq-learning/CBraMod."
        ) from exc


def _ensure_cbramod_weights(weights_path: Path) -> None:
    if weights_path.exists():
        return
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download CBraMod weights.") from exc
    downloaded = hf_hub_download(
        repo_id=CBRAMOD_HF_REPO,
        filename=CBRAMOD_WEIGHTS,
        local_dir=weights_path.parent,
        local_dir_use_symlinks=False,
    )
    Path(downloaded).replace(weights_path)
