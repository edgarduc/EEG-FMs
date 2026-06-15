from __future__ import annotations

import pickle
import subprocess
import sys
import urllib.request
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
    LABRAM_REPO_URL,
    LABRAM_WEIGHTS,
    LABRAM_WEIGHTS_URL,
    REVE_MODEL_ID,
    REVE_POSITION_MODEL_ID,
)


LABRAM_STANDARD_1020 = [
    "FP1",
    "FPZ",
    "FP2",
    "AF9",
    "AF7",
    "AF5",
    "AF3",
    "AF1",
    "AFZ",
    "AF2",
    "AF4",
    "AF6",
    "AF8",
    "AF10",
    "F9",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "F10",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T9",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "T10",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P9",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO9",
    "PO7",
    "PO5",
    "PO3",
    "PO1",
    "POZ",
    "PO2",
    "PO4",
    "PO6",
    "PO8",
    "PO10",
    "O1",
    "OZ",
    "O2",
    "O9",
    "CB1",
    "CB2",
    "IZ",
    "O10",
    "T3",
    "T5",
    "T4",
    "T6",
    "M1",
    "M2",
    "A1",
    "A2",
    "CFC1",
    "CFC2",
    "CFC3",
    "CFC4",
    "CFC5",
    "CFC6",
    "CFC7",
    "CFC8",
    "CCP1",
    "CCP2",
    "CCP3",
    "CCP4",
    "CCP5",
    "CCP6",
    "CCP7",
    "CCP8",
    "T1",
    "T2",
    "FTT9H",
    "TTP7H",
    "TPP9H",
    "FTT10H",
    "TPP8H",
    "TPP10H",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]

LABRAM_CHANNEL_INDEX = {channel: idx + 1 for idx, channel in enumerate(LABRAM_STANDARD_1020)}


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
        hf_token: str | None = None,
        device: torch.device,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("transformers is required for the REVE adapter. Install requirements.txt.") from exc

        self.info = ExtractorInfo(name="reve", target_sfreq=200.0)
        self.pooling = pooling
        try:
            self.model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=hf_token,
            ).to(device)
            self.position_model = AutoModel.from_pretrained(
                position_model_id,
                trust_remote_code=True,
                token=hf_token,
            ).to(device)
        except OSError as exc:
            raise RuntimeError(
                "Could not load REVE from Hugging Face. The REVE repositories are gated, "
                "so you need to request access on Hugging Face and authenticate the run. "
                "Use `huggingface-cli login`, set `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN`, "
                "or pass `--hf-token <token>`."
            ) from exc
        self._position_cache: dict[tuple[str, ...], tuple[torch.Tensor, list[int], list[str], list[str]]] = {}
        self.model.eval()
        self.position_model.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def extract(self, eeg: torch.Tensor, channel_names: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        eeg = eeg.to(device)
        positions, kept_indices, kept_names, dropped_names = self._positions_for_channels(
            channel_names,
            device=device,
            dtype=eeg.dtype,
        )
        if dropped_names:
            eeg = eeg[:, kept_indices, :]
        if positions.shape[0] % eeg.shape[1] != 0:
            raise RuntimeError(
                "REVE positional embeddings are incompatible with the current EEG window. "
                f"Got {positions.shape[0]} positional tokens for {eeg.shape[1]} channels "
                f"({kept_names}). Try the default --window-seconds 10.0."
            )
        positions = positions.to(device=device, dtype=eeg.dtype).unsqueeze(0).expand(eeg.shape[0], -1, -1)
        output = self.model(eeg, positions)
        return _pool_output(output, self.pooling)

    @torch.no_grad()
    def _positions_for_channels(
        self,
        channel_names: list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, list[int], list[str], list[str]]:
        key = tuple(channel_names)
        if key not in self._position_cache:
            kept_indices: list[int] = []
            kept_names: list[str] = []
            dropped_names: list[str] = []
            for idx, channel_name in enumerate(channel_names):
                try:
                    channel_positions = _as_tensor(self.position_model([channel_name]))
                except Exception:
                    channel_positions = torch.empty(0)
                if channel_positions.numel() > 0 and channel_positions.shape[0] > 0:
                    kept_indices.append(idx)
                    kept_names.append(channel_name)
                else:
                    dropped_names.append(channel_name)

            if not kept_names:
                raise RuntimeError(
                    "REVE position bank did not recognize any EEG channels. "
                    f"Original channel names: {channel_names}"
                )

            positions = _as_tensor(self.position_model(kept_names)).detach().cpu()
            self._position_cache[key] = (positions, kept_indices, kept_names, dropped_names)
            if dropped_names:
                print(
                    "Dropping channels not recognized by REVE position bank: "
                    + ", ".join(dropped_names),
                    flush=True,
                )

        positions, kept_indices, kept_names, dropped_names = self._position_cache[key]
        return positions.to(device=device, dtype=dtype), kept_indices, kept_names, dropped_names


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
        hf_token: str | None = None,
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
        _ensure_cbramod_weights(weights_path, hf_token=hf_token)

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


class LaBraMFeatureExtractor(FrozenFeatureExtractor):
    def __init__(
        self,
        *,
        source_dir: Path | None = None,
        weights_path: Path | None = None,
        pooling: str = "flatten",
        patch_size: int = 200,
        max_patches: int = 16,
        device: torch.device,
    ):
        super().__init__()
        self.info = ExtractorInfo(name="labram", target_sfreq=200.0)
        self.pooling = pooling
        self.patch_size = patch_size
        self.max_patches = max_patches
        self.chunk_points = patch_size * max_patches
        self._channel_cache: dict[tuple[str, ...], tuple[list[int], list[int], list[str], list[str]]] = {}

        source_dir = source_dir or DATA_DIR / "model_sources" / "LaBraM"
        weights_path = weights_path or DATA_DIR / "model_weights" / "labram" / LABRAM_WEIGHTS
        _ensure_labram_source(source_dir)
        _ensure_labram_weights(weights_path)

        sys.path.insert(0, str(source_dir))
        try:
            import modeling_finetune
        except ImportError as exc:
            raise RuntimeError(
                f"Could not import LaBraM from {source_dir}. Install the LaBraM dependencies, including timm."
            ) from exc

        self.model = modeling_finetune.labram_base_patch200_200(
            num_classes=0,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            qkv_bias=False,
            init_values=0.1,
        )
        state = _checkpoint_state_dict(_load_checkpoint(weights_path, map_location="cpu"))
        loaded = _load_matching_state_dict(self.model, state)
        if loaded == 0:
            raise RuntimeError(f"No LaBraM checkpoint weights from {weights_path} matched the model architecture.")
        self.model.to(device)
        self.model.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def extract(self, eeg: torch.Tensor, channel_names: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        input_chans, kept_indices, kept_names, dropped_names = self._input_channels_for_channels(channel_names)
        if dropped_names:
            eeg = eeg[:, kept_indices, :]
        eeg = eeg.to(device)
        chunks = _patch_chunks_or_pad(eeg, patch_size=self.patch_size, max_points=self.chunk_points)
        embeddings = []
        for chunk in chunks:
            patches = chunk.reshape(chunk.shape[0], chunk.shape[1], -1, self.patch_size)
            output = self.model.forward_features(
                patches,
                input_chans=input_chans,
                return_patch_tokens=self.pooling == "flatten",
            )
            embeddings.append(_pool_output(output, self.pooling))
        return torch.cat(embeddings, dim=1)

    def _input_channels_for_channels(
        self,
        channel_names: list[str],
    ) -> tuple[list[int], list[int], list[str], list[str]]:
        key = tuple(channel_names)
        if key not in self._channel_cache:
            input_chans = [0]
            kept_indices: list[int] = []
            kept_names: list[str] = []
            dropped_names: list[str] = []
            for idx, channel_name in enumerate(channel_names):
                normalized = _normalize_labram_channel_name(channel_name)
                channel_index = LABRAM_CHANNEL_INDEX.get(normalized)
                if channel_index is None:
                    dropped_names.append(channel_name)
                    continue
                input_chans.append(channel_index)
                kept_indices.append(idx)
                kept_names.append(channel_name)
            if not kept_names:
                raise RuntimeError(
                    "LaBraM did not recognize any EEG channels. "
                    f"Original channel names: {channel_names}"
                )
            self._channel_cache[key] = (input_chans, kept_indices, kept_names, dropped_names)
            if dropped_names:
                print(
                    "Dropping channels not recognized by LaBraM channel bank: "
                    + ", ".join(dropped_names),
                    flush=True,
                )
        return self._channel_cache[key]


def build_feature_extractor(
    model_name: str,
    *,
    device: torch.device,
    pooling: str = "flatten",
    hf_token: str | None = None,
    cbramod_source_dir: Path | None = None,
    cbramod_weights_path: Path | None = None,
    labram_source_dir: Path | None = None,
    labram_weights_path: Path | None = None,
) -> FrozenFeatureExtractor:
    normalized = model_name.lower()
    if normalized == "reve":
        return REVEFeatureExtractor(pooling=pooling, hf_token=hf_token, device=device)
    if normalized == "cbramod":
        return CBraModFeatureExtractor(
            source_dir=cbramod_source_dir,
            weights_path=cbramod_weights_path,
            pooling=pooling,
            hf_token=hf_token,
            device=device,
        )
    if normalized == "labram":
        return LaBraMFeatureExtractor(
            source_dir=labram_source_dir,
            weights_path=labram_weights_path,
            pooling=pooling,
            device=device,
        )
    raise ValueError(f"Unsupported model {model_name!r}. Choose one of: reve, cbramod, labram.")


def _pool_output(output: Any, pooling: str) -> torch.Tensor:
    tensor = _output_to_tensor(output)
    if tensor.ndim == 2:
        return tensor
    if pooling == "mean":
        return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1]).mean(dim=1)
    if pooling == "flatten":
        return tensor.reshape(tensor.shape[0], -1)
    raise ValueError(f"Unsupported embedding pooling mode: {pooling}")


def _as_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


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


def _patch_chunks_or_pad(eeg: torch.Tensor, *, patch_size: int, max_points: int) -> list[torch.Tensor]:
    chunks = []
    for start in range(0, eeg.shape[-1], max_points):
        chunk = eeg[:, :, start : start + max_points]
        remainder = chunk.shape[-1] % patch_size
        if remainder:
            chunk = F.pad(chunk, (0, patch_size - remainder))
        chunks.append(chunk)
    return chunks


def _normalize_labram_channel_name(channel_name: str) -> str:
    return channel_name.replace(".", "").strip().upper()


def _load_checkpoint(path: Path, *, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        # LaBraM's official checkpoint contains NumPy scalar metadata, which PyTorch
        # 2.6 rejects under weights_only=True before we can extract the model state.
        return torch.load(path, map_location=map_location, weights_only=False)


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return _strip_state_dict_prefixes(value)
        if all(isinstance(key, str) for key in checkpoint):
            tensor_items = {key: value for key, value in checkpoint.items() if torch.is_tensor(value)}
            if tensor_items:
                return _strip_state_dict_prefixes(tensor_items)
    raise TypeError(f"Cannot extract a PyTorch state dict from checkpoint type {type(checkpoint).__name__}.")


def _strip_state_dict_prefixes(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state.items():
        normalized = key
        for prefix in ("module.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        stripped[normalized] = value
    return stripped


def _load_matching_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> int:
    model_state = model.state_dict()
    matching = {
        key: value
        for key, value in state.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    model.load_state_dict(matching, strict=False)
    return len(matching)


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


def _ensure_cbramod_weights(weights_path: Path, hf_token: str | None = None) -> None:
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
        token=hf_token,
    )
    Path(downloaded).replace(weights_path)


def _ensure_labram_source(source_dir: Path) -> None:
    if (source_dir / "modeling_finetune.py").exists():
        return
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", LABRAM_REPO_URL, str(source_dir)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not clone LaBraM. Check network access or pass --labram-source-dir "
            "pointing to a local clone of https://github.com/935963004/LaBraM."
        ) from exc


def _ensure_labram_weights(weights_path: Path) -> None:
    if weights_path.exists():
        return
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = weights_path.with_suffix(weights_path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(LABRAM_WEIGHTS_URL, tmp)
        tmp.replace(weights_path)
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            "Could not download LaBraM weights. Check network access or pass --labram-weights-path "
            "pointing to labram-base.pth from https://github.com/935963004/LaBraM."
        ) from exc
