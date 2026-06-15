from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn


class FrozenBackbone(nn.Module, ABC):
    @abstractmethod
    def encode_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Return embeddings with shape [batch, channels, dim]."""


class MomentBackbone(FrozenBackbone):
    def __init__(self, checkpoint: str):
        super().__init__()
        try:
            from momentfm import MOMENTPipeline
        except ImportError as exc:
            raise ImportError(
                "MOMENT requires the optional dependency 'momentfm'. Install with "
                "`pip install -e .[moment]` from TS-FMs/."
            ) from exc

        self.model = MOMENTPipeline.from_pretrained(
            checkpoint,
            model_kwargs={"task_name": "embedding"},
        )
        self.model.init()
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def encode_channels(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time = x.shape
        flattened = x.reshape(batch * channels, 1, time)
        input_mask = torch.ones(flattened.shape[0], flattened.shape[-1], device=x.device)
        try:
            output = self.model(x_enc=flattened, input_mask=input_mask)
        except TypeError:
            output = self.model(flattened)
        embedding = _extract_embedding(output)
        embedding = _pool_to_vector(embedding)
        return embedding.reshape(batch, channels, -1)


class TSPulseBackbone(FrozenBackbone):
    """Adapter for IBM Granite TSPulse or a user supplied TSPulse module."""

    def __init__(self, checkpoint: str | None, module_factory: str | None, revision: str | None):
        super().__init__()
        if checkpoint is None:
            raise ValueError(
                "TSPulse checkpoint is required. Pass --tspulse-checkpoint MODEL_ID_OR_PATH, "
                "and optionally --tspulse-module package.module:factory if your checkpoint "
                "needs custom loading."
            )

        self.model = self._load_model(checkpoint, module_factory, revision)
        self.encoder = self.model.backbone if hasattr(self.model, "backbone") else self.model
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def encode_channels(self, x: torch.Tensor) -> torch.Tensor:
        past_values = x.transpose(1, 2).contiguous()
        try:
            output = self.encoder(
                past_values=past_values,
                output_hidden_states=True,
                return_dict=True,
                enable_masking=False,
            )
        except TypeError:
            try:
                output = self.encoder(past_values)
            except TypeError:
                output = self.encoder(x)
        embedding = _extract_embedding(output)

        if embedding.ndim == 4:
            embedding = embedding.mean(dim=2)
        elif embedding.ndim == 3:
            if embedding.shape[1] != x.shape[1]:
                embedding = embedding.mean(dim=1, keepdim=True).expand(-1, x.shape[1], -1)
        elif embedding.ndim == 2:
            embedding = embedding.unsqueeze(1).expand(-1, x.shape[1], -1)
        else:
            raise ValueError(f"Unsupported TSPulse embedding shape: {tuple(embedding.shape)}")

        return embedding

    @staticmethod
    def _load_model(checkpoint: str, module_factory: str | None, revision: str | None) -> nn.Module:
        if module_factory is not None:
            module_name, sep, factory_name = module_factory.partition(":")
            if not sep:
                raise ValueError("--tspulse-module must have the form package.module:factory")
            module = importlib.import_module(module_name)
            factory = getattr(module, factory_name)
            model = factory(checkpoint)
            if not isinstance(model, nn.Module):
                raise TypeError("TSPulse factory must return a torch.nn.Module.")
            return model

        if checkpoint.startswith("ibm-granite/") or "/" in checkpoint:
            try:
                from tsfm_public.models.tspulse import TSPulseForClassification
            except ImportError as exc:
                raise ImportError(
                    "IBM Granite TSPulse requires 'granite-tsfm'. Install with "
                    "`pip install -e .[all]` or `pip install granite-tsfm` from TS-FMs/."
                ) from exc
            return TSPulseForClassification.from_pretrained(
                checkpoint,
                revision=revision,
                ignore_mismatched_sizes=True,
            )

        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            try:
                scripted = torch.jit.load(str(checkpoint_path), map_location="cpu")
                if isinstance(scripted, nn.Module):
                    return scripted
            except RuntimeError:
                pass
            loaded = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(loaded, nn.Module):
                return loaded
            raise TypeError(
                "Torch checkpoint did not contain an nn.Module. Use --tspulse-module "
                "package.module:factory for custom state-dict loading."
            )

        raise FileNotFoundError(
            f"TSPulse checkpoint '{checkpoint}' is not a local file. Provide a local checkpoint "
            "or a Hugging Face model ID such as ibm-granite/granite-timeseries-tspulse-r1."
        )


class RandomFrozenBackbone(FrozenBackbone):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim

    @torch.no_grad()
    def encode_channels(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        features = torch.cat([mean, std], dim=-1)
        repeats = (self.embedding_dim + features.shape[-1] - 1) // features.shape[-1]
        return features.repeat_interleave(repeats, dim=-1)[..., : self.embedding_dim]


def build_backbone(
    kind: str,
    checkpoint: str | None,
    module_factory: str | None = None,
    revision: str | None = None,
) -> FrozenBackbone:
    if kind == "moment":
        if checkpoint is None:
            raise ValueError("MOMENT checkpoint must not be None.")
        return MomentBackbone(checkpoint)
    if kind == "tspulse":
        return TSPulseBackbone(checkpoint, module_factory, revision)
    if kind == "random":
        return RandomFrozenBackbone()
    raise ValueError(f"Unknown backbone kind '{kind}'.")


def _extract_embedding(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("embeddings", "embedding", "last_hidden_state", "hidden_states", "features"):
            if key in output:
                value = output[key]
                if isinstance(value, (tuple, list)):
                    value = value[-1]
                return value
    for attr in ("embeddings", "embedding", "last_hidden_state", "hidden_states", "features"):
        if hasattr(output, attr):
            value = getattr(output, attr)
            if isinstance(value, (tuple, list)):
                value = value[-1]
            return value
    raise ValueError(f"Could not find embeddings in backbone output of type {type(output)!r}.")


def _pool_to_vector(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.ndim == 2:
        return embedding
    if embedding.ndim == 3:
        return embedding.mean(dim=1)
    if embedding.ndim == 4:
        return embedding.mean(dim=(1, 2))
    raise ValueError(f"Unsupported embedding shape: {tuple(embedding.shape)}")
