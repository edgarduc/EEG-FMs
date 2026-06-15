# Time Series Foundation Models Expressivity On EEG Signals

This package evaluates frozen time-series foundation model representations on EEG
downstream tasks. For each selected study, it trains only:

- an EEG attention pooling head that receives channel embeddings, channel IDs, and
  electrode position encodings
- a linear classifier/probe on top of the pooled trial representation

The TS-FM backbone parameters remain frozen. The CLI runs the protocol for MOMENT
and then TSPulse on the same subject-disjoint train/validation/test split.

## Supported Studies

- `bcic-iv-2a`: loaded through MOABB as `BNCI2014_001`
- `mat`: PhysioNet EEGMAT / mental arithmetic dataset

The split protocol is subject-disjoint: test subjects never appear in the
training or validation splits.

## Installation

From this directory:

```bash
pip install -e ".[all]"
```

`momentfm` is required for the MOMENT run. `granite-tsfm` is required for the
IBM Granite TSPulse run. The first real run will also download the selected EEG
study and model checkpoints.

## Usage

Run both backbones on BCIC-IV-2a:

```bash
ts-fms \
  --config configs/bcic_iv_2a.yaml
```

Run both backbones on MAT:

```bash
ts-fms \
  --config configs/mat.yaml
```

Run direct linear probing without attention pooling:

```bash
ts-fms \
  --config configs/mat.yaml \
  --no-attention-pooling
```

In this mode, per-channel backbone embeddings are concatenated into one trial
vector before the classifier:

```text
[channels, embedding_dim] -> [channels * embedding_dim] -> Linear
```

By default, TSPulse uses:

```text
checkpoint: ibm-granite/granite-timeseries-tspulse-r1
revision: tspulse-block-dualhead-512-p16-r1
```

That revision is the model-card variant recommended by IBM Granite for
classification-style downstream use. To override it:

```bash
ts-fms \
  --config configs/mat.yaml \
  --tspulse-checkpoint ibm-granite/granite-timeseries-tspulse-r1 \
  --tspulse-revision tspulse-hybrid-dualhead-512-p8-r1
```

For a custom TSPulse loader, provide a factory that returns a `torch.nn.Module`:

```bash
ts-fms \
  --config configs/mat.yaml \
  --tspulse-checkpoint /path/or/model-id \
  --tspulse-module my_package.tspulse_loader:load_model
```

The factory receives the checkpoint string and must return a frozen-compatible
PyTorch module. The adapter accepts outputs as tensors, dictionaries, or objects
with fields such as `embeddings`, `embedding`, `last_hidden_state`, or `features`.

## Outputs

Each run writes JSON results to:

```text
runs/<study>/<backbone>_<pooling>_seed<seed>.json
```

The saved payload includes the config, subject split, validation history, and
test metrics:

- accuracy
- balanced accuracy
- macro F1

## Implementation Notes

MOMENT is loaded with `MOMENTPipeline.from_pretrained(..., task_name="embedding")`.
Because MOMENT is treated as channel-independent here, each EEG channel is encoded
separately and then passed to the EEG attention pooling head.

TSPulse is loaded from IBM Granite's Hugging Face checkpoint with
`TSPulseForClassification.from_pretrained(...)`. In this experiment it is used
like MOMENT: each EEG channel is encoded independently as a univariate time
series, then channel interactions are learned only by the downstream attention
pooling head or concatenated linear probe. The adapter also supports local
Torch/TorchScript checkpoints or a user-provided loader factory.
