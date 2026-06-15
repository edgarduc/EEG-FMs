# Calibration and epistemic VS aleatoric uncertainty quatification for EEG FMs

This experiment aims at testing SOTA EEG foundation models on calibration and uncertainty quantification, more specifically quantifying epistemic and aleatoric uncertainty across models and OOD setups.
This experiment is meant to be computationally lightweight, and should therefore not be considered exhaustive.

### Models

- REVE
- CBraMod
- LaBraM
- CS-Brain
- EEGPT

All EEG FMs will be loaded from a publicly available pretrained checkpoint and kept frozen. We will use linear probing.

### Datasets

- Mumtaz
- TUEB
- TUAV
- Physionet-MI
- BCIC-IV-2a
- HMC
- ISRUC
- FACED (Chen et. al. 2023)
- Mental arithmetic MAT (Zyma et. al. 2019)
- BCI2020-IV-3 (Jeong et. al. 2022)

We will only use 1 or 2 datasets.
Prioritized datasets are *MAT* and *BCIC-IV-2a* because they are not too heavy to download.

### Setup

We use 3 different seeds for every model run (for the data splitting, the linear probe training and setup configuration).
We train the linear probe on a training+val split (60%), and perform calibration (softmax temperature, conformal calibration) on the calibration split (20%). Calibration split should ideally contain sperate held-out subjects compared to training split.
The test set (20%) is used to compute metrics. It should always be corresponding to a specific "distribution shift" (or a subject-shift-only baseline).

### OOD types

- subject-disjoint baseline: same distribution (dataset), subject-disjoint protocol (the test subjects never appear in the training or calibration split)
- montage shift : randomly drop some channels (zero-out) between training split and test split (zero-ing probability is 20%). For each seed, we instanciate one mask so that for the whole test split, the exact same channels are being zeroed-out across trials.
- corruption shift: add Gaussian noise at different SNR levels (10dB, 5dB, 0dB, -5dB, -15dB).

Other shift such as population shift, label shift (changing class priors) or task shift could also be considered, but they are deemed less relevant.
Dataset shift and montage shift are somehow close (acquisition protocol shift), but artificial montage shift is more controlled and specific.

### Metrics

**Question: are the predictions calibrated ?**
OOD types : all
Metrics: NLL, Brier score and ECE on test split.

**Question: can we reliably predict errors ?**
OOD types: all
Metrics: AURC, E-AURC, coverage at fixed risk (5%, 10%, 25%)

**Question: are prediction sets valid ?**
OOD types: suject-disjoint baseline protocol only
If we use conformal prediction, do the prediction sets contain the true label with the promised frequency ?
Metrics: marginal coverage

**Question: how do aleatoric and epistemic uncertainties vary under different OOD setups ?**
OOD types: all
Method : use ensemble of (N = 10) classifiers
Metrics: total uncertainty (predictive entropy), aleatoric proxy (average entropy across ensemble members), epistemic proxy (mutual information)
*Only one seed is required for the aleatoric/epistemic uncertainty metrics*

