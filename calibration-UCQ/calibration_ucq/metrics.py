from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
    logits = logits.detach().float()
    labels = labels.detach().long()
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(min=1e-3, max=1e3)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().clamp(min=1e-3, max=1e3))


def conformal_threshold(probs: torch.Tensor, labels: torch.Tensor, alpha: float) -> float:
    true_probs = probs[torch.arange(labels.numel()), labels]
    scores = (1.0 - true_probs).detach().cpu().numpy()
    scores.sort()
    rank = int(math.ceil((scores.shape[0] + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), scores.shape[0] - 1)
    return float(scores[rank])


def conformal_coverage(probs: torch.Tensor, labels: torch.Tensor, threshold: float) -> dict[str, float]:
    prediction_sets = (1.0 - probs) <= threshold
    covered = prediction_sets[torch.arange(labels.numel()), labels]
    return {
        "marginal_coverage": float(covered.float().mean().item()),
        "avg_set_size": float(prediction_sets.float().sum(dim=1).mean().item()),
    }


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ece_bins: int = 15,
    fixed_risks: tuple[float, ...] = (0.05, 0.10, 0.25),
) -> dict[str, float]:
    probs = F.softmax(logits, dim=1)
    labels = labels.long()
    predictions = probs.argmax(dim=1)
    correct = predictions.eq(labels)
    one_hot = F.one_hot(labels, num_classes=probs.shape[1]).float()
    confidences = probs.max(dim=1).values

    metrics = {
        "accuracy": float(correct.float().mean().item()),
        "balanced_accuracy": balanced_accuracy(predictions, labels, num_classes=probs.shape[1]),
        "weighted_f1": weighted_f1_score(predictions, labels, num_classes=probs.shape[1]),
        "nll": float(F.cross_entropy(logits, labels).item()),
        "brier": float(((probs - one_hot) ** 2).sum(dim=1).mean().item()),
        "ece": expected_calibration_error(probs, labels, bins=ece_bins),
    }
    metrics.update(selective_risk_metrics(confidences, correct, fixed_risks=fixed_risks))
    return metrics


def balanced_accuracy(predictions: torch.Tensor, labels: torch.Tensor, *, num_classes: int) -> float:
    recalls = []
    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        if class_mask.any():
            recalls.append(predictions[class_mask].eq(labels[class_mask]).float().mean())
    if not recalls:
        return float("nan")
    return float(torch.stack(recalls).mean().item())


def weighted_f1_score(predictions: torch.Tensor, labels: torch.Tensor, *, num_classes: int) -> float:
    weighted_sum = torch.zeros((), dtype=torch.float32, device=labels.device)
    total_support = torch.zeros((), dtype=torch.float32, device=labels.device)
    for class_idx in range(num_classes):
        true_positive = ((predictions == class_idx) & (labels == class_idx)).sum().float()
        false_positive = ((predictions == class_idx) & (labels != class_idx)).sum().float()
        false_negative = ((predictions != class_idx) & (labels == class_idx)).sum().float()
        support = (labels == class_idx).sum().float()
        if support == 0:
            continue
        denominator = 2.0 * true_positive + false_positive + false_negative
        f1 = torch.zeros((), dtype=torch.float32, device=labels.device)
        if denominator > 0:
            f1 = 2.0 * true_positive / denominator
        weighted_sum = weighted_sum + support * f1
        total_support = total_support + support
    if total_support == 0:
        return float("nan")
    return float((weighted_sum / total_support).item())


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()
    ece = torch.zeros((), dtype=torch.float32, device=probs.device)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probs.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if mask.any():
            weight = mask.float().mean()
            ece = ece + weight * torch.abs(confidences[mask].mean() - accuracies[mask].mean())
    return float(ece.item())


def selective_risk_metrics(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    *,
    fixed_risks: tuple[float, ...],
) -> dict[str, float]:
    order = torch.argsort(confidences, descending=True)
    errors = (~correct[order]).float()
    k = torch.arange(1, errors.numel() + 1, device=errors.device, dtype=torch.float32)
    risks = torch.cumsum(errors, dim=0) / k
    aurc = float(risks.mean().item())

    optimal_errors = torch.sort((~correct).float()).values
    optimal_risks = torch.cumsum(optimal_errors, dim=0) / k
    eaurc = aurc - float(optimal_risks.mean().item())

    metrics = {"aurc": aurc, "eaurc": eaurc}
    coverages = k / float(errors.numel())
    for risk in fixed_risks:
        valid = coverages[risks <= risk]
        metrics[f"coverage_at_risk_{risk:g}"] = float(valid.max().item()) if valid.numel() else 0.0
    return metrics


def ensemble_uncertainty(probs: torch.Tensor) -> dict[str, float]:
    """Summarize uncertainty from probabilities with shape (members, examples, classes)."""
    mean_probs = probs.mean(dim=0)
    total = entropy(mean_probs)
    aleatoric = entropy(probs).mean(dim=0)
    epistemic = total - aleatoric
    return {
        "total_uncertainty_mean": float(total.mean().item()),
        "total_uncertainty_std": float(total.std(unbiased=False).item()),
        "aleatoric_uncertainty_mean": float(aleatoric.mean().item()),
        "aleatoric_uncertainty_std": float(aleatoric.std(unbiased=False).item()),
        "epistemic_uncertainty_mean": float(epistemic.mean().item()),
        "epistemic_uncertainty_std": float(epistemic.std(unbiased=False).item()),
    }


def entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)


def tensor_dict_to_float(metrics: dict[str, float | int | np.number]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}
