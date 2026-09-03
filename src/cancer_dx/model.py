"""
Naive Bayes classification of cancer type/stage from tumor marker levels.

Three tumor markers are used: HE4, AFP, and CA19-9. Each candidate class
(e.g. "Ovarian_Early") has one *signal* marker with a Gaussian distribution
fit to that class. Every marker that is *not* the signal marker for a class
is scored under a "healthy" Gaussian instead, which penalizes classes whose
non-signal markers are far outside a normal range. For example, Pancreatic
Stage IV (very high CA19-9) won't be confused with Ovarian Early just
because HE4 happens to be normal, since the very high CA19-9 will look
extremely unlikely under the healthy CA19-9 distribution.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Tuple

PatientMarkers = Dict[str, float]


class MarkerDistribution(NamedTuple):
    """A Gaussian distribution (mean, variance) for one tumor marker."""

    mean: float
    variance: float


class ClassProfile(NamedTuple):
    """A candidate class: which marker signals it, and that marker's distribution."""

    signal_marker: str
    distribution: MarkerDistribution


# ---- Cancer class definitions ----
# Each class is defined by its signal marker and that marker's
# (mean, variance) among patients with that class/stage.
CLASS_MARKERS: Dict[str, ClassProfile] = {
    # Ovarian Cancer
    "Ovarian_Early": ClassProfile("HE4", MarkerDistribution(151.0, 6_348.0)),
    "Ovarian_Late": ClassProfile("HE4", MarkerDistribution(570.0, 84_840.0)),
    # Liver Cancer (HCC)
    "Liver_Overall": ClassProfile("AFP", MarkerDistribution(450.0, 2_250_000.0)),
    "Liver_Stage_I": ClassProfile("AFP", MarkerDistribution(100.0, 7_500.0)),
    "Liver_Stage_II_III": ClassProfile("AFP", MarkerDistribution(600.0, 450_000.0)),
    "Liver_Stage_IV": ClassProfile("AFP", MarkerDistribution(6_000.0, 15_000_000.0)),
    # Pancreatic Cancer
    "Pancreatic_Overall": ClassProfile("CA19-9", MarkerDistribution(1_750.0, 3_500_000.0)),
    "Pancreatic_Stage_I": ClassProfile("CA19-9", MarkerDistribution(140.0, 17_500.0)),
    "Pancreatic_Stage_II_III": ClassProfile("CA19-9", MarkerDistribution(950.0, 750_000.0)),
    "Pancreatic_Stage_IV": ClassProfile("CA19-9", MarkerDistribution(12_500.0, 35_000_000.0)),
}

ALL_MARKERS: Tuple[str, ...] = ("HE4", "AFP", "CA19-9")

# ---- "Healthy" reference distributions ----
# Used to score a marker when it is NOT the signal marker for a class.
HEALTHY: Dict[str, MarkerDistribution] = {
    "HE4": MarkerDistribution(60.0, 15.0**2),
    "AFP": MarkerDistribution(5.0, 3.0**2),
    "CA19-9": MarkerDistribution(20.0, 10.0**2),
}

# Uniform prior over all classes: log P(Class) = -log(number of classes)
_LOG_PRIOR: float = -math.log(len(CLASS_MARKERS))


def logpdf(x: float, mean: float, variance: float) -> float:
    """
    Log of the Gaussian probability density function.

    log P(x | mean, variance) = -0.5 * [log(2*pi*variance) + (x - mean)^2 / variance]

    Raises:
        ValueError: if variance is not positive.
    """
    if variance <= 0:
        raise ValueError(f"variance must be positive, got {variance!r}")
    return -0.5 * (math.log(2.0 * math.pi * variance) + ((x - mean) ** 2) / variance)


def _validate_patient(patient: PatientMarkers) -> None:
    """Ensure every marker the model needs is present and numeric."""
    missing = [m for m in ALL_MARKERS if m not in patient or patient[m] is None]
    if missing:
        raise ValueError(
            f"Patient is missing required marker value(s): {', '.join(missing)}"
        )
    for marker in ALL_MARKERS:
        value = patient[marker]
        if not isinstance(value, (int, float)):
            raise TypeError(f"Marker {marker!r} must be numeric, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"Marker {marker!r} cannot be negative, got {value!r}")


def _logsumexp(values: List[float]) -> float:
    """Numerically stable log(sum(exp(v) for v in values))."""
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def rank_classes(patient: PatientMarkers) -> List[Tuple[str, float]]:
    """
    Rank every candidate class for a patient by posterior probability.

    For each class:
        log P(Class | x) is proportional to
            log P(Class) + log P(x_signal | Class) + sum(log P(x_other | Healthy))
    Scores are then normalized into probabilities with log-sum-exp.

    Args:
        patient: mapping of marker name -> observed value. Must contain
            every marker in ALL_MARKERS (HE4, AFP, CA19-9).

    Returns:
        List of (class_name, probability) tuples sorted by probability,
        highest first.

    Raises:
        ValueError: if a required marker is missing or invalid.
    """
    _validate_patient(patient)

    scores: Dict[str, float] = {}
    for cls, (signal_marker, (mu, var)) in CLASS_MARKERS.items():
        logp = _LOG_PRIOR + logpdf(float(patient[signal_marker]), mu, var)

        for marker in ALL_MARKERS:
            if marker == signal_marker:
                continue
            h_mean, h_var = HEALTHY[marker]
            logp += logpdf(float(patient[marker]), h_mean, h_var)

        scores[cls] = logp

    lse = _logsumexp(list(scores.values()))
    ranked = sorted(
        ((cls, math.exp(s - lse)) for cls, s in scores.items()),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked


def predict_class(patient: PatientMarkers) -> Tuple[str, List[Tuple[str, float]]]:
    """
    Predict the single most likely class for a patient.

    Returns:
        (best_class_name, full_ranked_list)
    """
    ranked = rank_classes(patient)
    return ranked[0][0], ranked