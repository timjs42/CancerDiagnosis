import math

import pytest

from cancer_dx.model import (
    ALL_MARKERS,
    CLASS_MARKERS,
    logpdf,
    predict_class,
    rank_classes,
)


def test_logpdf_peaks_at_mean():
    """The density should be highest exactly at the mean."""
    at_mean = logpdf(100.0, mean=100.0, variance=25.0)
    nearby = logpdf(105.0, mean=100.0, variance=25.0)
    assert at_mean > nearby


def test_logpdf_rejects_nonpositive_variance():
    with pytest.raises(ValueError):
        logpdf(1.0, mean=0.0, variance=0.0)


@pytest.mark.parametrize(
    "markers,expected_class",
    [
        ({"HE4": 180.0, "AFP": 6.0, "CA19-9": 22.0}, "Ovarian_Early"),
        ({"HE4": 65.0, "AFP": 900.0, "CA19-9": 28.0}, "Liver_Stage_II_III"),
        ({"HE4": 70.0, "AFP": 8.0, "CA19-9": 6000.0}, "Pancreatic_Stage_IV"),
        ({"HE4": 60.0, "AFP": 80.0, "CA19-9": 30.0}, "Liver_Stage_I"),
        ({"HE4": 70.0, "AFP": 6000.0, "CA19-9": 24.0}, "Liver_Stage_IV"),
    ],
)
def test_predict_class_matches_expected(markers, expected_class):
    predicted, _ = predict_class(markers)
    assert predicted == expected_class


def test_rank_classes_returns_every_class_sorted_descending():
    markers = {"HE4": 60.0, "AFP": 5.0, "CA19-9": 20.0}
    ranked = rank_classes(markers)

    assert {cls for cls, _ in ranked} == set(CLASS_MARKERS)
    probabilities = [prob for _, prob in ranked]
    assert probabilities == sorted(probabilities, reverse=True)


def test_rank_classes_probabilities_sum_to_one():
    markers = {"HE4": 60.0, "AFP": 5.0, "CA19-9": 20.0}
    ranked = rank_classes(markers)
    total = sum(prob for _, prob in ranked)
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_rank_classes_raises_on_missing_marker():
    incomplete = {"HE4": 60.0, "AFP": 5.0}  # missing CA19-9
    with pytest.raises(ValueError):
        rank_classes(incomplete)


def test_rank_classes_raises_on_negative_marker():
    markers = {"HE4": -1.0, "AFP": 5.0, "CA19-9": 20.0}
    with pytest.raises(ValueError):
        rank_classes(markers)


def test_all_markers_are_used_as_some_signal():
    signal_markers = {profile.signal_marker for profile in CLASS_MARKERS.values()}
    assert signal_markers == set(ALL_MARKERS)