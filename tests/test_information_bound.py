import numpy as np
import pytest

from netattdetector.information_bound import connection_information_bound, load_config


def examples():
    groups, labels, vectors = [], [], []
    # Six free positives, then 8 attacks/3 benign and 5 attacks/2 benign, plus one pure benign.
    for cluster, (benign, attacks) in enumerate([(0, 6), (3, 8), (2, 5), (1, 0)]):
        for label, total in ((0, benign), (1, attacks)):
            for number in range(total):
                groups.append(f"{cluster}-{label}-{number}")
                labels.append(label)
                vectors.append([cluster])
    return groups, labels, np.asarray(vectors, dtype="<f8")


def test_exact_knapsack_costs_and_integer_operating_budget():
    result = connection_information_bound(*examples(), maximum_fpr="1/3")
    assert result["benign_connections"] == 6 and result["attack_connections"] == 19
    assert result["benign_flag_budget"] == 2
    assert result["optimistic_max_true_positives_at_budget"] == 11
    assert result["required_true_positives"] == 16
    assert result["minimum_benign_flags_for_target_in_relaxation"] == 5
    assert not result["target_not_ruled_out_by_relaxation"]
    assert result["mixed_label_clusters"] == 2
    assert not result["learnable_operating_point_proven"]
    alternate = connection_information_bound(*examples(), maximum_fpr="0.5")
    assert (
        alternate["optimistic_max_true_positives_at_budget"] == 14
    )  # Choose cost3/value8, not cost2/value5.


def test_max_scoring_ignores_vector_multiplicity_and_order():
    result = connection_information_bound(
        ["a", "a", "a", "b", "b"],
        [0, 0, 0, 1, 1],
        np.array([[1], [1], [2], [2], [1]], dtype="<f8"),
        maximum_fpr="0",
    )
    assert result["exact_vector_set_clusters"] == 1
    assert result["optimistic_max_true_positives_at_budget"] == 0
    assert result["minimum_benign_flags_for_target_in_relaxation"] == 1


def test_shared_vectors_between_different_sets_make_the_bound_optimistic():
    # Scoring vector2 high would flag benign a too, but independent set groups ignore that constraint.
    result = connection_information_bound(
        ["a", "a", "b"], [0, 0, 1], np.array([[1], [2], [2]], dtype="<f8"), maximum_fpr="0"
    )
    assert result["optimistic_max_true_positives_at_budget"] == 1
    assert result["target_not_ruled_out_by_relaxation"]
    assert not result["learnable_operating_point_proven"]


def test_float32_and_float64_bounds_apply_to_different_representations():
    values = np.array([[1.0], [1.0 + 1e-9]], dtype="<f8")
    high = connection_information_bound(["a", "b"], [0, 1], values, maximum_fpr="0")
    low = connection_information_bound(["a", "b"], [0, 1], values.astype("<f4"), maximum_fpr="0")
    assert high["optimistic_max_true_positives_at_budget"] == 1
    assert low["optimistic_max_true_positives_at_budget"] == 0


def test_budget_failure_never_returns_a_partial_approximation():
    with pytest.raises(ValueError, match="update budget"):
        connection_information_bound(*examples(), max_dp_updates=1)


@pytest.mark.parametrize(
    "groups,labels,values",
    [
        (["a", "a"], [0, 1], [[1], [2]]),
        (["a", "b"], [0, 2], [[1], [2]]),
        (["a"], [0], [[1]]),
        (["a", "b"], [0, 1], [[float("nan")], [2]]),
    ],
)
def test_invalid_or_inconsistent_groups_fail(groups, labels, values):
    with pytest.raises(ValueError):
        connection_information_bound(groups, labels, np.asarray(values, dtype="<f8"))


def test_rational_budget_does_not_round_up_false_positive_allowance():
    groups = [f"b{i}" for i in range(199)] + ["attack"]
    result = connection_information_bound(
        groups, [0] * 199 + [1], np.array([[0]] * 199 + [[1]], dtype="<f8")
    )
    assert result["benign_flag_budget"] == 0
    assert result["required_true_positives"] == 1
    assert load_config("configs/data/information-bound.json")["training_enabled"] is False
