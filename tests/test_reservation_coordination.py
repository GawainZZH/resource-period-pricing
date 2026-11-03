from importlib import import_module
from pathlib import Path
import numpy as np
import pytest


def coordination_module():
    path = Path(__file__).resolve().parents[1] / "resource_pricing/reservation_allocation.py"
    assert path.is_file(), "The independent coordination audit is not implemented."
    return import_module("resource_pricing.reservation_allocation")


def single_class_data():
    return (np.array([[1.0, 1.0]]), np.array([1.0]), np.array([[1.0, 0.0]]), np.array([0.25]))


def test_excess_reservation_is_projected_onto_shared_capacity():
    coordinator = coordination_module()
    conservation, demand, delta, capacity = single_class_data()
    result = coordinator.clear_reservations(
        np.array([1.0, 0.0]), conservation, demand, delta, capacity
    )
    np.testing.assert_allclose(result.flow, [0.25, 0.75], atol=1e-10)
    assert result.capacity_violation <= 1e-10
    assert result.conservation_violation <= 1e-10
    assert result.first_order_gap <= 1e-09


def test_feasible_request_does_not_change_during_clearing():
    coordinator = coordination_module()
    flow = np.array([0.1, 0.9])
    result = coordinator.clear_reservations(flow, *single_class_data())
    np.testing.assert_allclose(result.flow, flow, atol=1e-09)


@pytest.mark.parametrize("step", [0.01, 0.1, 1.0, 10.0])
def test_known_capacity_vi_solution_is_a_clearing_fixed_point(step):
    coordinator = coordination_module()
    flow = np.array([0.25, 0.75])
    costs = np.array([0.0, 2.0])
    result = coordinator.clear_reservations(flow - step * costs, *single_class_data())
    np.testing.assert_allclose(result.flow, flow, atol=1e-09)


def test_non_equilibrium_allocation_is_not_a_fixed_point():
    coordinator = coordination_module()
    flow = np.array([0.0, 1.0])
    result = coordinator.clear_reservations(flow - 0.1 * np.array([0.0, 2.0]), *single_class_data())
    np.testing.assert_allclose(result.flow, [0.1, 0.9], atol=1e-09)


def test_common_capacity_multiplier_clears_two_classes():
    coordinator = coordination_module()
    conservation = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    demand = np.ones(2)
    delta = np.array([[1, 0, 1, 0]], dtype=float)
    capacity = np.array([0.6])
    costs = np.array([0.0, 2.0, 0.0, 2.0])
    flow = np.array([0.3, 0.7, 0.3, 0.7])
    result = coordinator.clear_reservations(
        flow - 0.4 * costs, conservation, demand, delta, capacity
    )
    np.testing.assert_allclose(result.flow, flow, atol=1e-09)
    effective_cost = costs + delta.T[:, 0] * 2.0
    np.testing.assert_allclose(effective_cost, np.full(4, 2.0))


def test_infeasible_capacity_is_rejected():
    coordinator = coordination_module()
    with pytest.raises(ValueError, match="empty"):
        coordinator.clear_reservations(
            np.ones(2), np.ones((1, 2)), np.ones(1), np.ones((1, 2)), np.array([0.5])
        )


def test_invalid_dimensions_are_rejected():
    coordinator = coordination_module()
    with pytest.raises(ValueError, match="dimension"):
        coordinator.clear_reservations(np.ones(3), *single_class_data())


def test_solver_success_cannot_accept_a_large_nonprojection():
    coordinator = coordination_module()
    try:
        result = coordinator.clear_reservations(np.array([1000000.0, 0.0]), *single_class_data())
    except RuntimeError as error:
        assert "acceptance" in str(error)
    else:
        np.testing.assert_allclose(result.flow, [0.25, 0.75], atol=1e-08)
        assert result.first_order_gap <= 1e-07


def test_empty_capacity_family_is_supported():
    coordinator = coordination_module()
    result = coordinator.clear_reservations(
        np.array([1.0, 0.0]), np.ones((1, 2)), np.ones(1), np.empty((0, 2)), np.empty(0)
    )
    np.testing.assert_allclose(result.flow, [1.0, 0.0], atol=1e-08)


def test_empty_conservation_family_is_supported():
    coordinator = coordination_module()
    result = coordinator.clear_reservations(
        np.array([1.0, 0.0]), np.empty((0, 2)), np.empty(0), np.ones((1, 2)), np.array([0.5])
    )
    np.testing.assert_allclose(result.flow, [0.5, 0.0], atol=1e-08)


def test_complex_request_cannot_be_silently_changed():
    coordinator = coordination_module()
    with pytest.raises(ValueError, match="real"):
        coordinator.clear_reservations(np.array([1.0 + 2j, 0.0]), *single_class_data())


def test_float_resolution_cannot_hide_nonprojection_gap():
    coordinator = coordination_module()
    with pytest.raises(RuntimeError, match="acceptance.*scale"):
        coordinator.clear_reservations(np.array([1e16, 1e16]), *single_class_data())
