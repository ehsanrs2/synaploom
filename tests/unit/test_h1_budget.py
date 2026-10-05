from synaploom.benchmarks.active_budget import synaploom_capacity
from synaploom.benchmarks.models import H1ModelShape, make_growth_model
from synaploom.benchmarks.runner import build_dense_matches


def test_synaploom_capacity_separates_total_and_executed() -> None:
    shape = H1ModelShape(total_nodes=6, initial_recruited=2, top_k=2, d_model=32, d_hidden=48, router_dim=32)
    model = make_growth_model(shape)
    capacity = synaploom_capacity(model, executed_nodes=2, recruited_nodes=2)
    assert capacity.total_parameters > capacity.estimated_executed_parameters
    assert capacity.estimated_recruited_parameters == capacity.estimated_executed_parameters


def test_dense_budget_matcher_returns_two_views() -> None:
    shape = H1ModelShape(total_nodes=6, initial_recruited=2, top_k=2, d_model=32, d_hidden=48, router_dim=32)
    matches = build_dense_matches(shape)
    assert set(matches) == {"dense_total_match", "dense_executed_match"}
    assert matches["dense_total_match"][0] >= matches["dense_executed_match"][0]
