from synaploom.data.synthetic import make_gaussian_cluster_stream
from synaploom.protocols.h1a import H1AProtocol, build_h1a_variant, task_start_recruitment_steps


def test_task_start_schedule_and_variant_capacity() -> None:
    protocol = H1AProtocol(
        feature_dim=4,
        num_classes=6,
        classes_per_task=2,
        d_model=8,
        d_hidden=12,
        num_nodes=5,
        initial_recruited=2,
        top_k=2,
        epochs_per_task=2,
        batch_size=4,
        eval_batch_size=8,
        seeds=(0,),
    )
    stream = make_gaussian_cluster_stream(
        num_tasks=3,
        classes_per_task=2,
        input_dim=4,
        train_per_class=4,
        eval_per_class=2,
        seed=1,
    )
    # Each task has 8 examples -> 2 batches/epoch -> 4 steps/task.
    assert task_start_recruitment_steps(stream, protocol) == (4, 8)

    progressive, _, engine, meta = build_h1a_variant(
        "progressive_reserve", stream=stream, seed=0, protocol=protocol
    )
    all_recruited, _, _, _ = build_h1a_variant(
        "all_recruited", stream=stream, seed=0, protocol=protocol
    )

    assert progressive.graph_state.recruited_count() == 2
    assert all_recruited.graph_state.recruited_count() == 5
    assert engine is not None
    assert meta["task_start_recruitment_steps"] == [4, 8]
