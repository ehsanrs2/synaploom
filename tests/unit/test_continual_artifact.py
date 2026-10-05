import json

import torch

from synaploom.training.continual import ContinualRunResult, StructuralSnapshot, save_continual_result


def test_continual_result_serializes_nan_as_null(tmp_path) -> None:
    result = ContinualRunResult(
        accuracy_matrix=torch.tensor([[0.5, float("nan")], [0.4, 0.6]]),
        eval_loss_matrix=torch.tensor([[1.0, float("nan")], [1.1, 0.9]]),
        train_loss_by_task=[1.2, 1.0],
        structural_history=[
            StructuralSnapshot(
                stage=0,
                task_id=0,
                structural_version=0,
                recruited_nodes=2,
                active_edges=0,
                mean_executed_nodes=1.0,
                selection_entropy=0.0,
                max_selection_share=1.0,
                node_selection_counts={0: 2},
            )
        ],
        global_steps=4,
    )
    path = tmp_path / "result.json"
    save_continual_result(result, path)
    payload = json.loads(path.read_text())
    assert payload["accuracy_matrix"][0][1] is None
    assert "average_forgetting" in payload["summary"]
    assert payload["run_config"] == {}
    assert payload["stream_manifest"] == []
