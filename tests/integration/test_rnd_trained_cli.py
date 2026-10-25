import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel


def test_strict_checkpoint_cli_and_no_overwrite(tmp_path):
    config = ModelConfig(graph=GraphConfig(num_nodes=6, initial_recruited=2, top_k=2,
        d_model=8, d_hidden=16, router_dim=8), input_dim=8, num_classes=3)
    torch.manual_seed(23)
    model = SynapLoomModel(config)
    torch.save(model.state_dict(), tmp_path / 'model.pt')
    (tmp_path / 'config.json').write_text(json.dumps({'kind': 'progressive_reserve', **asdict(config)}))
    torch.save({'x': torch.randn(24, 8), 'y': torch.arange(24) % 3,
                'sample_ids': [f'row-{i}' for i in range(24)]}, tmp_path / 'data.pt')
    command = [sys.executable, 'experiments/scripts/rnd_batch_composition.py',
        '--checkpoint', str(tmp_path / 'model.pt'), '--config', str(tmp_path / 'config.json'),
        '--dataset', str(tmp_path / 'data.pt'), '--output', str(tmp_path / 'out')]
    subprocess.run(command, check=True, capture_output=True)
    output = tmp_path / 'out/results.json'
    original = output.read_bytes()
    result = json.loads(original)['results']['progressive_reserve']
    assert result['rows'][0]['anchor_id'] == 'row-0'
    assert result['state_sha256'] == result['final_state_sha256']
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert output.read_bytes() == original
