import json
import subprocess
import sys

import pytest
import torch

from synaploom.research.grouped_performance import run


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_isolated_profiling_and_output_protection(device, tmp_path):
    root = tmp_path/'fresh'
    # CUPTI activity collection must start in a fresh process: initializing a
    # CPU-only profiler first can suppress CUDA activities in this torch build.
    subprocess.run([sys.executable, '-c',
        'import sys; from synaploom.research.grouped_performance import run; '
        'run(sys.argv[1],device=sys.argv[2],batches=(2,),ratios=((2,1),),repeats=1)',
        str(root), device], check=True, capture_output=True, text=True)
    result = json.loads((root/'results.json').read_text())
    cell = result['cells'][0]
    assert all(cell['parity'][key] for key in ('state_unchanged', 'predictions_equal', 'selected_ids_equal'))
    for name, entry in cell['forward'].items():
        assert entry['work']['effective_expert_rows'] == 2
        assert len(entry['wall_seconds']) == 1
        if device == 'cuda':
            assert entry['trace']['kernel_count'] > 0
            assert entry['trace']['launch_api_count'] > 0
    assert set(cell['independent_checks']) == {'finite_tensor','gradient_diagnostics',
                                             'parameter_diagnostics','model_state_sha256'}
    before = (root/'results.json').read_bytes()
    with pytest.raises(FileExistsError):
        run(root, device=device)
    assert (root/'results.json').read_bytes() == before
