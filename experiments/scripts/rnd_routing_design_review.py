"""Read-only artifact audit; no model execution or profiling."""
import argparse
import hashlib
import json
from pathlib import Path
from synaploom.research.grouped_performance import trace_summary


def review(root):
    result = json.loads((root/'results.json').read_text())
    cells = []
    reviewed = set([root/'results.json'])
    for cell in result['cells']:
        shape = cell['shape']
        path = root/f"b{shape['batch']}-r{shape['recruited']}-k{shape['top_k']}"
        assert cell['parity']['state_unchanged'] and cell['parity']['selected_ids_equal']
        assert cell['parity']['predictions_equal'] and cell['parity']['singleton_allclose']
        operators = {}
        entries = dict(cell['forward'], **cell['independent_checks'])
        for name, item in entries.items():
            trace = path/f'{name}.json'
            table = path/f'{name}.txt'
            assert trace_summary(trace) == item['trace'], str(trace)
            assert table.read_text().strip()
            reviewed.update([trace, table])
            operators[name] = table.read_text()
        for item in cell['forward'].values():
            assert item['work']['effective_expert_rows'] == shape['batch']*shape['top_k']
        cells.append(dict(shape=shape, parameters=cell['parameters'], state_sha256=cell['state_sha256'],
            parity=cell['parity'], forward=cell['forward'], independent_checks=cell['independent_checks'],
            operator_tables=operators))
    files = {p for p in root.rglob('*') if p.is_file()}
    assert reviewed == files, f'unreviewed artifacts: {files-reviewed}'
    return dict(source=str(root), metadata=result['metadata'], cells=cells,
        artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cpu', type=Path, required=True)
    p.add_argument('--cuda', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = {dev:review(getattr(args,dev)) for dev in ('cpu','cuda')}
    (args.output/'review.json').write_text(json.dumps(data,indent=2)+'\n')
