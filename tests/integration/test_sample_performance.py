from synaploom.research.sample_performance import measure


def test_bounded_profile_parity(tmp_path):
    kwargs = dict(steps=1, batch=4, width=8, hidden=16, nodes=4, recruited=2, top_k=2, classes=3)
    a = measure(**kwargs)
    b = measure(**kwargs, instrumented=True, trace_path=tmp_path/'trace.json')
    assert a['final_state_sha256'] == b['final_state_sha256']
    assert 'SparseRouter.forward' in b['table']
    assert (tmp_path/'trace.json').exists()
