from synaploom.research.performance import measure


def test_instrumentation_parity(tmp_path):
    a = measure(steps=1)
    b = measure(steps=1, instrumented=True, trace_path=tmp_path / 'trace.json')
    assert a['final_state_sha256'] == b['final_state_sha256']
    assert not a['instrumented'] and b['instrumented']
    assert 'check_model_gradients' in b['table']
    assert (tmp_path / 'trace.json').exists()
