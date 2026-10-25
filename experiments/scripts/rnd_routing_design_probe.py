"""Bounded design investigations by default; representative mode is manual-only."""
import argparse
import json
import platform
from pathlib import Path
import torch

from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.research.grouped_performance import timed, work
from synaploom.research.sample_routing import SampleRoutingReference
from synaploom.research.grouped_sample_routing import GroupedSampleRouting
from synaploom.research.routing_design_probes import SingletonProbe, SlotRestoreProbe, PrecomputedRowsProbe
from synaploom.training.state_digest import model_state_sha256


def run(output, device='cpu', representative=False, repeats=5):
    output=Path(output)
    output.mkdir(parents=True, exist_ok=False)
    threads, tf32 = torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
        width, hidden, router, classes = (256,512,64,100) if representative else (8,16,8,3)
        batches = (1,128) if representative else (1,8)
        cells=[]
        for batch in batches:
            for recruited in (4,16):
                path=output/f'b{batch}-r{recruited}-k4';path.mkdir()
                torch.manual_seed(7)
                model=SynapLoomModel(ModelConfig(input_dim=width,num_classes=classes,
                    graph=GraphConfig(num_nodes=16,initial_recruited=recruited,q_max=0,
                        top_k=4,d_model=width,d_hidden=hidden,router_dim=router))).to(device).eval()
                x=torch.randn(batch,width,generator=torch.Generator().manual_seed(8)).to(device)
                before=model_state_sha256(model)
                costs={}
                with torch.inference_mode():
                    oracle=SampleRoutingReference(model)(x)
                    for name, cls in [('row_reference',SampleRoutingReference),('grouped',GroupedSampleRouting),
                                      ('singleton_probe',SingletonProbe),('slot_restore_probe',SlotRestoreProbe),('rows_precomputed_probe',PrecomputedRowsProbe)]:
                        fn=cls(model)
                        actual=fn(x)
                        tol=1e-6 if device=='cpu' else 1e-5
                        assert torch.equal(actual.selected_node_ids,oracle.selected_node_ids)
                        torch.testing.assert_close(actual.logits,oracle.logits,atol=tol,rtol=tol)
                        torch.testing.assert_close(actual.weights,oracle.weights,atol=tol,rtol=tol)
                        accounting=work(model,fn,x)
                        assert accounting['effective_expert_rows']==batch*4
                        costs[name]=dict(work=accounting,parity_logit_linf=float((actual.logits-oracle.logits).abs().max()),
                            **timed(lambda:fn(x),device,repeats,path/f'{name}.json'))
                        if device=='cuda' and not costs[name]['trace']['kernel_count']:
                            raise RuntimeError('missing CUDA activity; use a fresh profiling process')
                        assert before==model_state_sha256(model)
                cells.append(dict(shape=dict(batch=batch,recruited=recruited,k=4,width=width,hidden=hidden,
                                             router_dim=router,classes=classes),costs=costs,state_sha256=before))
        metadata=dict(device=device,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if device=='cuda' else None,python=platform.python_version(),
            threads=1,tf32=False,seed=7,warmups=2,repeats=repeats,representative=representative,
            residency='device resident; initial H2D excluded; dispatch D2H included; hashes outside forward timing')
        (output/'results.json').write_text(json.dumps(dict(metadata=metadata,cells=cells),indent=2)+'\n')
    finally:
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32=tf32


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--representative',action='store_true')
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if args.repeats<1:p.error('positive repeats required')
    run(args.output,args.device,args.representative,args.repeats)
