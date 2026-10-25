"""Explicit manual representative mode; bounded default is qualification only."""
import argparse
from synaploom.research.grouped_performance import run

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--representative', action='store_true')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if min(args.threads, args.repeats) < 1:
        p.error('positive counts required')
    kwargs = dict(batches=(1, 32, 128), ratios=((4, 1), (4, 4), (16, 4)),
                  width=256, hidden=512, nodes=16, router_dim=64, classes=100) if args.representative else {}
    run(args.output, device=args.device, threads=args.threads, repeats=args.repeats, **kwargs)
