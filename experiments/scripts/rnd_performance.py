"""Manual CPU/GPU profiling command; GPU requires explicit --device cuda."""
import argparse
import json
from pathlib import Path
from synaploom.research.performance import measure

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ordinary = measure(device=args.device, threads=args.threads, steps=args.steps)
    profiled = measure(device=args.device, threads=args.threads, steps=args.steps,
                       instrumented=True, trace_path=args.output / "trace.json")
    (args.output / "operators.txt").write_text(profiled["table"])
    if ordinary["final_state_sha256"] != profiled["final_state_sha256"]:
        raise RuntimeError("instrumentation parity failed")
    (args.output / "timing.json").write_text(json.dumps([ordinary, profiled], indent=2))
