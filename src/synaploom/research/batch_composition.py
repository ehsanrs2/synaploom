"""Exploratory fixed-state diagnostics; does not alter scientific runners."""
from __future__ import annotations

import hashlib
from collections import defaultdict

import torch
from torch import nn

from synaploom.training.evaluation import logits_from_output
from synaploom.training.state_digest import model_state_sha256


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((str(value.dtype), list(value.shape))).encode() +
                          value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def grouping(labels, anchor, size, composition, seed):
    if size < 1 or not 0 <= anchor < len(labels):
        raise ValueError("invalid anchor or batch size")
    if composition not in {"same", "other", "mixed"}:
        raise ValueError("unknown composition")
    pool = [i for i in range(len(labels)) if i != anchor and
            (composition == "mixed" or (int(labels[i]) == int(labels[anchor])) ==
             (composition == "same"))]
    if len(pool) < size - 1:
        raise ValueError("insufficient distinct companions")
    generator = torch.Generator().manual_seed(seed + anchor * 1009)
    order = torch.randperm(len(pool), generator=generator).tolist()
    ordered = [pool[i] for i in order]
    if composition == "mixed" and size > 1:
        other = next((i for i in ordered if int(labels[i]) != int(labels[anchor])), None)
        if other is None:
            raise ValueError("mixed composition requires another class")
        ordered.remove(other)
        ordered.insert(0, other)
    return [anchor] + ordered[:size - 1]


def guarded_forward(model, x):
    before = model_state_sha256(model)
    try:
        with torch.inference_mode():
            output = model(x)
            logits = logits_from_output(output).detach().cpu().clone()
            if not torch.isfinite(logits).all():
                raise RuntimeError("nonfinite diagnostic logits")
            trace = getattr(output, "trace", None)
            ids = [] if trace is None else trace.selected_node_ids.cpu().tolist()
            weights = [] if trace is None else trace.router_weights.cpu().tolist()
            query = [] if trace is None else trace.routing_context.cpu().tolist()
            return logits, dict(zip(ids, weights)), query
    finally:
        if model_state_sha256(model) != before:
            raise RuntimeError("model-state mutation during diagnostic forward")


def compare(reference, observed):
    a, aw, aq = reference
    b, bw, bq = observed
    sa, sb = set(aw), set(bw)
    weights = sum(abs(aw.get(i, 0.) - bw.get(i, 0.)) for i in sa | sb)
    delta = float((a - b).abs().max())
    return {"jaccard": len(sa & sb) / len(sa | sb) if sa | sb else 1.,
            "overlap": len(sa & sb), "weight_l1": weights,
            "query_linf": max((abs(x-y) for x, y in zip(aq, bq)), default=0.),
            "logit_linf": delta,
            "prediction_flip": int(a.argmax() != b.argmax()),
            "confidence_delta": float(b.softmax(-1).max() - a.softmax(-1).max()),
            "selection_dependent": sa != sb, "weight_dependent": weights > 1e-6,
            "prediction_dependent": delta > 1e-6}


def diagnose(model: nn.Module, x, y, *, sample_ids=None, anchors=None,
             sizes=(2, 4, 8), seed=7):
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("expected aligned feature matrix and labels")
    sample_ids = list(range(len(x))) if sample_ids is None else list(sample_ids)
    anchors = list(range(len(x))) if anchors is None else list(anchors)
    if len(sample_ids) != len(x) or len(set(sample_ids)) != len(x):
        raise ValueError("sample IDs must be unique and aligned")
    if not anchors or len(set(anchors)) != len(anchors):
        raise ValueError("anchors must be nonempty and unique")
    modes = [(module, module.training) for module in model.modules()]
    initial = model_state_sha256(model)
    input_hash, label_hash = tensor_sha256(x), tensor_sha256(y)
    rows = []
    try:
        model.eval()
        for anchor in anchors:
            if not 0 <= anchor < len(x):
                raise ValueError("anchor outside dataset")
            logits, weights, query = guarded_forward(model, x[[anchor]])
            reference = logits[0], weights, query
            for size in sizes:
                for composition in ("same", "other", "mixed"):
                    indices = grouping(y.cpu(), anchor, size, composition, seed)
                    logits, weights, query = guarded_forward(model, x[indices])
                    observed = logits[indices.index(anchor)], weights, query
                    reversed_ids = list(reversed(indices))
                    pl, pw, pq = guarded_forward(model, x[reversed_ids])
                    permutation = compare(observed, (pl[reversed_ids.index(anchor)], pw, pq))
                    rows.append({"anchor_id": sample_ids[anchor],
                                 "anchor_sha256": tensor_sha256(x[anchor]),
                                 "member_ids": [sample_ids[i] for i in indices],
                                 "size": size, "composition": composition,
                                 "baseline_correct": int(reference[0].argmax() == y[anchor].cpu()),
                                 "correct": int(observed[0].argmax() == y[anchor].cpu()),
                                 "reference_logits": reference[0].tolist(),
                                 "logits": observed[0].tolist(),
                                 "selected_ids": list(weights), "weights": weights,
                                 "reference_weights": reference[1],
                                 **compare(reference, observed), "permutation": permutation,
                                 "permutation_consistent": bool(torch.allclose(
                                     observed[0], pl[reversed_ids.index(anchor)],
                                     atol=1e-6, rtol=1e-6)) and
                                     permutation["weight_l1"] <= 1e-6 and
                                     permutation["jaccard"] == 1.})
    finally:
        for module, mode in modes:
            module.training = mode
        if tensor_sha256(x) != input_hash or tensor_sha256(y) != label_hash:
            raise RuntimeError("diagnostic input mutation")
        if model_state_sha256(model) != initial:
            raise RuntimeError("model-state mutation during diagnostic")
    groups = defaultdict(list)
    for row in rows:
        groups[f'{row["size"]}/{row["composition"]}'].append(row)
    metrics = ("jaccard", "overlap", "weight_l1", "query_linf", "logit_linf",
               "prediction_flip", "confidence_delta", "correct", "baseline_correct",
               "selection_dependent", "weight_dependent", "prediction_dependent")
    return {"label": "EXPLORATORY", "seed": seed, "state_sha256": initial,
            "final_state_sha256": model_state_sha256(model),
            "input_sha256": tensor_sha256(x), "labels_sha256": tensor_sha256(y),
            "unique_anchors": len(anchors), "rows": rows,
            "summary": {key: {metric: sum(r[metric] for r in values) / len(values)
                              for metric in metrics} for key, values in groups.items()}}
