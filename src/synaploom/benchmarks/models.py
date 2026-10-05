from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from synaploom.baselines import DenseControlModel
from synaploom.config import GraphConfig, ModelConfig
from synaploom.model import SynapLoomModel
from synaploom.vision import CifarConvEncoder


@dataclass(frozen=True)
class H1ModelShape:
    total_nodes: int = 12
    initial_recruited: int = 2
    top_k: int = 2
    d_model: int = 128
    d_hidden: int = 256
    router_dim: int = 128
    q_max: int = 0
    num_classes: int = 100


def synaploom_config(
    shape: H1ModelShape,
    *,
    initial_recruited: int | None = None,
) -> ModelConfig:
    recruited = shape.initial_recruited if initial_recruited is None else initial_recruited
    return ModelConfig(
        graph=GraphConfig(
            num_nodes=shape.total_nodes,
            initial_recruited=recruited,
            d_model=shape.d_model,
            d_hidden=shape.d_hidden,
            router_dim=shape.router_dim,
            top_k=shape.top_k,
            q_max=shape.q_max,
            message_rounds=0,
        ),
        input_dim=shape.d_model,
        num_classes=shape.num_classes,
    )


def make_growth_model(shape: H1ModelShape) -> SynapLoomModel:
    return SynapLoomModel(
        synaploom_config(shape),
        encoder=CifarConvEncoder(shape.d_model),
    )


def make_routed_no_growth_model(shape: H1ModelShape) -> SynapLoomModel:
    return SynapLoomModel(
        synaploom_config(shape),
        encoder=CifarConvEncoder(shape.d_model),
    )


def make_all_recruited_routed_model(shape: H1ModelShape) -> SynapLoomModel:
    return SynapLoomModel(
        synaploom_config(shape, initial_recruited=shape.total_nodes),
        encoder=CifarConvEncoder(shape.d_model),
    )


def make_dense_model(shape: H1ModelShape, *, d_hidden: int) -> DenseControlModel:
    return DenseControlModel(
        input_dim=shape.d_model,
        d_model=shape.d_model,
        d_hidden=d_hidden,
        num_classes=shape.num_classes,
        encoder=CifarConvEncoder(shape.d_model),
    )
