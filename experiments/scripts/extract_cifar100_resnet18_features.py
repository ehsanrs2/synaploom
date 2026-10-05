from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.models import ResNet18_Weights, resnet18


def _extract(loader: DataLoader, model: nn.Module, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    features, targets = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            z = model(x.to(device))
            features.append(z.cpu())
            targets.append(y.long().cpu())
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen ImageNet ResNet-18 features for M1.3 H1a.")
    parser.add_argument("--root", type=Path, default=Path("data/cifar100"))
    parser.add_argument("--output", type=Path, default=Path("data/features/cifar100_resnet18_imagenet1k_v1.pt"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    weights = ResNet18_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    train = CIFAR100(root=str(args.root), train=True, download=args.download, transform=transform)
    test = CIFAR100(root=str(args.root), train=False, download=args.download, transform=transform)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    device = torch.device(args.device)
    backbone.to(device)

    train_features, train_targets = _extract(train_loader, backbone, device)
    eval_features, eval_targets = _extract(test_loader, backbone, device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_features": train_features,
        "train_targets": train_targets,
        "eval_features": eval_features,
        "eval_targets": eval_targets,
        "metadata": {
            "dataset": "CIFAR100",
            "feature_extractor": "torchvision.resnet18",
            "weights": "IMAGENET1K_V1",
            "feature_dim": int(train_features.shape[1]),
            "train_examples": int(train_features.shape[0]),
            "eval_examples": int(eval_features.shape[0]),
            "torch_version": torch.__version__,
        },
    }
    torch.save(payload, args.output)
    print(json.dumps(payload["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
