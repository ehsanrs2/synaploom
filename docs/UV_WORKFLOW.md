# uv workflow

SynapLoom uses [uv](https://docs.astral.sh/uv/) for dependency resolution, virtual environments, command execution and CI.

## Bootstrap

```bash
uv sync
uv run pytest
```

uv creates and manages the project-local `.venv` automatically.

## Optional vision dependencies

Required for CIFAR-100 / torchvision feature extraction:

```bash
uv sync --extra vision
```

## Add or remove dependencies

Runtime dependency:

```bash
uv add <package>
```

Development-only dependency:

```bash
uv add --dev <package>
```

Vision extra:

```bash
uv add --optional vision <package>
```

Remove a dependency with the corresponding `uv remove` command.

Commit both `pyproject.toml` and `uv.lock` after dependency changes.

## Reproducible execution

For normal development, `uv run` automatically checks/synchronizes the project environment:

```bash
uv run pytest
uv run python experiments/scripts/m1_3_protocol_smoke.py
```

For preregistered/scientific runs, use the existing lockfile without allowing dependency resolution changes:

```bash
uv sync --locked --extra vision
uv run --locked python experiments/scripts/m1_3_h1a_preflight.py \
  --features data/features/cifar100_resnet18_imagenet1k_v1.pt
```

## Update dependencies deliberately

Check the lock:

```bash
uv lock --check
```

Upgrade all allowed dependencies:

```bash
uv lock --upgrade
```

Upgrade one dependency:

```bash
uv lock --upgrade-package <package>
```

Dependency upgrades that can affect scientific results should happen in their own PR and must not be mixed into an already preregistered experiment run.

## Python

The default development Python is pinned in `.python-version`. The package metadata still declares the supported Python range separately in `pyproject.toml`.

## CI

GitHub Actions uses `astral-sh/setup-uv`, synchronizes from the lockfile, and runs all checks through `uv run`.

Do not add a separate `requirements.txt` unless an external system explicitly requires one. If one is needed, derive it from uv rather than maintaining it manually.
