set shell := ["bash", "-uc"]

default:
    @just --list

setup:
    uv sync --all-groups
    uv run pre-commit install

lint:
    uv run ruff check --fix .
    uv run ruff format .
    uv run ty check src tests

test:
    uv run pytest -q

smoke:
    uv run dinov3-hot train --config conf/train.yaml --data-pct 1 --max-epochs 5 --run-name smoke

train:
    uv run dinov3-hot train --config conf/train.yaml

predict raster out:
    uv run dinov3-hot predict --raster {{raster}} --out {{out}}

export ckpt out:
    uv run dinov3-hot export --ckpt {{ckpt}} --out {{out}}

eval-fair:
    uv run dinov3-hot eval-fair --samples-dir /home/krschap/code/personal/geoml-toolkits/banepa_test
