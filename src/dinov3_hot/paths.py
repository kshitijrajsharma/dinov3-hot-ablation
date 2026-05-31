"""Path/IO helpers: label-geojson resolver, generic checkpoint downloader."""

import tempfile
from pathlib import Path


def resolve_labels_geojson(path: Path) -> Path:
    """Direct `.geojson` file or a directory containing exactly one `.geojson`."""
    if path.is_file() and path.suffix == ".geojson":
        return path
    if path.is_dir():
        match = next(iter(path.glob("*.geojson")), None)
        if match is not None:
            return match
    raise FileNotFoundError(f"No .geojson at {path}")


def download_checkpoint(url: str) -> Path:
    """Fetch a Lightning checkpoint via upath; supports http(s)/s3/local."""
    # upath/universal-pathlib is an optional dep for s3/http; callers only pay this cost
    # when they actually need remote ckpt download.
    from upath import UPath

    local = Path(tempfile.mkdtemp()) / UPath(url).name
    local.write_bytes(UPath(url).read_bytes())
    return local
