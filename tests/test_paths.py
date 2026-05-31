"""Unit tests for dinov3_hot.paths."""

from pathlib import Path

import pytest

from dinov3_hot.paths import resolve_labels_geojson


def test_resolve_direct_geojson_file(tmp_path: Path) -> None:
    geojson = tmp_path / "labels.geojson"
    geojson.write_text("{}")
    assert resolve_labels_geojson(geojson) == geojson


def test_resolve_directory_containing_geojson(tmp_path: Path) -> None:
    geojson = tmp_path / "labels.geojson"
    geojson.write_text("{}")
    assert resolve_labels_geojson(tmp_path) == geojson


def test_resolve_raises_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_labels_geojson(tmp_path)


def test_resolve_rejects_non_geojson_file(tmp_path: Path) -> None:
    other = tmp_path / "labels.txt"
    other.write_text("not geojson")
    with pytest.raises(FileNotFoundError):
        resolve_labels_geojson(other)
