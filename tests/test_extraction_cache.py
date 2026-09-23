"""Tests for the shared, opt-in archive/pkg extraction cache."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

from webstation_broker.emulators.base import Emulator
from webstation_broker.emulators.extraction_cache import ExtractionCache


def _make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def _touch(path: Path, mtime: Optional[float] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 5)
    if mtime is not None:
        import os
        os.utime(path, (mtime, mtime))


class _FakeEmulator(Emulator):
    """Minimal Emulator stand-in exposing only extraction_phase."""


def _find_eboot(root: Path) -> Optional[Path]:
    for candidate in root.rglob("EBOOT.BIN"):
        return candidate
    return None


def test_root_returns_the_configured_cache_dir(tmp_path: Path) -> None:
    """root() reflects the cache_dir callable, read live rather than snapshotted."""
    cache_dir = tmp_path / "cache"
    cache = ExtractionCache(
        name="test",
        cache_dir=lambda: cache_dir,
        enabled=lambda: True,
        max_gb=lambda: 10.0,
        find_boot_target=_find_eboot,
    )
    assert cache.root() == cache_dir


def test_root_reflects_a_live_change_to_the_cache_dir_callable(tmp_path: Path) -> None:
    """A constructor callable is re-read on every call, not captured once at construction."""
    current = {"dir": tmp_path / "first"}
    cache = ExtractionCache(
        name="test",
        cache_dir=lambda: current["dir"],
        enabled=lambda: True,
        max_gb=lambda: 10.0,
        find_boot_target=_find_eboot,
    )
    assert cache.root() == tmp_path / "first"
    current["dir"] = tmp_path / "second"
    assert cache.root() == tmp_path / "second"
