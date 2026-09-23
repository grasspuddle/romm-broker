"""Tests for the shared, opt-in archive/pkg extraction cache."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Callable, Optional

import pytest

from webstation_broker.emulators import extraction_cache
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


def _cache(tmp_path: Path, *, enabled: bool = True, max_gb: float = 8 / 1024**3,
           on_evict: Optional[Callable[[Path], None]] = None) -> ExtractionCache:
    cache_dir = tmp_path / "cache"
    return ExtractionCache(
        name="test", cache_dir=lambda: cache_dir, enabled=lambda: enabled,
        max_gb=lambda: max_gb, find_boot_target=_find_eboot, on_evict=on_evict,
    )


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


def test_cache_key_combines_stem_and_a_content_fingerprint(tmp_path: Path) -> None:
    """The key is the stem plus a short hash of the resolved path, size, and mtime."""
    rom = tmp_path / "Game.zip"
    rom.write_bytes(b"data")
    key = extraction_cache._cache_key(rom)
    assert key.startswith("Game-")
    assert len(key) == len("Game-") + 12


def test_cache_key_differs_for_files_sharing_a_stem_but_not_an_extension(tmp_path: Path) -> None:
    """Two archives that share a stem but differ in extension never collide."""
    a = tmp_path / "Game.zip"
    b = tmp_path / "Game.7z"
    a.write_bytes(b"same content")
    b.write_bytes(b"same content")
    assert extraction_cache._cache_key(a) != extraction_cache._cache_key(b)


def test_cache_key_changes_when_a_same_named_file_is_replaced(tmp_path: Path) -> None:
    """A same-named re-upload with different content never reuses the old cache entry."""
    rom = tmp_path / "Game.zip"
    rom.write_bytes(b"original")
    first = extraction_cache._cache_key(rom)
    rom.write_bytes(b"replaced, different size")
    second = extraction_cache._cache_key(rom)
    assert first != second


def test_cache_key_raises_when_the_file_cannot_be_read(tmp_path: Path) -> None:
    """An unreadable file raises rather than falling back to the collision-prone bare stem."""
    missing = tmp_path / "Missing.zip"
    with pytest.raises(RuntimeError, match="could not read"):
        extraction_cache._cache_key(missing)


def test_dir_size_sums_files_and_skips_the_marker(tmp_path: Path) -> None:
    """Dir size sums files and skips the last-accessed marker."""
    game_dir = tmp_path / "Game"
    _touch(game_dir / "eboot.bin")
    _touch(game_dir / "sub" / "data.bin")
    _touch(game_dir / extraction_cache._LAST_ACCESSED_MARKER)
    assert extraction_cache._dir_size(game_dir) == 10


def test_touch_last_accessed_writes_a_marker_file(tmp_path: Path) -> None:
    """Touch last accessed writes a marker file game_dir did not have before."""
    game_dir = tmp_path / "Game"
    _touch(game_dir / "eboot.bin")
    extraction_cache._touch_last_accessed(game_dir)
    assert (game_dir / extraction_cache._LAST_ACCESSED_MARKER).exists()


def test_cache_size_bytes_sums_across_every_game_dir(tmp_path: Path) -> None:
    """Cache size bytes sums across every game dir under root()."""
    cache_dir = tmp_path / "cache"
    _touch(cache_dir / "GameA" / "eboot.bin")
    _touch(cache_dir / "GameB" / "eboot.bin")
    cache = ExtractionCache(
        name="test", cache_dir=lambda: cache_dir, enabled=lambda: True,
        max_gb=lambda: 10.0, find_boot_target=_find_eboot,
    )
    assert cache._cache_size_bytes() == 10


def test_cache_size_bytes_is_zero_without_a_cache_dir(tmp_path: Path) -> None:
    """Cache size bytes is zero when the configured cache dir does not exist yet."""
    cache = ExtractionCache(
        name="test", cache_dir=lambda: tmp_path / "never-created", enabled=lambda: True,
        max_gb=lambda: 10.0, find_boot_target=_find_eboot,
    )
    assert cache._cache_size_bytes() == 0


def test_evict_lru_is_a_noop_when_disabled(tmp_path: Path) -> None:
    """Evict LRU is a no-op when the cache is disabled."""
    cache = _cache(tmp_path, enabled=False)
    game_dir = cache.root() / "GameA"
    _touch(game_dir / "eboot.bin")
    cache._evict_lru(10**9, "SomethingElse")
    assert game_dir.exists()


def test_evict_lru_removes_the_least_recently_used_entry_first(tmp_path: Path) -> None:
    """Evict LRU removes the least recently used entry first."""
    cache = _cache(tmp_path)
    old = cache.root() / "Old"
    new = cache.root() / "New"
    _touch(old / "eboot.bin")
    _touch(new / "eboot.bin")
    _touch(old / extraction_cache._LAST_ACCESSED_MARKER, mtime=1000)
    _touch(new / extraction_cache._LAST_ACCESSED_MARKER, mtime=2000)
    cache._evict_lru(2, "Incoming")
    assert not old.exists()
    assert new.exists()


def test_evict_lru_never_removes_the_entry_being_extracted(tmp_path: Path) -> None:
    """Evict LRU never removes the entry currently being (re-)extracted."""
    cache = _cache(tmp_path, max_gb=1 / 1024**3)
    keep = cache.root() / "Incoming"
    _touch(keep / "eboot.bin")
    _touch(keep / extraction_cache._LAST_ACCESSED_MARKER, mtime=1)
    cache._evict_lru(50, "Incoming")
    assert keep.exists()


def test_evict_lru_calls_on_evict_once_per_evicted_dir(tmp_path: Path) -> None:
    """on_evict fires once per successfully evicted dir, after the rmtree."""
    evicted: list[Path] = []
    cache = _cache(tmp_path, max_gb=6 / 1024**3, on_evict=evicted.append)
    old = cache.root() / "Old"
    _touch(old / "eboot.bin")
    _touch(old / extraction_cache._LAST_ACCESSED_MARKER, mtime=1)
    cache._evict_lru(2, "Incoming")
    assert evicted == [old]
    assert not old.exists()


def test_require_room_refuses_when_the_cache_cap_would_be_exceeded(tmp_path: Path) -> None:
    """require_room refuses an extraction that would push the cache past max_gb."""
    cache = _cache(tmp_path, max_gb=1 / 1024**3)
    with pytest.raises(RuntimeError, match="max_gb"):
        cache._require_room(2, 2, "Game.zip")


def test_require_room_refuses_when_free_disk_is_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """require_room refuses an extraction whose peak would not fit on the free disk."""
    cache = _cache(tmp_path, max_gb=100.0)
    cache.root().mkdir(parents=True)
    monkeypatch.setattr(
        extraction_cache.shutil, "disk_usage",
        lambda path: type("U", (), {"free": 1})(),
    )
    with pytest.raises(RuntimeError, match="free on"):
        cache._require_room(10**9, 1, "Game.zip")


def test_require_room_charges_the_cap_on_kept_and_the_disk_on_peak(tmp_path: Path) -> None:
    """A budget where peak and kept differ charges each guard its own figure."""
    cache = _cache(tmp_path, max_gb=100.0)
    cache.root().mkdir(parents=True)
    # kept (1 byte) fits the cap; peak (huge) must still be checked against
    # free disk space rather than being ignored because kept passed.
    free = shutil.disk_usage(str(cache.root())).free
    cache._require_room(1, 1, "Game.zip")  # both tiny: passes without raising
    with pytest.raises(RuntimeError, match="needs about"):
        cache._require_room(free + 10**12, 1, "Game.zip")
