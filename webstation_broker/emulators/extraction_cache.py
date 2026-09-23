"""Shared, opt-in extraction cache for emulator modules that boot from an archive or package.

Every tunable a consumer supplies is a zero-argument callable, never a
snapshotted value: consumer modules read their own config from module-level
globals (e.g. `rpcs3.CACHE_DIR`) that existing tests monkeypatch at call
time, and a captured value here would make that monkeypatching a silent
no-op.
"""
from __future__ import annotations

import hashlib
import logging
import os  # noqa: F401
import shutil  # noqa: F401
import subprocess  # noqa: F401
import tempfile  # noqa: F401
import threading
import time
import zipfile  # noqa: F401
from contextlib import contextmanager  # noqa: F401
from pathlib import Path
from typing import Callable, Iterator, Optional  # noqa: F401

from .base import Emulator

log = logging.getLogger(__name__)

_GB = 1024**3
"""Bytes per GB, the unit max_gb and the space guard are expressed in."""

_ARCHIVE_EXTS = (".7z", ".zip", ".rar")
"""Archive formats the class's own default safe-extract stage understands."""

_LAST_ACCESSED_MARKER = ".last_accessed"
"""Marker file inside a cache entry, touched on every hit, read to pick an LRU victim."""

_SCRATCH_DIR_NAME = ".scratch"
"""Subdirectory of a cache dir every staged extraction lives under until renamed into place."""


def _cache_key(rom: Path) -> str:
    """Cache dir name for rom: its stem plus a short hash of the file's identity.

    A bare stem collides two ROMs that share a name but differ in extension,
    and survives a same-named re-upload with different content, either of
    which would otherwise serve up whatever is sitting in the old cache dir
    as if it were the new ROM. The hash covers the resolved path, the size,
    and the nanosecond mtime: same-second rewrites are exactly how a library
    sync replaces a dump, so second granularity would let a replacement keep
    the old key.

    Args:
        rom: The archive or package being extracted.

    Returns:
        The cache directory name for this ROM.

    Raises:
        RuntimeError: If the file cannot be read. Falling back to the bare
            name here would hand back the collision-prone key this function
            exists to avoid, and the extraction that follows would fail on
            the same unreadable file anyway.
    """
    try:
        st = rom.stat()
        fingerprint = f"{rom.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    except OSError as exc:
        log.error("extraction cache: could not read %s to key its extraction: %s", rom, exc)
        raise RuntimeError(f"could not read {rom.name} to key its extraction: {exc}") from exc
    digest = hashlib.sha1(fingerprint.encode()).hexdigest()[:12]
    return f"{rom.stem}-{digest}"


def _dir_size(path: Path) -> int:
    """Sum all file sizes under path, skipping the last-accessed marker.

    The marker file is skipped so the LRU eviction logic doesn't count it
    toward the cache size, which would pollute the accounting with a file
    that exists only for bookkeeping.

    Args:
        path: The directory to measure.

    Returns:
        The total size in bytes of all files under path, excluding the marker.
    """
    total = 0
    for f in path.rglob("*"):
        if f.name == _LAST_ACCESSED_MARKER:
            continue
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError as exc:
            log.debug("extraction cache: skipping unreadable %s while sizing %s: %s", f, path, exc)
            continue
    return total


def _touch_last_accessed(game_dir: Path) -> None:
    """Write a marker file in game_dir with the current Unix timestamp.

    The marker is used by LRU eviction to identify which cache entries have
    been recently accessed; touching it on each hit provides the eviction
    logic with a mtime-based candidate list.

    Args:
        game_dir: The cache entry directory to mark as accessed now.
    """
    try:
        (game_dir / _LAST_ACCESSED_MARKER).write_text(str(time.time()))
    except OSError as exc:
        log.warning("extraction cache: could not update last-accessed marker for %s: %s", game_dir, exc)


class ExtractionCache:
    """A per-instance, opt-in archive/pkg extraction cache.

    One instance owns one cache directory tree and one lock; no two
    instances may ever own or nest the same directory (a convention this
    class does not itself enforce).
    """

    def __init__(
        self,
        name: str,
        cache_dir: Callable[[], Path],
        enabled: Callable[[], bool],
        max_gb: Callable[[], float],
        find_boot_target: Callable[[Path], Optional[Path]],
        *,
        budget: Optional[Callable[[Path], tuple[int, int]]] = None,
        stage: Optional[Callable[[Path, Path, Path, Emulator, int], None]] = None,
        phase_name: Callable[[Path], str] = lambda rom: "extracting_archive",
        on_evict: Optional[Callable[[Path], None]] = None,
        expansion_factor: Callable[[], float] = lambda: 4.0,
        extract_timeout: Callable[[], float] = lambda: 1800.0,
        lock_wait: Optional[Callable[[], float]] = lambda: 120.0,
        missing_target_error: str = "held no bootable target",
    ) -> None:
        """Initialize the extraction cache with callables for all configuration.

        All configuration parameters are stored as callables, never snapshotted,
        so consumer modules can monkeypatch their module-level config at test time
        and this instance will read the current values on each call.

        Args:
            name: The cache name, for logging and identification.
            cache_dir: Callable returning the cache root directory.
            enabled: Callable returning True if the cache is enabled.
            max_gb: Callable returning the maximum cache size in GB.
            find_boot_target: Callable taking a Path and returning the bootable
                target inside it, or None.
            budget: Optional callable for cache budget calculation.
            stage: Optional callable for custom extraction staging.
            phase_name: Callable returning the phase name for a ROM.
            on_evict: Optional callable run when evicting a cache entry.
            expansion_factor: Callable returning the expected extraction expansion factor.
            extract_timeout: Callable returning extraction timeout in seconds.
            lock_wait: Optional callable returning lock wait timeout in seconds.
            missing_target_error: Error message when no bootable target is found.
        """
        self._name = name
        self._cache_dir = cache_dir
        self._enabled = enabled
        self._max_gb = max_gb
        self._find_boot_target = find_boot_target
        self._budget = budget
        self._stage = stage
        self._phase_name = phase_name
        self._on_evict = on_evict
        self._expansion_factor = expansion_factor
        self._extract_timeout = extract_timeout
        self._lock_wait = lock_wait
        self._missing_target_error = missing_target_error
        self._lock = threading.Lock()

    def root(self) -> Path:
        """The configured cache directory, read live from the `cache_dir` callable."""
        return self._cache_dir()

    def _cache_size_bytes(self) -> int:
        """Sum the sizes of all cache entries, or zero if the cache dir doesn't exist yet.

        Returns:
            The total size in bytes of all files in all cache entries under root(),
                or 0 if root() is not a directory.
        """
        cache_dir = self._cache_dir()
        if not cache_dir.is_dir():
            return 0
        return sum(_dir_size(d) for d in cache_dir.iterdir() if d.is_dir())

    def _evict_lru(self, needed_bytes: int, keep: str) -> None:
        """Evict least-recently-used cache entries until `needed_bytes` fits within max_gb.

        Args:
            needed_bytes: Additional bytes that must fit under the cache cap.
            keep: The cache key currently being (re-)extracted, so a stale
                entry for it already removed by the caller is never chosen.
        """
        cache_dir = self._cache_dir()
        if not self._enabled() or not cache_dir.is_dir():
            return
        max_bytes = int(self._max_gb() * _GB)
        current = self._cache_size_bytes()
        while current + needed_bytes > max_bytes:
            candidates = []
            for game_dir in cache_dir.iterdir():
                if not game_dir.is_dir() or game_dir.name in (keep, _SCRATCH_DIR_NAME):
                    continue
                marker = game_dir / _LAST_ACCESSED_MARKER
                try:
                    mtime = marker.stat().st_mtime if marker.exists() else 0.0
                except OSError as exc:
                    log.debug(
                        "%s extraction cache: could not read last-accessed marker for %s: %s",
                        self._name, game_dir, exc,
                    )
                    mtime = 0.0
                candidates.append((mtime, game_dir))
            if not candidates:
                log.warning(
                    "%s extraction cache: nothing left to evict under the %.0f GB cap",
                    self._name, self._max_gb(),
                )
                return
            candidates.sort(key=lambda c: c[0])
            victim = candidates[0][1]
            victim_size = _dir_size(victim)
            log.info("%s extraction cache: evicting %s (least recently used)", self._name, victim.name)
            try:
                shutil.rmtree(victim)
            except OSError as exc:
                log.warning("%s extraction cache: could not evict %s: %s", self._name, victim, exc)
                return
            current -= victim_size
            if self._on_evict is not None:
                self._on_evict(victim)

    def _require_room(self, peak_bytes: int, kept_bytes: int, rom_name: str) -> None:
        """Refuse an extraction that cannot fit before any of it is written.

        Two different figures cover two different ceilings: the cache cap
        counts only what survives (`kept_bytes`), while the free-space guard
        counts what is on disk at the extraction's worst moment (`peak_bytes`),
        which can exceed what is kept when a consumer's staging needs scratch
        space alongside its final output.

        Args:
            peak_bytes: Bytes on disk at the height of the extraction.
            kept_bytes: Bytes the finished extraction leaves in the cache.
            rom_name: The ROM being extracted, named in the error.

        Raises:
            RuntimeError: If the cache cap or the filesystem cannot hold it.
        """
        max_bytes = int(self._max_gb() * _GB)
        current = self._cache_size_bytes()
        if current + kept_bytes > max_bytes:
            raise RuntimeError(
                f"{rom_name} would leave about {kept_bytes / _GB:.1f} GB cached, more than "
                f"max_gb ({self._max_gb():.0f} GB) allows with {current / _GB:.1f} GB already there"
            )
        cache_dir = self._cache_dir()
        try:
            free = shutil.disk_usage(cache_dir).free
        except OSError as exc:
            log.warning(
                "%s extraction cache: could not read free space on %s: %s", self._name, cache_dir, exc,
            )
            return
        if free < peak_bytes:
            raise RuntimeError(
                f"{rom_name} needs about {peak_bytes / _GB:.1f} GB to extract, but only "
                f"{free / _GB:.1f} GB is free on {cache_dir}"
            )

    @contextmanager
    def _locked(self, what: str) -> Iterator[None]:
        """Hold this instance's lock for the block.

        Blocks with no timeout when `lock_wait` is None; otherwise gives up
        and raises after `lock_wait()` seconds.

        Args:
            what: The operation waiting for the lock, named in the log and the error.

        Raises:
            RuntimeError: When a bounded `lock_wait` elapses before the lock is free.
        """
        timeout = -1.0 if self._lock_wait is None else self._lock_wait()
        if not self._lock.acquire(timeout=timeout):
            log.error(
                "%s extraction cache: %s gave up after waiting %.0fs for the cache lock",
                self._name, what, timeout,
            )
            raise RuntimeError(
                f"another {self._name} extraction is still running; {what} waited "
                f"{timeout:.0f}s for the extraction cache"
            )
        try:
            yield
        finally:
            self._lock.release()

    def _clear_scratch(self) -> None:
        """Remove every staged extraction under the scratch dir. Callers must hold `_locked`.

        The lock is what makes this safe: no extraction can be mid-flight
        while it is held, so anything still sitting here was orphaned by a
        process that died.
        """
        scratch_root = self._cache_dir() / _SCRATCH_DIR_NAME
        if not scratch_root.is_dir():
            return
        for entry in scratch_root.iterdir():
            log.warning("%s extraction cache: removing orphaned scratch dir %s", self._name, entry.name)
            shutil.rmtree(entry, ignore_errors=True)

    def sweep_stale_extractions(self) -> None:
        """Remove extraction scratch dirs orphaned by a crashed broker process.

        Call once at broker startup: the only other caller is an extraction,
        which a library of already-extracted (or never-archived) titles may
        never run again.
        """
        with self._locked(f"{self._name} startup scratch sweep"):
            self._clear_scratch()
