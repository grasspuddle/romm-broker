"""Shared, opt-in extraction cache for emulator modules that boot from an archive or package.

Every tunable a consumer supplies is a zero-argument callable, never a
snapshotted value: consumer modules read their own config from module-level
globals (e.g. `rpcs3.CACHE_DIR`) that existing tests monkeypatch at call
time, and a captured value here would make that monkeypatching a silent
no-op.
"""
from __future__ import annotations

import hashlib  # noqa: F401
import logging
import os  # noqa: F401
import shutil  # noqa: F401
import subprocess  # noqa: F401
import tempfile  # noqa: F401
import threading
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
