"""Administration session that launches the full webstation desktop.

Starts selkies-desktop so the user can configure emulators through the GUI.
Managed like any emulator session, just with no ROM and no save sync.

Teardown is the one place it is not like the others. The shell starts each app
with a double fork, so ending the session has to find those apps by the tag
they inherited rather than by the process tree they are no longer in (see
`Desktop.stop`).
"""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from . import base
from .base import Emulator, base_launch_env

log = logging.getLogger(__name__)


class Desktop(Emulator):
    """The webstation desktop, run as a session with no ROM and no save sync.

    Attributes:
        name: Registry key, `desktop`.
        display_name: Shown as "Webstation Desktop".
        requires_rom: Off; a desktop session boots nothing.
        save_root: Left at `/config`.
        save_subtrees: Empty, so the save routes have nothing to dump or restore.
        log_path: `/config/selkies-desktop.log`.
    """

    name = "desktop"
    """Registry key for the desktop session."""
    display_name = "Webstation Desktop"
    """Name the UI shows for the desktop session."""
    requires_rom = False
    """A desktop session boots nothing, so no ROM is needed."""
    save_root = Path("/config")
    """Root of the writable data; nothing under it is synced."""
    save_subtrees = ()
    """Empty: there is no save data to dump or restore."""
    log_path = Path("/config/selkies-desktop.log")
    """Where selkies-desktop's output is appended."""
    term_timeout = float(os.environ.get("DESKTOP_STOP_WAIT", "15"))
    """Seconds a desktop teardown waits on SIGTERM, from `DESKTOP_STOP_WAIT` (default 15).

    Longer than the base default because the shell itself is not what it is
    spent on: it is spent on the emulators the user left open, which write
    their config and save data on the way out. One budget covers the shell and
    those apps together, since `stop` signals them all before waiting on any.
    """
    _tag: Optional[str] = None
    """The tag `launch` stamped into this session, kept for when the shell is gone.

    The session outlives the shell process: quitting the desktop from the GUI
    ends it and leaves its apps running, and `/proc` only holds an environment
    for as long as its process lives. The broker keeps one launcher per session
    (`sess["emulator_obj"]`), so what `launch` recorded is still here at `stop`.
    """

    def launch(self, rom_path: Optional[Path], resume_slot: Optional[int]) -> None:
        """Start selkies-desktop, replacing any session already running.

        The binary comes from `DESKTOP_BIN` (default `selkies-desktop`). It is
        launched with a tag in its environment, which every app started from
        the desktop inherits and which `stop` then sweeps by.

        Args:
            rom_path: Ignored; the desktop has no content to boot.
            resume_slot: Ignored; the desktop has no state.

        Raises:
            OSError: When the binary cannot be started or its pid cannot be
                recorded; logged here since this launch failure otherwise
                surfaced nowhere.
        """
        self.stop()
        binary = os.environ.get("DESKTOP_BIN", "selkies-desktop")
        env = base_launch_env()
        # Fresh per launch, so a sweep can never match something a previous
        # desktop session left behind and a restarted broker reads the live
        # session's own tag off the shell rather than guessing at one.
        self._tag = env[base.SESSION_TAG_ENV] = uuid.uuid4().hex
        try:
            self._spawn([binary], env)
        except OSError:
            log.exception("desktop: failed to launch %s", binary)
            raise

    def stop(self) -> None:
        """Stop the shell and close everything the user left open on the desktop.

        selkies-desktop runs a `.desktop` entry's `Exec=` line through fork,
        `setsid`, a second fork and exec. The app that comes out is parented to
        pid 1, sits in a process group whose leader has already exited, and
        draws on a compositor that is a separate service, so nothing about it
        ends when the shell does: an emulator opened to be configured and left
        open kept running, and kept rendering into the stream, after the
        session that started it had ended.

        What it does still carry is the environment it inherited, which is why
        `launch` stamps a tag into it. The tag is preferred off the live shell
        and falls back to the one `launch` kept, because the shell is the first
        thing to go: quitting the desktop from inside the GUI ends it and
        leaves every app it started running, which is precisely the case this
        sweep exists for and the one where there is no environment left to read.

        The apps are signalled before the shell is waited on, so the one grace
        period `term_timeout` buys is spent on both at once. Signalling them
        afterwards instead put two full budgets back to back on the exit route.
        """
        proc = self._proc
        tag = base.session_tag(proc.pid) if proc is not None and proc.poll() is None else None
        tag = tag or self._tag
        exclude = {proc.pid} if proc is not None else set()
        left_open = base.tagged_processes(tag, exclude=exclude) if tag else []
        deadline = time.monotonic() + self.term_timeout
        signalled = base.term_processes(left_open)
        super().stop()
        closed = base.kill_survivors(signalled, deadline, self.kill_timeout)
        if closed:
            log.info("desktop: closed %d app(s) left open on the desktop", closed)
        # Dropped only once nothing carries the tag any more, which is not the
        # same as everything signalled having gone: an app the sweep could not
        # signal at all never reaches `signalled`, so counting that as a clean
        # result would throw away the one handle left on a process still
        # running. Kept otherwise for the same reason `Emulator.stop` keeps the
        # pid record: the next `stop` is the next chance to use it.
        if tag and not base.tagged_processes(tag, exclude=exclude):
            self._tag = None
