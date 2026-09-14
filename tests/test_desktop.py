"""Desktop launch and teardown: the spawned binary, and closing what the shell left open."""

import logging
import os
import subprocess
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest

from webstation_broker.emulators import base, desktop

from .conftest import DETACHED_CMD, SLEEPER_CMD, await_gone

DesktopTree = tuple[subprocess.Popen[bytes], int, str]
"""What the `desktop_tree` fixture yields: the shell, the pid it detached, and their tag."""


def _ppid(pid: int) -> int:
    """Read a process's parent pid, so a test can say where it is not.

    Args:
        pid: The process to look up.

    Returns:
        The parent pid, or 0 when the process is gone.
    """
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return 0
    for line in status.splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return 0


def test_launch_spawns_configured_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Launch spawns whatever DESKTOP_BIN names."""
    calls = []
    monkeypatch.setenv("DESKTOP_BIN", "custom-desktop")
    monkeypatch.setattr(desktop.Desktop, "_spawn", lambda self, cmd, env: calls.append(cmd))
    monkeypatch.setattr(desktop.Desktop, "stop", lambda self: None)

    desktop.Desktop().launch(None, None)

    assert calls == [["custom-desktop"]]


def test_launch_defaults_to_selkies_desktop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Launch falls back to selkies-desktop when DESKTOP_BIN is unset."""
    calls = []
    monkeypatch.delenv("DESKTOP_BIN", raising=False)
    monkeypatch.setattr(desktop.Desktop, "_spawn", lambda self, cmd, env: calls.append(cmd))
    monkeypatch.setattr(desktop.Desktop, "stop", lambda self: None)

    desktop.Desktop().launch(None, None)

    assert calls == [["selkies-desktop"]]


def test_launch_failure_is_logged_and_reraised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A spawn failure is logged and still propagates, so it does not vanish silently."""

    def _explode(self: desktop.Desktop, cmd: list[str], env: dict[str, str]) -> None:
        """Fail the way a missing binary does.

        Args:
            self: The desktop being launched.
            cmd: The argv that would have been spawned.
            env: The environment it would have been spawned into.

        Raises:
            OSError: Always.
        """
        raise OSError("no such binary")

    monkeypatch.setattr(desktop.Desktop, "_spawn", _explode)
    monkeypatch.setattr(desktop.Desktop, "stop", lambda self: None)

    with caplog.at_level(logging.ERROR, logger="webstation_broker.emulators.desktop"):
        with pytest.raises(OSError):
            desktop.Desktop().launch(None, None)

    assert any("failed to launch" in r.getMessage() for r in caplog.records)


def test_launch_stops_any_running_session_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Launch replaces whatever is already running before spawning a new one."""
    order = []
    monkeypatch.setattr(desktop.Desktop, "stop", lambda self: order.append("stop"))
    monkeypatch.setattr(desktop.Desktop, "_spawn", lambda self, cmd, env: order.append("spawn"))

    desktop.Desktop().launch(None, None)

    assert order == ["stop", "spawn"]


def test_launch_stamps_a_fresh_tag_each_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every desktop session is launched with its own tag.

    A tag reused between sessions would let one teardown sweep what an earlier
    session left behind, and would make a restarted broker unable to tell the
    two apart.
    """
    envs: list[dict[str, str]] = []
    monkeypatch.setattr(desktop.Desktop, "_spawn", lambda self, cmd, env: envs.append(env))
    monkeypatch.setattr(desktop.Desktop, "stop", lambda self: None)

    desktop.Desktop().launch(None, None)
    desktop.Desktop().launch(None, None)

    tags = [env.get(base.SESSION_TAG_ENV) for env in envs]
    assert all(tags)
    assert tags[0] != tags[1]


def test_the_double_fork_puts_the_app_out_of_reach_of_the_shell(
    desktop_tree: DesktopTree,
) -> None:
    """The app the shell started is neither below it nor in its process group.

    This is the whole reason the sweep exists: `Emulator.stop` signals the
    shell's process group, and a walk down from the shell finds nothing either,
    because the second fork left the app parented away from it.
    """
    shell, app_pid, _tag = desktop_tree

    assert _ppid(app_pid) != shell.pid
    assert os.getpgid(app_pid) != os.getpgid(shell.pid)


def test_the_tag_reaches_the_app_the_shell_detached(desktop_tree: DesktopTree) -> None:
    """What the tree lost, the inherited environment keeps.

    The app is two forks and an exec away from the shell and still carries the
    shell's tag, which is what makes it findable at teardown.
    """
    shell, app_pid, tag = desktop_tree

    assert base.session_tag(shell.pid) == tag
    assert base.session_tag(app_pid) == tag


def test_session_tag_is_none_for_an_untagged_process(
    sleeper: Callable[[], subprocess.Popen[bytes]],
) -> None:
    """A process launched outside a desktop session carries no tag."""
    assert base.session_tag(sleeper().pid) is None


def test_tagged_processes_finds_the_app_and_can_leave_the_shell_out(
    desktop_tree: DesktopTree,
) -> None:
    """The sweep sees the app, and the shell is excluded because stop handles it itself."""
    shell, app_pid, tag = desktop_tree

    found = base.tagged_processes(tag, exclude={shell.pid})

    assert (app_pid, DETACHED_CMD) in found
    assert all(pid != shell.pid for pid, _cmd in found)


def test_tagged_processes_matches_nothing_for_an_unused_tag(desktop_tree: DesktopTree) -> None:
    """The sweep is bounded by the tag, so another session's apps are not in it.

    Selkies, labwc and the broker's own service all run as the same user, so a
    sweep that matched more widely would take the stream down with the session.
    """
    assert base.tagged_processes(uuid.uuid4().hex) == []


def test_stop_takes_down_an_app_left_open_on_the_desktop(desktop_tree: DesktopTree) -> None:
    """Ending the desktop session closes an emulator the user left open on it."""
    shell, app_pid, _tag = desktop_tree
    emu = desktop.Desktop()
    emu._proc = shell

    emu.stop()

    assert await_gone(shell.pid, SLEEPER_CMD)
    assert await_gone(app_pid, DETACHED_CMD)


def test_stop_does_not_sweep_for_a_plain_emulator(desktop_tree: DesktopTree) -> None:
    """A plain emulator does not sweep, so its teardown cost stays a single signal.

    The desktop is the only session that starts other apps and detaches them;
    every other emulator keeps its helpers in its own process group, where the
    signal `Emulator.stop` already sends reaches them.
    """
    shell, app_pid, _tag = desktop_tree

    class _Plain(base.Emulator):
        """An emulator with the base teardown, standing in for every non-desktop session."""

        name = "plain"
        """Registry key this stand-in would be registered under."""

    emu = _Plain()
    emu._proc = shell

    emu.stop()

    assert await_gone(shell.pid, SLEEPER_CMD)
    assert base._cmdline(app_pid) == DETACHED_CMD


def test_stop_leaves_an_untagged_process_alone(
    desktop_tree: DesktopTree,
    sleeper: Callable[[], subprocess.Popen[bytes]],
) -> None:
    """Nothing outside the session is swept, however long the session ran."""
    shell, app_pid, _tag = desktop_tree
    bystander = sleeper()
    emu = desktop.Desktop()
    emu._proc = shell

    emu.stop()

    assert await_gone(app_pid, DETACHED_CMD)
    assert bystander.poll() is None


def test_stop_without_a_running_shell_sweeps_nothing(desktop_tree: DesktopTree) -> None:
    """With no shell to read a tag off, the teardown sweeps nothing rather than guessing.

    A desktop whose shell already exited has no tag to read, and matching on
    anything looser would reach processes this session never started.
    """
    _shell, app_pid, _tag = desktop_tree
    emu = desktop.Desktop()

    emu.stop()

    assert base._cmdline(app_pid) == DETACHED_CMD


def test_launch_sweeps_before_starting_a_new_shell(
    desktop_tree: DesktopTree, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Relaunching the desktop closes what the previous session left open.

    `launch` opens with `stop`, so a second desktop session does not start on
    top of the emulators the first one left running.
    """
    shell, app_pid, _tag = desktop_tree
    monkeypatch.setattr(desktop.Desktop, "_spawn", lambda self, cmd, env: None)
    emu = desktop.Desktop()
    emu._proc = shell

    emu.launch(None, None)

    assert await_gone(app_pid, DETACHED_CMD)


def test_kill_processes_skips_a_pid_that_no_longer_runs_its_argv(
    sleeper: Callable[[], subprocess.Popen[bytes]],
) -> None:
    """A snapshotted pid the kernel handed to something else is left alone.

    The snapshot is taken before the shell is signalled and acted on after, so
    a pid in it can have died and been reissued in between.
    """
    bystander = sleeper()

    assert base.kill_processes([(bystander.pid, ["/usr/bin/sleep", "somethingelse"])]) == 0
    assert bystander.poll() is None


def test_kill_processes_refuses_the_brokers_own_process(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The broker never signals itself, however badly a /proc read went.

    The sweep signals whole process groups, so an entry naming the broker would
    take the session down with the desktop it was tearing down.
    """
    own = (os.getpid(), base._cmdline(os.getpid()))

    with caplog.at_level(logging.ERROR, logger="webstation_broker.emulators.base"):
        assert base.kill_processes([own]) == 0

    assert any("broker's own" in r.getMessage() for r in caplog.records)


def test_kill_processes_on_an_empty_snapshot_does_nothing() -> None:
    """A desktop session that left nothing open sweeps nothing."""
    assert base.kill_processes([]) == 0
