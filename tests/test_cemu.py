"""Cemu ROM resolution, settings.xml patching, pad profile seeding, and the exit-time save refresh.

Covers picking a bootable title out of a folder, the settings keys the broker
pins, the GamePad profile it seeds, and which saves exit re-stamps.
"""

import logging
import os
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional, Union

import pytest

from webstation_broker import imports
from webstation_broker.emulators import cemu

from .conftest import import_zip, preflight_import, restore_import


@pytest.fixture
def rom_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the Cemu ROM root at a fresh directory under tmp_path.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: The per-test temporary directory.

    Returns:
        The ROM root directory.
    """
    root = tmp_path / "romm"
    root.mkdir()
    monkeypatch.setattr(cemu, "ROM_ROOT", root)
    return root


@pytest.fixture
def config_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the Cemu settings and pad profile paths under tmp_path.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: The per-test temporary directory.

    Returns:
        The Cemu config directory; it is not created.
    """
    d = tmp_path / "config" / "Cemu"
    monkeypatch.setattr(cemu, "SETTINGS_PATH", d / "settings.xml")
    monkeypatch.setattr(cemu, "PROFILE_PATH", d / "controllerProfiles" / "controller0.xml")
    return d


@pytest.fixture
def save_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the Cemu MLC tree and its save directory under tmp_path.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        tmp_path: The per-test temporary directory.

    Returns:
        The usr/save directory inside the MLC tree.
    """
    mlc = tmp_path / "mlc01"
    save = mlc / "usr" / "save"
    save.mkdir(parents=True)
    monkeypatch.setattr(cemu, "MLC_DIR", mlc)
    monkeypatch.setattr(cemu, "SAVE_DIR", save)
    # The save subtree hangs off the MLC root, which the class resolves once at
    # import, so the clear would reach outside tmp_path without this.
    monkeypatch.setattr(cemu.Cemu, "save_root", mlc)
    return save


def _touch(path: Path, mtime: Optional[float] = None) -> Path:
    """Write a placeholder file, creating parents, optionally with a fixed mtime.

    Args:
        path: The file to create.
        mtime: Modification time to stamp on it, if any.

    Returns:
        The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_rom_pick_prefers_the_archive_over_the_raw_image(rom_root: Path) -> None:
    """A .wua beside a .wud of the same game is the one picked."""
    game = rom_root / "Game"
    _touch(game / "Game.wud")
    best = _touch(game / "Game.wua")

    assert cemu.Cemu().resolve_rom_file(game) == best


def test_rom_pick_finds_the_rpx_inside_an_extracted_dump(rom_root: Path) -> None:
    """An extracted dump resolves to the .rpx under its code directory."""
    game = rom_root / "Game"
    rpx = _touch(game / "code" / "Game.rpx")
    _touch(game / "content" / "data.bin")
    _touch(game / "meta" / "meta.xml")

    assert cemu.Cemu().resolve_rom_file(game) == rpx


def test_rom_pick_reaches_a_dump_wrapped_in_a_library_folder(rom_root: Path) -> None:
    """A dump nested one folder deeper still resolves to its .rpx."""
    game = rom_root / "Game"
    rpx = _touch(game / "Game [TITLEID]" / "code" / "Game.rpx")

    assert cemu.Cemu().resolve_rom_file(game) == rpx


def test_rom_pick_skips_the_update_beside_the_base_game(rom_root: Path) -> None:
    """An update dump beside the base game image is not the thing booted."""
    game = rom_root / "Game"
    _touch(game / "Game (Update)" / "code" / "Game.rpx")
    base = _touch(game / "Game.wux")

    assert cemu.Cemu().resolve_rom_file(game) == base


def test_rom_pick_refuses_a_link_out_of_the_library(rom_root: Path, tmp_path: Path) -> None:
    """A title symlinked from outside the ROM root is never picked."""
    outside = _touch(tmp_path / "outside" / "Game.wua")
    game = rom_root / "Game"
    game.mkdir()
    (game / "Game.wua").symlink_to(outside)

    assert cemu.Cemu().resolve_rom_file(game) is None


def test_resolve_takes_a_file_as_given(rom_root: Path) -> None:
    """A path that is already a file resolves to itself."""
    rom = _touch(rom_root / "Game.wux")
    assert cemu.Cemu().resolve_rom_file(rom) == rom


def test_settings_are_seeded_when_the_file_is_missing(config_dir: Path) -> None:
    """A missing settings.xml is created with updates and Discord presence off."""
    cemu._patch_settings()

    root = ET.parse(cemu.SETTINGS_PATH).getroot()
    assert root.tag == "content"
    assert root.find("check_update").text == "false"
    assert root.find("use_discord_presence").text == "false"


def test_settings_patch_keeps_what_the_user_tuned(config_dir: Path) -> None:
    """Patching pins the broker's keys and keeps every other key the user set."""
    config_dir.mkdir(parents=True)
    cemu.SETTINGS_PATH.write_text(
        "<content><check_update>true</check_update>"
        "<vsync>1</vsync><graphic_api>1</graphic_api></content>"
    )

    cemu._patch_settings()

    root = ET.parse(cemu.SETTINGS_PATH).getroot()
    assert root.find("check_update").text == "false"
    assert root.find("vsync").text == "1"
    assert root.find("graphic_api").text == "1"


def test_settings_pins_the_audio_device_to_default(config_dir: Path) -> None:
    """TVDevice and PadDevice are forced to the cubeb default sentinel."""
    cemu._patch_settings()

    root = ET.parse(cemu.SETTINGS_PATH).getroot()
    audio = root.find("Audio")
    assert audio.find("TVDevice").text == "default"
    assert audio.find("PadDevice").text == "default"


def test_settings_patch_overwrites_a_blank_audio_device(config_dir: Path) -> None:
    """An existing blank TVDevice/PadDevice, Cemu's own default, is patched too."""
    config_dir.mkdir(parents=True)
    cemu.SETTINGS_PATH.write_text(
        "<content><Audio><TVDevice /><PadDevice /></Audio></content>"
    )

    cemu._patch_settings()

    root = ET.parse(cemu.SETTINGS_PATH).getroot()
    audio = root.find("Audio")
    assert audio.find("TVDevice").text == "default"
    assert audio.find("PadDevice").text == "default"


def test_a_broken_settings_file_is_reseeded_not_fatal(config_dir: Path) -> None:
    """A settings.xml that does not parse is reseeded instead of raising."""
    config_dir.mkdir(parents=True)
    cemu.SETTINGS_PATH.write_text("<content><unclosed>")

    cemu._patch_settings()

    root = ET.parse(cemu.SETTINGS_PATH).getroot()
    assert root.find("check_update").text == "false"


def test_an_unwritable_settings_file_aborts_the_launch(config_dir: Path) -> None:
    """A settings.xml the broker cannot write must stop the launch, not pass it.

    Launching anyway parks Cemu on its Getting Started modal while the
    activate reports a healthy session.
    """
    # A plain file where the config directory belongs fails every write.
    config_dir.parent.mkdir(parents=True)
    config_dir.write_text("not a directory")

    with pytest.raises(RuntimeError, match="broker settings"):
        cemu._patch_settings()


def test_crc16_matches_the_arc_check_value() -> None:
    """The CRC-16 implementation produces the standard ARC check value."""
    assert cemu._crc16(b"123456789") == 0xBB3D


def test_sdl_guid_is_the_interposers_name_based_fallback() -> None:
    """With crc=0, the GUID is a zero bus and crc followed by the pad name."""
    assert cemu._sdl_guid(0) == "000000004d6963726f736f6674205800"


def test_sdl_guid_matches_the_interposers_measured_guid_with_crc() -> None:
    """With the real name crc, the GUID matches the one measured against a live interposer."""
    assert cemu._sdl_guid(cemu._crc16(cemu._PAD_NAME.encode())) == "000081b84d6963726f736f6674205800"


def test_pad_profile_carries_both_guid_variants(config_dir: Path) -> None:
    """The seeded GamePad profile lists two distinct controller GUIDs with full mappings."""
    cemu._seed_pad_profile()

    root = ET.parse(cemu.PROFILE_PATH).getroot()
    assert root.find("type").text == "Wii U GamePad"
    uuids = [c.find("uuid").text for c in root.findall("controller")]
    assert len(uuids) == 2 and len(set(uuids)) == 2
    for controller in root.findall("controller"):
        entries = controller.find("mappings").findall("entry")
        assert len(entries) == len(cemu._VPAD_SDL_MAPPINGS)


def test_pad_profile_is_not_overwritten_once_present(config_dir: Path) -> None:
    """An existing pad profile is left exactly as the player tuned it."""
    cemu.PROFILE_PATH.parent.mkdir(parents=True)
    cemu.PROFILE_PATH.write_text("player tuned")

    cemu._seed_pad_profile()

    assert cemu.PROFILE_PATH.read_text() == "player tuned"


def test_pad_uuids_can_be_pinned_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """CEMU_PAD_UUIDS overrides the computed pad UUIDs with a comma-separated list."""
    monkeypatch.setenv("CEMU_PAD_UUIDS", "0_aaaa, 1_bbbb")
    assert cemu._pad_uuids() == ["0_aaaa", "1_bbbb"]


def test_clearing_the_slot_drops_the_last_sessions_saves(save_dir: Path) -> None:
    """Every title save left on the mlc goes before the next player's archive lands.

    A restore only writes the members the incoming archive names, so a title the
    archive does not mention would stay readable by this player and be swept into
    their dump at exit.
    """
    stale = _touch(save_dir / "00050000" / "aaaaaaaa" / "user" / "80000001" / "old.dat")
    other = _touch(save_dir / "00050000" / "bbbbbbbb" / "user" / "80000001" / "old.dat")

    cemu.Cemu().clear_working_slot()

    assert not stale.exists()
    assert not other.exists()
    # The tree itself is where the restore extracts to.
    assert save_dir.is_dir()


def test_clearing_the_slot_keeps_the_account_cemu_created_for_itself(save_dir: Path) -> None:
    """The account store under usr/save survives the clear.

    Cemu writes the account itself on first boot and every save path is keyed by
    its persistent id, so deleting it would strand the saves the restore is about
    to lay down.
    """
    account = _touch(save_dir / "system" / "act" / "80000001" / "account.dat")
    stale = _touch(save_dir / "00050000" / "aaaaaaaa" / "user" / "80000001" / "old.dat")

    cemu.Cemu().clear_working_slot()

    assert account.exists()
    assert not stale.exists()


def test_a_missing_save_tree_is_nothing_to_clear(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A container whose mlc has not been created yet clears cleanly."""
    monkeypatch.setattr(cemu.Cemu, "save_root", tmp_path / "absent")

    cemu.Cemu().clear_working_slot()


def test_launch_always_states_the_mlc_the_dump_reads_back(
    save_dir: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mlc path must be passed even without CEMU_MLC_DIR set.

    XDG_DATA_HOME moves MLC_DIR too, and the dump reads back whichever tree
    the command line named.
    """
    monkeypatch.delenv("CEMU_MLC_DIR", raising=False)
    spawned: list[list[str]] = []
    monkeypatch.setattr(
        cemu.Cemu,
        "_spawn",
        lambda self, cmd, env, stdin_pipe=False: spawned.append(cmd),
    )
    rom = tmp_path / "game.wua"
    rom.write_bytes(b"")

    cemu.Cemu().launch(rom, resume_slot=None)

    assert spawned[0][spawned[0].index("-m") + 1] == str(cemu.MLC_DIR)


def test_launch_creates_the_mlc_cemu_is_pointed_at(
    save_dir: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cemu refuses an mlc path that is not there, so the broker makes one."""
    mlc = tmp_path / "fresh-mlc"
    monkeypatch.setattr(cemu, "MLC_DIR", mlc)
    monkeypatch.setattr(cemu.Cemu, "_spawn", lambda self, cmd, env, stdin_pipe=False: None)
    rom = tmp_path / "game.wua"
    rom.write_bytes(b"")

    cemu.Cemu().launch(rom, resume_slot=None)

    assert mlc.is_dir()


def test_launch_sends_cemu_to_the_config_the_broker_just_patched(
    save_dir: Path, config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spawned emulator resolves the same config directory the broker writes into.

    Nothing on Cemu's command line names it, so the exported XDG root is the
    whole of the agreement: let it drift and the patched settings.xml and the
    seeded pad profile belong to a Cemu that is not the one running.
    """
    monkeypatch.setattr(cemu, "CONFIG_DIR", tmp_path / "cfg" / "Cemu")
    monkeypatch.setattr(cemu, "DATA_DIR", tmp_path / "data" / "Cemu")
    spawned: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(
        cemu.Cemu,
        "_spawn",
        lambda self, cmd, env, stdin_pipe=False: spawned.update(env=env),
    )
    rom = tmp_path / "game.wua"
    rom.write_bytes(b"")

    cemu.Cemu().launch(rom, resume_slot=None)

    env = spawned["env"]
    assert Path(env["XDG_CONFIG_HOME"]) / "Cemu" == cemu.CONFIG_DIR
    assert Path(env["XDG_DATA_HOME"]) / "Cemu" == cemu.DATA_DIR


def test_exit_refreshes_only_the_title_saves_this_session_wrote(save_dir: Path) -> None:
    """Exit re-stamps every file of a title written this session and nothing else."""
    emu = cemu.Cemu()
    emu._session_start = time.time() - 100

    stale = _touch(
        save_dir / "00050000" / "aaaaaaaa" / "user" / "80000001" / "old.dat",
        mtime=emu._session_start - 500,
    )
    partner = _touch(
        save_dir / "00050000" / "bbbbbbbb" / "user" / "80000001" / "old.dat",
        mtime=emu._session_start - 500,
    )
    _touch(save_dir / "00050000" / "bbbbbbbb" / "user" / "80000001" / "new.dat")
    system = _touch(
        save_dir / "system" / "act" / "80000001" / "account.dat",
        mtime=emu._session_start + 10,
    )
    system_mtime = system.stat().st_mtime

    emu.save_and_exit(10)

    # The touched title ships whole; the untouched title and the system
    # tree keep their stamps.
    assert partner.stat().st_mtime >= emu._session_start
    assert stale.stat().st_mtime < emu._session_start
    assert system.stat().st_mtime == system_mtime


def test_exit_without_a_launch_restamps_nothing(save_dir: Path) -> None:
    """A save_and_exit that never saw a launch must not claim every title's saves.

    A zero baseline is newer than every file on disk, which would drag
    unrelated titles into this session's dump.
    """
    other = _touch(
        save_dir / "00050000" / "aaaaaaaa" / "user" / "80000001" / "old.dat",
        mtime=time.time() - 5000,
    )
    before = other.stat().st_mtime

    cemu.Cemu().save_and_exit(10)

    assert other.stat().st_mtime == before


def test_exit_skips_a_save_dir_whose_low_half_is_not_a_title_id(save_dir: Path) -> None:
    """A `usr/save/<high>/<name>` whose low half is not 8 hex is not a title save.

    Both halves feed the restamp walk, so an unvalidated low half hands
    whatever name is on disk straight to it.
    """
    emu = cemu.Cemu()
    emu._session_start = time.time() - 100

    stray = _touch(
        save_dir / "00050000" / "not-a-title-id" / "old.dat",
        mtime=emu._session_start - 500,
    )
    _touch(save_dir / "00050000" / "not-a-title-id" / "new.dat")

    emu.save_and_exit(10)

    assert stray.stat().st_mtime < emu._session_start


def test_exit_survives_a_save_tree_that_cannot_be_listed(
    save_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A save tree that fails to list is logged, not raised through the exit path.

    The MLC can vanish under the walk, and the caller still has a process to
    stop and a report to hand back.
    """

    def boom(self: Path) -> None:
        raise OSError("mlc went away")

    monkeypatch.setattr(Path, "iterdir", boom)
    emu = cemu.Cemu()
    emu._session_start = time.time() - 100

    with caplog.at_level(logging.WARNING, logger="webstation_broker.emulators.cemu"):
        report = emu.save_and_exit(10)

    assert report == {"state_saved": None, "state_slot": None, "state_file": None}
    assert any("could not list the save tree" in r.getMessage() for r in caplog.records)


def test_exit_survives_a_title_dir_that_cannot_be_walked(
    save_dir: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A title dir that fails to walk is logged and stepped over, not raised."""
    _touch(save_dir / "00050000" / "aaaaaaaa" / "user" / "80000001" / "old.dat")

    def boom(self: Path, pattern: str) -> None:
        raise OSError("save vanished")

    monkeypatch.setattr(Path, "rglob", boom)
    emu = cemu.Cemu()
    emu._session_start = time.time() - 100

    with caplog.at_level(logging.WARNING, logger="webstation_broker.emulators.cemu"):
        report = emu.save_and_exit(10)

    assert report == {"state_saved": None, "state_slot": None, "state_file": None}
    assert any("could not walk" in r.getMessage() for r in caplog.records)


def test_exit_reports_no_state(save_dir: Path) -> None:
    """Exit reports that Cemu has no save state to offer."""
    report = cemu.Cemu().save_and_exit(10)
    assert report == {"state_saved": None, "state_slot": None, "state_file": None}


# -- declared imports --

_ROMM = imports.RomRef(1, "Game", "wiiu", title_id="1010EC00")
"""The rom an activate for the session's game carries; RomM sends the low half of the title id."""
_TITLE_DIR = "usr/save/00050000/1010ec00"
"""Where the session's game keeps its save, below the MLC root."""


def _preflight(
    members: dict[str, bytes],
    *,
    rom: Optional[imports.RomRef] = _ROMM,
    v1: Optional[dict[str, bytes]] = None,
) -> imports.PreflightResult:
    """Preflight an archive of import members against a Cemu whose MLC is patched into tmp_path.

    Args:
        members: `.import/<kind>/...` names mapped to bytes.
        rom: The activate body's rom, or None.
        v1: Ordinary archive members to carry beside them, or None.

    Returns:
        What preflight decided.
    """
    return preflight_import(cemu.Cemu(), import_zip(members, v1), rom_file=None, rom=rom)


def _save_member(rel: str) -> imports.ImportMember:
    """Build a save member as preflight hands one to a hook, with no archive behind it.

    Args:
        rel: The member's path below `.import/save/`.

    Returns:
        The member; its data cannot be read.
    """
    return imports.ImportMember(
        f".import/save/{rel}",
        "save",
        "unknown",
        PurePosixPath(rel),
        tuple(rel.split("/")),
        1,
        zipfile.ZipInfo(rel),
    )


@pytest.mark.usefixtures("save_dir")
@pytest.mark.parametrize(
    ("rel", "dest"),
    [
        ("usr/save/00050000/1010EC00/user/80000001/slot0.dat", f"{_TITLE_DIR}/user/80000001/slot0.dat"),
        ("mlc01/usr/save/00050000/1010ec00/user/80000001/slot0.dat", f"{_TITLE_DIR}/user/80000001/slot0.dat"),
        (
            "storage_mlc/usr/save/00050000/1010ec00/user/80000001/slot0.dat",
            f"{_TITLE_DIR}/user/80000001/slot0.dat",
        ),
        ("save/00050000/1010ec00/user/80000001/slot0.dat", f"{_TITLE_DIR}/user/80000001/slot0.dat"),
        ("usr/save/000500001010EC00/user/80000001/slot0.dat", f"{_TITLE_DIR}/user/80000001/slot0.dat"),
        ("usr/save/00050000/1010ec00/user/80000007/slot0.dat", f"{_TITLE_DIR}/user/80000001/slot0.dat"),
        ("usr/save/00050000/1010ec00/user/common/opts.dat", f"{_TITLE_DIR}/user/common/opts.dat"),
        ("usr/save/00050000/1010ec00/meta/meta.xml", f"{_TITLE_DIR}/meta/meta.xml"),
        ("user/80000003/slot0.dat", f"{_TITLE_DIR}/user/80000001/slot0.dat"),
        ("user/common/opts.dat", f"{_TITLE_DIR}/user/common/opts.dat"),
    ],
    ids=[
        "usr/save",
        "mlc01",
        "storage_mlc",
        "save",
        "saviine",
        "donor account",
        "common",
        "meta",
        "anchorless",
        "anchorless common",
    ],
)
def test_a_save_lands_in_the_title_folder_cemu_opens(rel: str, dest: str) -> None:
    """A save is placed under the session's title in lower case, on the account Cemu created.

    The donor's account id names nothing on this console, so it is replaced by
    `80000001`. A member with no title folder takes the session's.

    Args:
        rel: The member's path below `.import/save/`.
        dest: Where it lands, below the MLC root.
    """
    result = _preflight({f".import/save/{rel}": b"save"})

    assert result.refusals == ()
    assert [str(p.dest) for p in result.placements] == [dest]


@pytest.mark.usefixtures("save_dir")
@pytest.mark.parametrize(
    ("rel", "reason"),
    [
        ("usr/save/system/act/80000001/account.dat", "protected_destination"),
        ("usr/save/system/save/80000001.dat", "protected_destination"),
        ("usr/save/notes/readme.txt", "protected_destination"),
        ("usr/save/00050000/10143500/user/80000001/slot0.dat", "identity_mismatch"),
        ("usr/save/0005000010143500/user/80000001/slot0.dat", "identity_mismatch"),
        ("usr/save/00050002/1010ec00/user/80000001/slot0.dat", "unrecognised_layout"),
        ("usr/save/0005000E/1010ec00/user/80000001/slot0.dat", "unrecognised_layout"),
        ("usr/save/00050000/loose.bin", "unrecognised_layout"),
        ("usr/save/00050000/1010ec00", "unrecognised_layout"),
        ("usr/save/loose.bin", "unrecognised_layout"),
        ("mlc01/sys/title/00050010/1000400a/code/app.xml", "unrecognised_layout"),
        ("00050000/1010ec00/user/80000001/slot0.dat", "unrecognised_layout"),
        ("slot0.dat", "unrecognised_layout"),
        ("user/00000001/slot0.dat", "unrecognised_layout"),
        ("user/80000001", "unrecognised_layout"),
        ("usr/save/00050000/1010ec00/user/00000001/slot0.dat", "unrecognised_layout"),
        ("usr/save/00050000/1010ec00/user/80000001", "unrecognised_layout"),
        ("usr/save/00050000/1010ec00/user/common", "unrecognised_layout"),
    ],
    ids=[
        "account store",
        "play stats",
        "not a title folder",
        "another title",
        "another title, saviine",
        "a demo's high half",
        "an update's high half",
        "a loose file under the high half",
        "a title folder with no file",
        "a loose file under the save tree",
        "outside the save tree",
        "no wrapper",
        "loose file",
        "a persistent id Cemu never issues",
        "an account folder with no file",
        "an anchored persistent id Cemu never issues",
        "an anchored account folder with no file",
        "an anchored common folder with no file",
    ],
)
def test_a_member_cemu_would_not_read_is_refused(rel: str, reason: str) -> None:
    """Each shape the spec names is refused with its own code, and nothing is placed.

    Args:
        rel: The member's path below `.import/save/`.
        reason: The refusal code.
    """
    result = _preflight({f".import/save/{rel}": b"x"})

    assert [r.reason for r in result.refusals] == [reason]
    assert result.placements == ()


@pytest.mark.usefixtures("save_dir")
def test_an_anchorless_save_needs_a_game_to_sit_under() -> None:
    """With no title folder in the path and no game named by RomM, there is nowhere to place it."""
    result = _preflight({".import/save/user/80000001/slot0.dat": b"x"}, rom=None)

    assert [r.reason for r in result.refusals] == ["identity_unknown"]


@pytest.mark.usefixtures("save_dir")
def test_an_anchored_save_is_placed_when_romm_names_no_game() -> None:
    """A path that names its own title is held to the session's game only when there is one."""
    result = _preflight({".import/save/usr/save/00050000/1010ec00/user/80000001/slot0.dat": b"x"}, rom=None)

    assert result.refusals == ()
    assert [str(p.dest) for p in result.placements] == [f"{_TITLE_DIR}/user/80000001/slot0.dat"]


@pytest.mark.usefixtures("save_dir")
def test_two_donor_accounts_are_refused_rather_than_merged() -> None:
    """Cemu has one account here, so two donor accounts would land on top of each other.

    The `common` file names no account, so it is not part of the clash.
    """
    result = _preflight(
        {
            ".import/save/usr/save/00050000/1010ec00/user/80000001/a.dat": b"a",
            ".import/save/usr/save/00050000/1010ec00/user/80000002/b.dat": b"b",
            ".import/save/usr/save/00050000/1010ec00/user/common/c.dat": b"c",
        }
    )

    assert [(r.reason, r.member) for r in result.refusals] == [
        ("destination_conflict", ".import/save/usr/save/00050000/1010ec00/user/80000001/a.dat"),
        ("destination_conflict", ".import/save/usr/save/00050000/1010ec00/user/80000002/b.dat"),
    ]


@pytest.mark.usefixtures("save_dir")
def test_one_donor_account_across_many_files_is_not_a_clash() -> None:
    """Two files under one donor account both land under the session's account."""
    result = _preflight(
        {
            ".import/save/user/80000004/a.dat": b"a",
            ".import/save/user/80000004/b.dat": b"b",
        }
    )

    assert result.refusals == ()
    assert sorted(str(p.dest) for p in result.placements) == [
        f"{_TITLE_DIR}/user/80000001/a.dat",
        f"{_TITLE_DIR}/user/80000001/b.dat",
    ]


@pytest.mark.usefixtures("save_dir")
def test_an_imported_save_beside_an_archived_one_is_refused() -> None:
    """A save that lands on a file the archive already holds is a destination conflict."""
    result = _preflight(
        {".import/save/user/80000004/slot0.dat": b"imported"},
        v1={f"{_TITLE_DIR}/user/80000001/slot0.dat": b"archived"},
    )

    assert [r.reason for r in result.refusals] == ["destination_conflict"]


@pytest.mark.usefixtures("save_dir")
def test_a_state_or_memory_card_is_not_taken() -> None:
    """Cemu has no states and no cards, so the kind gate stops them before the hook."""
    result = _preflight({".import/state/game.sav": b"x", ".import/memcard/card.mcd": b"x"})

    assert sorted(r.reason for r in result.refusals) == ["kind_not_accepted", "kind_not_accepted"]


def test_an_imported_save_is_where_cemus_own_lookups_find_it(save_dir: Path) -> None:
    """The placed file is the one the exit restamp walks and the stale clear removes, in lower case.

    The read-back is by literal lower-case path: on a case-sensitive
    filesystem, a destination Cemu never opens would be written and no
    refusal could catch it.

    Args:
        save_dir: The patched save tree.
    """
    emu = cemu.Cemu()
    body = import_zip(
        {
            ".import/save/usr/save/00050000/1010EC00/user/80000009/slot0.dat": b"progress",
            ".import/save/usr/save/00050000/1010EC00/meta/meta.xml": b"<meta/>",
        }
    )
    result = preflight_import(emu, body, rom_file=None, rom=_ROMM)
    restore_import(emu, body, result)

    title = save_dir / "00050000" / "1010ec00"
    assert (title / "user" / "80000001" / "slot0.dat").read_bytes() == b"progress"
    emu._session_start = time.time() - 100
    assert emu._modified_title_saves() == [title]

    _touch(save_dir / "system" / "act" / "80000001" / "account.dat")
    emu.clear_working_slot()

    assert not title.exists()
    assert (save_dir / "system" / "act" / "80000001" / "account.dat").exists()


@pytest.mark.usefixtures("save_dir")
def test_cemu_declares_a_save_kind_only() -> None:
    """The spec names the save kind alone, with no state channel, and protects the account store."""
    emu = cemu.Cemu()
    spec = emu.import_spec()

    assert [k.kind for k in spec.kinds] == ["save"]
    assert spec.state_channel == "none"
    assert spec.protected == ("usr/save/system/*",)
    assert spec.case_insensitive_dest is False
    assert cemu.DEFAULT_PERSISTENT_ID == "80000001"
    assert emu.identity_source() == imports.IdentitySource("hex8")


def test_the_donor_accounts_are_collected_once_per_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """The archive is read for donor accounts on the first ask, and the answer is reused.

    Args:
        monkeypatch: Pytest's attribute patcher.
    """
    calls: list[str] = []
    real_split = cemu._split

    def counting_split(member: imports.ImportMember) -> Union[cemu._Split, imports.ImportRefusal]:
        """Count each member the hook reads, then split it as usual.

        Args:
            member: The member.

        Returns:
            What `_split` returns.
        """
        calls.append(member.name)
        return real_split(member)

    monkeypatch.setattr(cemu, "_split", counting_split)
    members = tuple(
        _save_member(rel)
        for rel in ("user/80000001/a.dat", "user/80000001/b.dat", "user/common/c.dat", "user/80000002/d.dat")
    )
    ctx = imports.ImportCtx(
        rom_file=None, rom=None, memory_card_synced=False, excluded=(), resume_slot=None, members=members
    )

    first = cemu._donor_persistent_ids(ctx)
    second = cemu._donor_persistent_ids(ctx)

    assert first == frozenset({"80000001", "80000002"})
    assert second is first
    assert len(calls) == len(members)


@pytest.mark.usefixtures("save_dir")
def test_a_file_that_names_no_account_never_asks_who_the_donors_are(monkeypatch: pytest.MonkeyPatch) -> None:
    """The archive is not scanned for accounts until a member that names one needs the answer.

    Args:
        monkeypatch: Pytest's attribute patcher.
    """
    asked: list[str] = []

    def record_scan(ctx: imports.ImportCtx) -> frozenset[str]:
        """Record that the scan ran.

        Args:
            ctx: The launch context.

        Returns:
            No accounts.
        """
        asked.append("scan")
        return frozenset()

    monkeypatch.setattr(cemu, "_donor_persistent_ids", record_scan)

    result = _preflight(
        {
            ".import/save/usr/save/00050000/1010ec00/user/common/c.dat": b"c",
            ".import/save/usr/save/00050000/1010ec00/meta/meta.xml": b"m",
        }
    )

    assert result.refusals == ()
    assert asked == []


def test_the_persistent_id_is_read_only_from_an_account_folder() -> None:
    """Only `user/<8xxxxxxx>/...` names an account; `common`, `meta` and a bare `user` folder do not."""

    def split(tail: tuple[str, ...]) -> cemu._Split:
        """Build a split path below a title folder.

        Args:
            tail: The components below the title's folder.

        Returns:
            The split path.
        """
        return cemu._Split("00050000", "1010ec00", tail)

    assert split(("user", "8000000a", "x.dat")).persistent_id == "8000000A"
    assert split(("user", "common", "x.dat")).persistent_id is None
    assert split(("meta", "meta.xml")).persistent_id is None
    assert split(("user",)).persistent_id is None
