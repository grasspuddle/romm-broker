"""Tests for the Wii NAND placement rules Dolphin and RetroArch's Dolphin core share."""

import zipfile
from pathlib import PurePosixPath
from typing import Optional, Union

import pytest

from webstation_broker import imports
from webstation_broker.emulators import wii_nand

_RMCE = imports.SessionIdentity("524D4345", "romm")
"""The session of a Wii game whose code is `RMCE`."""

_NONE = imports.SessionIdentity(None, "none")
"""A session nobody has named a game for."""

_WRAPPERS: tuple[tuple[str, ...], ...] = (("saves", "dolphin-emu", "User", "Wii"), ("Wii",))
"""The leading folders a NAND path may arrive under."""

_TITLE = "title/00010000/524D4345"
"""A Wii title folder for `RMCE`."""


def _member(tail: str) -> imports.ImportMember:
    """Build a hygienic `save` member without an archive behind it.

    Args:
        tail: The path below `.import/save/`.

    Returns:
        The member.
    """
    name = f".import/save/{tail}"
    info = zipfile.ZipInfo(name)
    info.file_size = 4
    info.flag_bits |= 0x800
    member = imports.normalise_member(info, imports.ManifestEntry(name, "save", "unknown"), zf=None)
    assert isinstance(member, imports.ImportMember)
    return member


def _place(
    tail: str, session: imports.SessionIdentity = _RMCE, subtree: str = "Wii"
) -> Union[imports.Placement, imports.ImportRefusal]:
    """Place one member with the Dolphin wrappers.

    Args:
        tail: The path below `.import/save/`.
        session: The session's identity.
        subtree: The folder the NAND tree lives under.

    Returns:
        The placement, or a refusal.
    """
    return wii_nand.place_nand(_member(tail), session, subtree=subtree, wrappers=_WRAPPERS)


@pytest.mark.parametrize(
    ("parts", "expected"),
    [
        (("Wii", "title", "x"), ("title", "x")),
        (("saves", "dolphin-emu", "User", "Wii", "title", "x"), ("title", "x")),
        (("title", "x"), ("title", "x")),
        (("Wii",), ("Wii",)),
    ],
)
def test_unwrap_strips_one_wrapper_and_never_empties_a_path(
    parts: tuple[str, ...], expected: tuple[str, ...]
) -> None:
    """A wrapper is stripped only when something is left below it.

    Args:
        parts: The member's components.
        expected: What is left.
    """
    assert wii_nand.unwrap(parts, _WRAPPERS) == expected


@pytest.mark.parametrize(
    ("rest", "system"),
    [
        (("sys", "uid.sys"), True),
        (("sys",), False),
        (("ticket", "00010000", "x.tik"), True),
        (("title", "00000001", "00000002", "content", "a.app"), True),
        (("title", "00000001"), False),
        (("title", "00010000", "524D4345", "content", "a.app"), True),
        (("title", "00010000", "524D4345", "data", "a.bin"), False),
        (("title", "00010000", "not-hex", "content", "a.app"), False),
        (("shared2", "a"), False),
    ],
)
def test_is_nand_system_picks_out_install_and_system_data(rest: tuple[str, ...], system: bool) -> None:
    """System and install data is told apart from a title's own save.

    Args:
        rest: The path's components below `Wii`.
        system: Whether it is system or install data.
    """
    assert wii_nand.is_nand_system(rest) is system


@pytest.mark.parametrize(
    ("tail", "subtree", "dest"),
    [
        (f"{_TITLE}/data/banner.bin", "Wii", "Wii/title/00010000/524d4345/data/banner.bin"),
        (f"Wii/{_TITLE}/data/banner.bin", "Wii", "Wii/title/00010000/524d4345/data/banner.bin"),
        (
            f"saves/dolphin-emu/User/Wii/{_TITLE}/data/a/b.bin",
            "saves/dolphin-emu/User/Wii",
            "saves/dolphin-emu/User/Wii/title/00010000/524d4345/data/a/b.bin",
        ),
    ],
)
def test_a_title_save_lands_under_the_subtree_with_lower_case_ids(tail: str, subtree: str, dest: str) -> None:
    """The destination is the subtree, then the NAND id folders in lower case, then the save.

    Args:
        tail: The member's path.
        subtree: The folder the NAND tree lives under.
        dest: The destination.
    """
    placed = _place(tail, subtree=subtree)

    assert isinstance(placed, imports.Placement)
    assert placed.dest == PurePosixPath(dest)


def test_a_title_for_another_game_is_refused_strictly() -> None:
    """A title reads its save from the folder named for its own id."""
    refused = _place("title/00010000/524D4346/data/a.bin")

    assert isinstance(refused, imports.ImportRefusal)
    assert refused.reason == "identity_mismatch"


def test_a_session_with_no_id_takes_any_title() -> None:
    """With nobody naming the game, the strict policy has nothing to compare."""
    assert isinstance(_place(f"{_TITLE}/data/a.bin", _NONE), imports.Placement)


def test_system_and_install_data_is_placed_as_named() -> None:
    """The plan check refuses it against the protected globs, which reads better than a layout refusal."""
    sys_file = _place("sys/uid.sys")
    content = _place(f"{_TITLE}/content/a.app")

    assert isinstance(sys_file, imports.Placement) and sys_file.dest == PurePosixPath("Wii/sys/uid.sys")
    assert isinstance(content, imports.Placement)
    assert content.dest == PurePosixPath(f"Wii/{_TITLE}/content/a.app")


@pytest.mark.parametrize(
    ("tail", "reason", "detail"),
    [
        ("data.bin", "needs_conversion", "a Wii SD-card export"),
        ("nand.bin", "source_incompatible", "a whole NAND dump"),
        ("Card A/save.gci", "unrecognised_layout", "a GameCube save"),
        ("junk.txt", "unrecognised_layout", None),
        (f"{_TITLE}/banner.bin", "unrecognised_layout", None),
        ("title/00010002/524D4345/data/a.bin", "unrecognised_layout", None),
    ],
)
def test_a_member_that_is_not_a_title_save_is_refused_with_what_it_looks_like(
    tail: str, reason: str, detail: Optional[str]
) -> None:
    """Each near miss gets the refusal that says what to send instead.

    Args:
        tail: The member's path.
        reason: The refusal code.
        detail: A fragment the detail opens with, or None when it carries none.
    """
    refused = _place(tail)

    assert isinstance(refused, imports.ImportRefusal)
    assert refused.reason == reason
    assert refused.expected == wii_nand.NAND_EXPECTED
    if detail is None:
        assert refused.detail is None
    else:
        assert refused.detail is not None and refused.detail.startswith(detail)
