"""Placement rules for a Wii NAND title save, shared by Dolphin and RetroArch's Dolphin core.

Both keep the emulated NAND as an ordinary tree: `title/<high>/<low>/data/...`
holds a title's save, and `sys`, `ticket`, a system title and a title's
`content` are the console's own. Where the tree lives differs (`Wii` in
Dolphin's user folder, `saves/dolphin-emu/User/Wii` under RetroArch), so the
caller names it. The functions here take no emulator, so they stay pure.
"""

import re
from typing import Union

from .. import imports

SAVE_SHAPE = "title/<00010000|00010001|00010004>/<title id>/data/..."
"""The save shape a `save` KindSpec advertises for a Wii session."""

NAND_EXPECTED = f"a Wii save, {SAVE_SHAPE}"
"""What a Wii `save` member must look like, for refusals."""

NAND_TITLE_RE = re.compile(r"title", re.ASCII)
"""The NAND folder every installed title's data sits under."""

NAND_HIGH_RE = re.compile(r"0001000[014]", re.ASCII)
"""A title id's high half for a title with player saves: a disc, a channel, or a disc with a channel."""

NAND_LOW_RE = re.compile(r"[0-9A-Fa-f]{8}", re.ASCII)
"""A title id's low half: the four-character game code, in hex."""


def unwrap(parts: tuple[str, ...], wrappers: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """Strip the first wrapper a member path arrived under.

    A wrapper is only stripped when something is left below it, so a file
    named like a wrapper folder is not emptied away.

    Args:
        parts: The member's components.
        wrappers: Candidate leading folders, longest first.

    Returns:
        The components below the wrapper, or `parts` when none matched.
    """
    for wrapper in wrappers:
        if len(parts) > len(wrapper) and parts[: len(wrapper)] == wrapper:
            return parts[len(wrapper):]
    return parts


def is_nand_system(rest: tuple[str, ...]) -> bool:
    """Tell whether a NAND path is system or install data rather than a title's save.

    Such a path is placed as named, so the plan check refuses it against the
    emulator's protected globs as emulator configuration, which tells the
    player more than a layout refusal would.

    Args:
        rest: The path's components below the NAND root.

    Returns:
        True under `sys` or `ticket`, a system title (`title/00000001`), or a title's `content`.
    """
    if rest[0] in ("sys", "ticket") and len(rest) > 1:
        return True
    if rest[:2] == ("title", "00000001") and len(rest) > 2:
        return True
    return (
        rest[0] == "title"
        and len(rest) > 4
        and rest[3] == "content"
        and all(NAND_LOW_RE.fullmatch(p) for p in rest[1:3])
    )


def nand_leaf_refusal(
    member: imports.ImportMember, expected: str = NAND_EXPECTED
) -> imports.ImportRefusal:
    """Refuse a Wii member that is not under a title's `data` folder, naming what it looks like.

    Args:
        member: The member.
        expected: The accepted shape, in words.

    Returns:
        A conversion refusal for an SD-card export, a source refusal for a
        whole NAND dump, and a layout refusal for anything else.
    """
    leaf = member.parts[-1].lower()
    if leaf == "data.bin":
        return imports.ImportRefusal(
            "needs_conversion",
            member.name,
            expected,
            detail=(
                "a Wii SD-card export; import it with Dolphin's Import Wii Save, then send the title folder"
            ),
        )
    if leaf == "nand.bin":
        return imports.ImportRefusal(
            "source_incompatible",
            member.name,
            expected,
            detail="a whole NAND dump; send the title folder from it instead",
        )
    if leaf.endswith(".gci"):
        return imports.ImportRefusal(
            "unrecognised_layout",
            member.name,
            expected,
            detail="a GameCube save; a Wii session takes NAND title saves",
        )
    return imports.ImportRefusal("unrecognised_layout", member.name, expected)


def place_nand(
    member: imports.ImportMember,
    session: imports.SessionIdentity,
    *,
    subtree: str,
    wrappers: tuple[tuple[str, ...], ...],
    expected: str = NAND_EXPECTED,
) -> Union[imports.Placement, imports.ImportRefusal]:
    """Place a file of a Wii title's save under `<subtree>/title/<high>/<low>/data`.

    The path is found below at most one wrapper. A title reads its save from
    the folder named for its own id, so the title is held strictly to the
    session's game. System and install data is placed as named, for the plan
    check to refuse as protected.

    Args:
        member: The member.
        session: The session's identity.
        subtree: Where the NAND tree lives, relative to the save root.
        wrappers: The leading folders a NAND path may arrive under, longest first.
        expected: The accepted shape, in words.

    Returns:
        The placement, or a refusal.
    """
    rest = unwrap(member.parts, wrappers)
    if is_nand_system(rest):
        dest = imports.build_dest(subtree, (), rest, member=member, expected=expected)
        if isinstance(dest, imports.ImportRefusal):
            return dest
        return imports.Placement(member, dest)
    found = imports.match_anchored(
        rest, wrappers=((),), levels=(NAND_TITLE_RE, NAND_HIGH_RE, NAND_LOW_RE), min_tail=2
    )
    if found is None or found.tail[0] != "data":
        return nand_leaf_refusal(member, expected)
    refusal = imports.check_member_identity(
        member,
        imports.NORMALISERS["gc_wii_disc"](found.ids[2]),
        session,
        family="gc_wii_disc",
        policy="strict",
        expected=expected,
    )
    if refusal is not None:
        return refusal
    # Dolphin writes the NAND's id folders in lower case.
    dest = imports.build_dest(
        subtree,
        ("title", found.ids[1], found.ids[2].lower()),
        found.tail,
        member=member,
        expected=expected,
    )
    if isinstance(dest, imports.ImportRefusal):
        return dest
    return imports.Placement(member, dest)
