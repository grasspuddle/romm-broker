"""Declared imports: the pure pieces of `webstation_broker.imports`.

Each test drives one helper directly. The activate wiring is covered in
test_api.py, and the per-emulator hooks in test_emulators.py.
"""

import dataclasses
import io
import json
import re
import zipfile
from pathlib import PurePosixPath
from typing import Any, Optional

import pytest

from webstation_broker import imports, saves


def _zip(members: dict[str, bytes], manifest: Optional[Any] = None) -> bytes:
    """Build an in-memory zip, with a manifest when one is given.

    Args:
        members: Archive member names mapped to their bytes.
        manifest: JSON-serialisable manifest to add, or None for none.

    Returns:
        The zip file contents.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
        if manifest is not None:
            zf.writestr(saves.MANIFEST_NAME, json.dumps(manifest))
    return buf.getvalue()


def test_reasons_is_the_closed_set_of_seventeen() -> None:
    """The refusal codes are closed, and an unknown one cannot be built."""
    assert len(imports.REASONS) == 17
    with pytest.raises(ValueError, match="unknown import refusal reason"):
        imports.ImportRefusal("made_up", None, None)


def test_a_refusal_dict_carries_its_docs_anchor() -> None:
    """`docs` is the site-relative anchor, with underscores as hyphens."""
    body = imports.ImportRefusal("identity_mismatch", ".import/state/x", "a state").as_dict()

    assert body == {
        "reason": "identity_mismatch",
        "member": ".import/state/x",
        "expected": "a state",
        "detail": None,
        "suggest_emulator": None,
        "docs": "/docs/api/imports#identity-mismatch",
    }


def test_refusal_body_sorts_and_caps() -> None:
    """Refusals sort by (member, reason), archive-level first, and cap at 200."""
    many = [imports.ImportRefusal("unsafe_path", f".import/save/{i:03d}", None) for i in range(205)]
    many.append(imports.ImportRefusal("too_large", None, None))

    body = imports.refusal_body(many)

    assert body["error"] == "import_refused"
    assert len(body["refusals"]) == imports.REFUSAL_CAP
    assert body["truncated"] == 6
    assert body["refusals"][0]["reason"] == "too_large"
    assert body["refusals"][1]["member"] == ".import/save/000"


def test_fold_v1_problems_maps_each_kind() -> None:
    """Legacy v1 problems fold into refusal codes, keeping the legacy text as detail."""
    plan = saves.V1Plan(
        (),
        0,
        (
            ("../a", "archive member escapes save dir: ../a", "escapes"),
            ("s/b", "archive member resolves outside save dir: s/b", "symlink"),
            ("saves", "archive member names a save subtree: saves", "names_subtree"),
            ("x/c", "archive member outside save subtrees: x/c", "outside"),
        ),
    )

    folded = imports.fold_v1_problems("archive exceeds size limit when extracted", plan)

    assert [(r.reason, r.member) for r in folded] == [
        ("too_large", None),
        ("unsafe_path", "../a"),
        ("unsafe_path", "s/b"),
        ("unrecognised_layout", "saves"),
        ("unrecognised_layout", "x/c"),
    ]
    assert folded[1].detail == "archive member escapes save dir: ../a"


def test_import_spec_as_dict_is_the_discovery_shape() -> None:
    """The spec serialises to the discovery route's `kinds`, `state_channel` and `card_subtree`."""
    spec = imports.ImportSpec(
        kinds=(imports.KindSpec("state", ("<game>.sNN",), requires_resume_slot=True, max_members=1),),
        state_channel="archive",
        card_subtree="GC",
    )

    assert spec.kind("state") is spec.kinds[0]
    assert spec.kind("save") is None
    assert spec.as_dict() == {
        "kinds": [
            {
                "kind": "state",
                "shapes": ["<game>.sNN"],
                "requires_resume_slot": True,
                "max_members": 1,
            }
        ],
        "state_channel": "archive",
        "card_subtree": "GC",
    }


def test_member_head_reads_at_most_the_cap() -> None:
    """`head` never reads past `HEAD_MAX_BYTES`, however much is asked for."""
    body = _zip({".import/save/big": b"x" * (imports.HEAD_MAX_BYTES + 10)})
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        info = zf.getinfo(".import/save/big")
        member = imports.ImportMember(
            ".import/save/big", "save", "unknown", PurePosixPath("big"), ("big",),
            info.file_size, info, zf,
        )
        assert member.head(10) == b"x" * 10
        assert len(member.head(10**9)) == imports.HEAD_MAX_BYTES


def test_rom_ref_copies_the_body_fields() -> None:
    """`RomRef.from_body` copies what imports needs, and nothing ties it to the api module."""
    from webstation_broker.api import RomIn

    ref = imports.RomRef.from_body(
        RomIn(id=4, name="G", platform="ps2", path="/r", title_id="SLUS-20001")
    )

    assert ref == imports.RomRef(4, "G", "ps2", "SLUS-20001", None, None)


# ── manifest v2 ────────────────────────────────────────────────────────


def _v2(*files: dict[str, Any]) -> dict[str, Any]:
    """Build a version 2 manifest.

    Args:
        *files: The `files` entries.

    Returns:
        The manifest.
    """
    return {"version": 2, "created_at": 0, "files": list(files)}


def test_manifest_v2_declares_each_member() -> None:
    """Each `.import/` entry maps its member to a kind and origin; v1 entries are ignored."""
    entries, refusals = imports.parse_manifest_v2(
        _v2(
            {"path": "saves/a.srm", "kind": "save"},
            {"path": ".import/save/a.srm", "kind": "save", "origin": "standalone"},
            {"path": ".import/state/b.p2s", "kind": "state", "origin": "martian"},
        ),
        [".import/save/a.srm", ".import/state/b.p2s"],
    )

    assert refusals == []
    assert entries[".import/save/a.srm"] == imports.ManifestEntry(".import/save/a.srm", "save", "standalone")
    assert entries[".import/state/b.p2s"].origin == "unknown"


def test_manifest_v2_is_skipped_without_imports() -> None:
    """An archive without imports is never checked against v2 rules."""
    assert imports.parse_manifest_v2({"version": 1}, []) == ({}, [])


@pytest.mark.parametrize(
    ("manifest", "error"),
    [
        (None, "archive has no manifest"),
        ([1, 2], None),
        ({"version": 1, "files": []}, None),
        ({"version": 2, "files": "nope"}, None),
    ],
)
def test_an_unusable_manifest_refuses_the_whole_import(manifest: Any, error: Optional[str]) -> None:
    """No usable v2 manifest gives one archive-level `manifest_invalid`.

    Args:
        manifest: The parsed manifest.
        error: The manifest error `read_archive` recorded, if any.
    """
    entries, refusals = imports.parse_manifest_v2(manifest, [".import/save/a"], error)

    assert entries == {}
    assert [(r.reason, r.member) for r in refusals] == [("manifest_invalid", None)]


@pytest.mark.parametrize(
    ("files", "names", "member"),
    [
        ([], [".import/save/a"], ".import/save/a"),
        ([{"path": ".import/save/a", "kind": "state"}], [".import/save/a"], ".import/save/a"),
        ([{"path": ".import/save/a", "kind": "bios"}], [".import/save/a"], ".import/save/a"),
        ([{"path": ".import/save/gone", "kind": "save"}], [], ".import/save/gone"),
        (
            [{"path": ".import/save/a", "kind": "save"}, {"path": ".import/save/a", "kind": "save"}],
            [".import/save/a"],
            ".import/save/a",
        ),
    ],
    ids=["undeclared", "segment-mismatch", "unknown-kind", "missing-member", "duplicate"],
)
def test_manifest_v2_refuses_a_bad_declaration_once(
    files: list[dict[str, Any]], names: list[str], member: str
) -> None:
    """Each bad declaration is one `manifest_invalid` for its member, never two.

    Args:
        files: The manifest's `files` entries.
        names: The archive's `.import/` member names.
        member: The member the refusal must name.
    """
    entries, refusals = imports.parse_manifest_v2(_v2(*files), names or [".import/save/other"])

    assert [(r.reason, r.member) for r in refusals if r.member == member] == [
        ("manifest_invalid", member)
    ]
    assert member not in entries


def test_a_non_object_entry_is_refused_at_archive_level() -> None:
    """A `files` entry that is not an object cannot name a member."""
    _, refusals = imports.parse_manifest_v2(
        _v2("x", {"path": ".import/save/a", "kind": "save"}), [".import/save/a"]
    )

    assert [(r.reason, r.member) for r in refusals] == [("manifest_invalid", None)]


@pytest.mark.parametrize(
    "extra",
    [{"origin": []}, {"origin": {}}, {}],
    ids=["list-origin", "dict-origin", "no-origin"],
)
def test_manifest_v2_reads_an_odd_or_missing_origin_as_unknown(extra: dict[str, Any]) -> None:
    """An origin that is not a known string, or no origin at all, reads as `unknown` and is not refused.

    Args:
        extra: The origin field to add to the entry, if any.
    """
    entries, refusals = imports.parse_manifest_v2(
        _v2({"path": ".import/save/a", "kind": "save", **extra}), [".import/save/a"]
    )

    assert refusals == []
    assert entries[".import/save/a"].origin == "unknown"


# ── hygiene ────────────────────────────────────────────────────────────


def _info(name: str, *, utf8: bool = True, size: int = 4) -> zipfile.ZipInfo:
    """Build a zip entry without an archive behind it.

    Args:
        name: The entry name.
        utf8: Whether the entry carries the zip UTF-8 flag.
        size: The uncompressed size to record.

    Returns:
        The entry.
    """
    info = zipfile.ZipInfo(name)
    info.file_size = size
    if utf8:
        info.flag_bits |= 0x800
    return info


def _entry(name: str, kind: str = "save") -> imports.ManifestEntry:
    """Declare `name` as `kind` with an unknown origin.

    Args:
        name: The member name.
        kind: The declared kind.

    Returns:
        The declaration.
    """
    return imports.ManifestEntry(name, kind, "unknown")


def test_normalise_member_keeps_the_users_own_path() -> None:
    """The path below `.import/<kind>/` becomes `rel`, component for component."""
    name = ".import/save/BASLUS-20001ALL/icon.sys"
    member = imports.normalise_member(_info(name), _entry(name), zf=None)

    assert isinstance(member, imports.ImportMember)
    assert member.rel == PurePosixPath("BASLUS-20001ALL/icon.sys")
    assert member.parts == ("BASLUS-20001ALL", "icon.sys")
    assert member.size == 4


@pytest.mark.parametrize(
    "tail",
    [
        "a\\b",
        "a:b",
        "a\x01b",
        "a\x7fb",
        "",
        "a//b",
        "a/./b",
        "a/../b",
        ".DS_Store",
        "dir/.hidden",
        "__MACOSX/a",
        "x" * 256,
    ],
)
def test_normalise_member_refuses_unsafe_paths(tail: str) -> None:
    """Every hygiene rule answers `unsafe_path`.

    Args:
        tail: The path below `.import/save/`.
    """
    name = f".import/save/{tail}"

    refusal = imports.normalise_member(_info(name), _entry(name), zf=None)

    assert isinstance(refusal, imports.ImportRefusal)
    assert refusal.reason == "unsafe_path"
    assert refusal.member == name


def test_normalise_member_refuses_non_ascii_without_the_utf8_flag() -> None:
    """A non-ASCII name only counts when the zip says it is UTF-8."""
    name = ".import/save/café.srm"

    flagged = imports.normalise_member(_info(name), _entry(name), zf=None)
    unflagged = imports.normalise_member(_info(name, utf8=False), _entry(name), zf=None)

    assert isinstance(flagged, imports.ImportMember)
    assert isinstance(unflagged, imports.ImportRefusal) and unflagged.reason == "unsafe_path"


def test_normalise_member_honours_a_tighter_component_limit() -> None:
    """An emulator with a short filesystem limit (xemu's 42) can lower it."""
    name = ".import/save/" + "x" * 43

    refusal = imports.normalise_member(_info(name), _entry(name), zf=None, max_component_bytes=42)

    assert isinstance(refusal, imports.ImportRefusal) and refusal.reason == "unsafe_path"


# ── the kind gate and the placement helpers ────────────────────────────


def _member(tail: str, kind: str = "save", size: int = 4, origin: str = "unknown") -> imports.ImportMember:
    """Build a hygienic member without an archive behind it.

    Args:
        tail: The path below `.import/<kind>/`.
        kind: The declared kind.
        size: The recorded size.
        origin: The declared origin.

    Returns:
        The member.
    """
    name = f".import/{kind}/{tail}"
    member = imports.normalise_member(
        _info(name, size=size), imports.ManifestEntry(name, kind, origin), zf=None
    )
    assert isinstance(member, imports.ImportMember)
    return member


def _ctx(**kwargs: Any) -> imports.ImportCtx:
    """Build a launch context with nothing set but what the test passes.

    Args:
        **kwargs: Fields to set.

    Returns:
        The context.
    """
    base: dict[str, Any] = {
        "rom_file": None,
        "rom": None,
        "memory_card_synced": False,
        "excluded": (),
        "resume_slot": None,
    }
    base.update(kwargs)
    return imports.ImportCtx(**base)


def test_gate_kind_answers_for_a_kind_the_spec_lacks() -> None:
    """No spec for the kind refuses it, naming the kinds that are taken."""
    spec = imports.ImportSpec(kinds=(imports.KindSpec("memcard", ("a card",)),))

    refusal = imports.gate_kind(_member("a.srm"), spec, _ctx())

    assert refusal is not None
    assert (refusal.reason, refusal.expected) == ("kind_not_accepted", "memcard")
    empty = imports.gate_kind(_member("a.srm"), imports.ImportSpec(), _ctx())
    assert empty is not None and empty.expected == "no imports"


def test_gate_kind_sends_states_to_the_push_route() -> None:
    """A push-channel emulator refuses archive states with `state_uses_push`."""
    refusal = imports.gate_kind(
        _member("a.p2s", kind="state"), imports.ImportSpec(state_channel="push"), _ctx()
    )

    assert refusal is not None and refusal.reason == "state_uses_push"


def test_gate_kind_requires_a_resume_slot_when_the_kind_does() -> None:
    """An archive-channel state without `resume_slot` is refused; with one it passes."""
    spec = imports.ImportSpec(
        kinds=(imports.KindSpec("state", ("s",), requires_resume_slot=True),), state_channel="archive"
    )

    refusal = imports.gate_kind(_member("a.s", kind="state"), spec, _ctx())

    assert refusal is not None and refusal.reason == "resume_slot_required"
    assert imports.gate_kind(_member("a.s", kind="state"), spec, _ctx(resume_slot=1)) is None


_SRM = re.compile(r"[^/]+\.srm", re.I)


def test_place_single_file_renames_into_the_subtree() -> None:
    """A matching single file lands under the subtree with the renamer's name."""
    dest = imports.place_single_file(
        _member("wrap/Game (USA).srm"),
        subtree="saves",
        pattern=_SRM,
        rename=lambda _: "Game.srm",
        expected="<game>.srm",
        allow_wrappers=("wrap",),
    )

    assert dest == PurePosixPath("saves/Game.srm")


@pytest.mark.parametrize(
    ("tail", "size", "reason"),
    [
        ("a/b/Game.srm", 4, "unrecognised_layout"),
        ("Game.state3", 4, "source_incompatible"),
        ("Game.state.auto", 4, "source_incompatible"),
        ("Game.sav", 4, "unrecognised_layout"),
        ("Game.srm", 0, "incomplete_unit"),
    ],
)
def test_place_single_file_refusals(tail: str, size: int, reason: str) -> None:
    """Nested, libretro-state, mismatched and empty members are each refused.

    Args:
        tail: The path below `.import/save/`.
        size: The member's size.
        reason: The expected refusal.
    """
    result = imports.place_single_file(
        _member(tail, size=size),
        subtree="saves",
        pattern=re.compile(r".+\.(srm|state\d+|state\.auto)"),
        rename=lambda n: n,
        expected="<game>.srm",
        nonempty=True,
    )

    assert isinstance(result, imports.ImportRefusal)
    assert result.reason == reason


def test_place_single_file_rechecks_the_renamed_name() -> None:
    """A renamer that produces an unsafe name is caught."""
    result = imports.place_single_file(
        _member("Game.srm"), subtree="saves", pattern=_SRM, rename=lambda _: ".hidden", expected="x"
    )

    assert isinstance(result, imports.ImportRefusal) and result.reason == "unsafe_path"


_SERIAL = re.compile(r"[A-Z]{4}\d{5}")


def test_match_anchored_strips_one_wrapper_and_matches_each_level() -> None:
    """The first matching wrapper wins, each id level must fullmatch, and a tail must remain."""
    match = imports.match_anchored(
        ("PSP", "SAVEDATA", "ULUS10064", "DATA.BIN"),
        wrappers=(("PSP", "SAVEDATA"), ("SAVEDATA",), ()),
        levels=(_SERIAL,),
    )

    assert match == imports.AnchoredMatch(("PSP", "SAVEDATA"), ("ULUS10064",), ("DATA.BIN",))
    assert imports.match_anchored(("ULUS10064",), wrappers=((),), levels=(_SERIAL,)) is None
    assert imports.match_anchored(("x", "ULUS10064", "a"), wrappers=((),), levels=(_SERIAL,)) is None


def test_match_anchored_never_falls_through_to_a_later_wrapper() -> None:
    """Once a wrapper matches, a failed id level is a miss, not a retry without it."""
    assert (
        imports.match_anchored(
            ("SAVEDATA", "SAVEDATA", "x"), wrappers=(("SAVEDATA",), ()), levels=(_SERIAL,)
        )
        is None
    )


def test_build_dest_joins_or_refuses() -> None:
    """Rewritten ids and the tail join under the subtree, and unsafe ids are refused."""
    member = _member("x/y")

    assert imports.build_dest(
        "SAVEDATA", ("ULUS10064",), ("DATA.BIN",), member=member, expected="e"
    ) == PurePosixPath("SAVEDATA/ULUS10064/DATA.BIN")
    refused = imports.build_dest("SAVEDATA", ("..",), ("a",), member=member, expected="e")
    assert isinstance(refused, imports.ImportRefusal) and refused.reason == "unsafe_path"


def test_place_single_file_refuses_when_the_renamer_rejects_the_name() -> None:
    """A renamer returning None is an `unrecognised_layout`, not a crash."""
    result = imports.place_single_file(
        _member("Game.srm"), subtree="saves", pattern=_SRM, rename=lambda _: None, expected="x"
    )

    assert isinstance(result, imports.ImportRefusal)
    assert (result.reason, result.detail) == ("unrecognised_layout", "name not recognised by the emulator")


@pytest.mark.parametrize("renamed", ["a/b", "/etc/passwd"])
def test_place_single_file_refuses_a_renamed_path(renamed: str) -> None:
    """A renamer that returns a path, relative or absolute, cannot escape the subtree.

    Args:
        renamed: What the renamer returns.
    """
    result = imports.place_single_file(
        _member("Game.srm"), subtree="saves", pattern=_SRM, rename=lambda _: renamed, expected="x"
    )

    assert isinstance(result, imports.ImportRefusal) and result.reason == "unsafe_path"


def test_place_single_file_refuses_a_trailing_newline() -> None:
    """`fullmatch` rejects a name the pattern only matches up to a trailing newline.

    Hygiene already refuses the control character, so the member is built past
    it: this pins the placement check on its own.
    """
    member = dataclasses.replace(_member("Game.srm"), parts=("Game.srm\n",))

    result = imports.place_single_file(
        member, subtree="saves", pattern=_SRM, rename=lambda n: n, expected="x"
    )

    assert isinstance(result, imports.ImportRefusal) and result.reason == "unrecognised_layout"


def test_place_single_file_refuses_a_member_that_is_only_the_wrapper() -> None:
    """A file named like the wrapper is not stripped to nothing, and is refused."""
    result = imports.place_single_file(
        _member("wrap"),
        subtree="saves",
        pattern=_SRM,
        rename=lambda n: n,
        expected="x",
        allow_wrappers=("wrap",),
    )

    assert isinstance(result, imports.ImportRefusal) and result.reason == "unrecognised_layout"


@pytest.mark.parametrize("tail", ["a/b", ""])
def test_build_dest_refuses_an_unsafe_tail_component(tail: str) -> None:
    """A tail component holding a slash, or empty, is `unsafe_path`.

    Args:
        tail: The tail component.
    """
    refused = imports.build_dest("SAVEDATA", ("ULUS10064",), (tail,), member=_member("x/y"), expected="e")

    assert isinstance(refused, imports.ImportRefusal) and refused.reason == "unsafe_path"
