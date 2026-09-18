"""Declared imports: the pure pieces of `webstation_broker.imports`.

Each test drives one helper directly. The activate wiring is covered in
test_api.py, and the per-emulator hooks in test_emulators.py.
"""

import io
import json
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
