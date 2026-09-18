"""Declared imports: save data RomM names, placed by each emulator's own rules.

A restored dump puts every member back at its own path. An import cannot
work that way: a player's save from another emulator, a card from real
hardware or an EmulatorJS export has no path this broker wrote. So RomM
declares what each import member is (a save, a state or a memory card),
under the `.import/<kind>/` prefix and in a version 2 manifest, and the
emulator either places it or refuses it with one of `REASONS`.

Everything here runs in preflight, before the working slot is cleared, so a
refusal always leaves the player's slot as it was. The emulator side is four
hooks on `Emulator` (`import_spec`, `place_import`, `validate_import_plan`
and `identity_source`); this module holds the shared machinery they lean on.
"""

import logging
import zipfile
from collections.abc import Hashable, Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Optional

from . import saves

if TYPE_CHECKING:
    from .api import RomIn

log = logging.getLogger(__name__)

ImportKind = Literal["save", "state", "memcard"]
"""What RomM declares an import member to be."""
Origin = Literal["emulatorjs", "standalone", "hardware", "unknown"]
"""Where RomM says a member came from. Advisory: placement never reads it."""
StateChannel = Literal["archive", "push", "none"]
"""How an emulator takes states: in the archive, pushed after activate, or not at all."""
IdFamily = Literal[
    "ps_serial_dashed",
    "ps2_card_dir",
    "ps_serial_nodash",
    "hex8",
    "xbox",
    "gc_wii_disc",
    "hex16",
    "dc_product",
    "scummvm_target",
]
"""The game-id notations `NORMALISERS` can bring to one canonical form."""

REASONS: frozenset[str] = frozenset(
    {
        "manifest_invalid",
        "unsafe_path",
        "kind_not_accepted",
        "state_uses_push",
        "resume_slot_required",
        "memcard_synced_separately",
        "source_incompatible",
        "needs_conversion",
        "unrecognised_layout",
        "shape_unverified",
        "incomplete_unit",
        "destination_unresolvable",
        "identity_unknown",
        "identity_mismatch",
        "protected_destination",
        "destination_conflict",
        "too_large",
    }
)
"""Every refusal code. The set is closed: RomM branches on these."""
REFUSAL_CAP = 200
"""Most refusals one 422 lists; the rest are counted in `truncated`."""
HEAD_MAX_BYTES = 64 * 1024
"""Most bytes `ImportMember.head` reads, however many are asked for."""


@dataclass(frozen=True)
class ImportRefusal:
    """Why one member (or the whole import) cannot be placed.

    Attributes:
        reason: One of `REASONS`.
        member: The member's zip name, or None for an archive-level refusal.
        expected: What would have been accepted, in words, or None.
        detail: Specifics for this member, or None.
        suggest_emulator: An emulator that would take the member, or None.
    """

    reason: str
    member: Optional[str]
    expected: Optional[str]
    detail: Optional[str] = None
    suggest_emulator: Optional[str] = None

    def __post_init__(self) -> None:
        """Refuse a reason outside the closed set.

        Raises:
            ValueError: When `reason` is not in `REASONS`.
        """
        if self.reason not in REASONS:
            raise ValueError(f"unknown import refusal reason: {self.reason}")

    def as_dict(self) -> dict[str, Optional[str]]:
        """The refusal as the 422 lists it.

        Returns:
            The fields, plus `docs`: the site-relative anchor for the reason.
        """
        return {
            "reason": self.reason,
            "member": self.member,
            "expected": self.expected,
            "detail": self.detail,
            "suggest_emulator": self.suggest_emulator,
            "docs": f"/docs/api/imports#{self.reason.replace('_', '-')}",
        }


@dataclass(frozen=True)
class RomRef:
    """The broker's copy of the activate body's rom, so this module never imports `api`.

    Attributes:
        id: RomM's id for the rom.
        name: The rom's display name.
        platform: The platform slug.
        title_id: RomM's game id for the rom.
        save_target: RomM's name for where the game keeps its saves.
        save_target_layout: How `save_target` names that place.
    """

    id: Optional[int]
    name: Optional[str]
    platform: Optional[str]
    title_id: Optional[str] = None
    save_target: Optional[str] = None
    save_target_layout: Optional[str] = None

    @classmethod
    def from_body(cls, rom: "RomIn") -> "RomRef":
        """Copy the fields imports needs off the activate body's rom.

        Args:
            rom: The validated rom from the activate body.

        Returns:
            The copy.
        """
        return cls(
            rom.id, rom.name, rom.platform, rom.title_id, rom.save_target, rom.save_target_layout
        )


@dataclass(frozen=True)
class ImportMember:
    """One `.import/` member that passed hygiene, ready for placement.

    Attributes:
        name: The original zip name, echoed in refusals.
        kind: The declared kind.
        origin: The declared origin; advisory, placement never reads it.
        rel: The path below `.import/<kind>/`.
        parts: `rel`'s components.
        size: The uncompressed size, from the zip header.
        info: The zip entry.
    """

    name: str
    kind: ImportKind
    origin: Origin
    rel: PurePosixPath
    parts: tuple[str, ...]
    size: int
    info: zipfile.ZipInfo = field(repr=False, compare=False)
    _zf: Optional[zipfile.ZipFile] = field(default=None, repr=False, compare=False)

    def head(self, n: int) -> bytes:
        """Read the start of the member, for a hook that sniffs content.

        Only valid inside preflight, while its archive is open.

        Args:
            n: Bytes wanted; clamped to `HEAD_MAX_BYTES`.

        Returns:
            Up to `n` bytes from the start of the member.

        Raises:
            RuntimeError: When the member was built without an open archive.
        """
        if self._zf is None:
            raise RuntimeError(f"{self.name} has no open archive to read from")
        with self._zf.open(self.info) as fh:
            return fh.read(max(0, min(n, HEAD_MAX_BYTES)))


@dataclass(frozen=True)
class ImportCtx:
    """What preflight knows about the launch, handed to every placement hook.

    Attributes:
        rom_file: The resolved bootable file, or None.
        rom: The activate body's rom, or None.
        memory_card_synced: Whether the card travels on its own routes this session.
        excluded: Subtrees the restore leaves alone this session.
        resume_slot: The activate's `save.resume_slot`, or None.
        members: Every member that passed hygiene.
        archive_paths: The same zip's v1 member names, less excluded ones.
        v1_bytes: Those members' total uncompressed size.
        memo: Per-preflight cache for hooks, keyed however they like.
    """

    rom_file: Optional[Path]
    rom: Optional[RomRef]
    memory_card_synced: bool
    excluded: tuple[str, ...]
    resume_slot: Optional[int]
    members: tuple[ImportMember, ...] = ()
    archive_paths: frozenset[str] = frozenset()
    v1_bytes: int = 0
    memo: dict[Hashable, object] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class KindSpec:
    """One kind an emulator accepts.

    Attributes:
        kind: The kind.
        shapes: The accepted shapes, in words, for `expected` and discovery.
        requires_resume_slot: Whether the member only boots with `save.resume_slot` set.
        max_members: Most members of this kind one import may place, or None.
        counts_v1: Whether v1 members of the same kind count toward `max_members`.
    """

    kind: ImportKind
    shapes: tuple[str, ...]
    requires_resume_slot: bool = False
    max_members: Optional[int] = None
    counts_v1: bool = False


@dataclass(frozen=True)
class ImportSpec:
    """What an emulator accepts on one platform, and the rules its plan is checked against.

    Attributes:
        kinds: The accepted kinds; empty means no imports.
        state_channel: How the emulator takes states.
        protected: `fnmatch` globs, relative to `save_root`, no member may land on.
        case_insensitive_dest: Whether destinations collide regardless of case.
        unit_depth: Leading destination components that name one save unit.
        unit_requires: Names every unit must hold, relative to the unit.
        max_component_bytes: Longest path component the emulator's filesystem takes.
        card_subtree: The memory-card subtree, for discovery.
    """

    kinds: tuple[KindSpec, ...] = ()
    state_channel: StateChannel = "none"
    protected: tuple[str, ...] = ()
    case_insensitive_dest: bool = False
    unit_depth: int = 0
    unit_requires: frozenset[str] = frozenset()
    max_component_bytes: int = 255
    card_subtree: Optional[str] = None

    def kind(self, k: str) -> Optional[KindSpec]:
        """Look up the spec for one kind.

        Args:
            k: The kind.

        Returns:
            Its `KindSpec`, or None when the kind is not accepted.
        """
        return next((s for s in self.kinds if s.kind == k), None)

    def as_dict(self) -> dict[str, Any]:
        """The spec as the discovery route reports it.

        Returns:
            `kinds`, `state_channel` and `card_subtree`.
        """
        return {
            "kinds": [
                {
                    "kind": s.kind,
                    "shapes": list(s.shapes),
                    "requires_resume_slot": s.requires_resume_slot,
                    "max_members": s.max_members,
                }
                for s in self.kinds
            ],
            "state_channel": self.state_channel,
            "card_subtree": self.card_subtree,
        }


@dataclass(frozen=True)
class Placement:
    """Where one member lands.

    Attributes:
        member: The member.
        dest: The destination, relative to `save_root`.
        sidecars: `(destination, bytes)` files the broker writes beside it.
    """

    member: ImportMember
    dest: PurePosixPath
    sidecars: tuple[tuple[PurePosixPath, bytes], ...] = ()


@dataclass(frozen=True)
class SessionIdentity:
    """The game id this session runs as, and who supplied it.

    Attributes:
        value: The canonical id, or None when nobody supplied one.
        source: `rom` (read off the rom), `romm` (from the activate body) or `none`.
    """

    value: Optional[str]
    source: Literal["rom", "romm", "none"]


@dataclass(frozen=True)
class PreflightResult:
    """What preflight decided.

    Attributes:
        placements: Every placement; empty whenever `refusals` is not.
        refusals: Every refusal, de-duplicated.
        identity: The session's identity.
    """

    placements: tuple[Placement, ...]
    refusals: tuple[ImportRefusal, ...]
    identity: SessionIdentity


def refusal_body(refusals: Iterable[ImportRefusal]) -> dict[str, Any]:
    """Build the 422 detail for a refused import.

    Args:
        refusals: Every refusal.

    Returns:
        `{"error": "import_refused", "refusals": [...], "truncated": n}`, the
        list sorted by `(member or "", reason)` and capped at `REFUSAL_CAP`.
    """
    ordered = sorted(refusals, key=lambda r: (r.member or "", r.reason))
    return {
        "error": "import_refused",
        "refusals": [r.as_dict() for r in ordered[:REFUSAL_CAP]],
        "truncated": max(0, len(ordered) - REFUSAL_CAP),
    }


_V1_REASONS: dict[str, str] = {
    "escapes": "unsafe_path",
    "symlink": "unsafe_path",
    "names_subtree": "unrecognised_layout",
    "outside": "unrecognised_layout",
}
"""Refusal code for each `saves.V1Problem` in an archive that also holds imports."""


def fold_v1_problems(
    view_error: Optional[str], v1_plan: Optional[saves.V1Plan]
) -> list[ImportRefusal]:
    """Turn the legacy whole-archive and v1-member errors into refusals.

    A v1-only archive keeps its legacy 422 string; this is for one that also
    holds imports, where RomM expects the structured list.

    Args:
        view_error: `ArchiveView.error`, or None.
        v1_plan: The v1 plan, or None when it was not run.

    Returns:
        A `too_large` refusal for a whole-archive error, then one per v1 problem,
        each carrying the legacy message as `detail`.
    """
    out: list[ImportRefusal] = []
    if view_error:
        out.append(ImportRefusal("too_large", None, None, detail=view_error))
    for name, message, kind in v1_plan.problems if v1_plan else ():
        out.append(ImportRefusal(_V1_REASONS[kind], name, None, detail=message))
    return out
