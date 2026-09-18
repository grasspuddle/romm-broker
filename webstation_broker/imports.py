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
import re
import zipfile
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from . import saves

if TYPE_CHECKING:
    from .api import RomIn
    from .emulators.base import Emulator

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


_KINDS: tuple[str, ...] = ("save", "state", "memcard")
"""The kinds a manifest may declare."""
_ORIGINS: frozenset[str] = frozenset({"emulatorjs", "standalone", "hardware", "unknown"})
"""The origins a manifest may declare; anything else is read as `unknown`."""
_MANIFEST_EXPECTED = "a version 2 manifest declaring every .import/ member once, by its kind"
"""The `expected` text on every `manifest_invalid`."""


@dataclass(frozen=True)
class ManifestEntry:
    """One `.import/` member's declaration.

    Attributes:
        path: The member's zip name.
        kind: The declared kind, which matches the path's kind segment.
        origin: The declared origin.
    """

    path: str
    kind: ImportKind
    origin: Origin


def _manifest_invalid(member: Optional[str], detail: str) -> ImportRefusal:
    """Build a `manifest_invalid` refusal.

    Args:
        member: The member, or None for the whole manifest.
        detail: What is wrong.

    Returns:
        The refusal.
    """
    return ImportRefusal("manifest_invalid", member, _MANIFEST_EXPECTED, detail=detail)


def parse_manifest_v2(
    manifest: Any, import_names: Sequence[str], manifest_error: Optional[str] = None
) -> tuple[dict[str, ManifestEntry], list[ImportRefusal]]:
    """Match an archive's `.import/` members to their declarations.

    Entries whose path is not under `.import/` are the v1 entries RomM
    carried over and are ignored. Every import member needs exactly one
    entry whose kind matches its path segment.

    Args:
        manifest: The parsed manifest, or None.
        import_names: The archive's `.import/` member names.
        manifest_error: Why the manifest could not be read, if it could not.

    Returns:
        The valid declarations keyed by member name, and every refusal.
        Without imports, both are empty and the manifest is not checked.
    """
    if not import_names:
        return {}, []
    if manifest_error:
        return {}, [_manifest_invalid(None, manifest_error)]
    if not isinstance(manifest, dict):
        return {}, [_manifest_invalid(None, "manifest is not a JSON object")]
    if manifest.get("version") != 2:
        return {}, [_manifest_invalid(None, f"manifest version is {manifest.get('version')!r}, not 2")]
    files = manifest.get("files")
    if not isinstance(files, list):
        return {}, [_manifest_invalid(None, "manifest files is not a list")]
    if "import" in manifest:
        log.info("imports: manifest import block: %r", manifest["import"])

    present = set(import_names)
    entries: dict[str, ManifestEntry] = {}
    refusals: list[ImportRefusal] = []
    seen: set[str] = set()
    refused: set[str] = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            refusals.append(_manifest_invalid(None, f"files[{index}] is not an object"))
            continue
        path = entry.get("path")
        if not isinstance(path, str) or not path.startswith(saves.IMPORT_PREFIX):
            continue
        if path in seen:
            entries.pop(path, None)
            if path not in refused:
                refused.add(path)
                refusals.append(_manifest_invalid(path, "declared more than once"))
            continue
        seen.add(path)
        kind = entry.get("kind")
        if kind not in _KINDS:
            refused.add(path)
            refusals.append(_manifest_invalid(path, f"kind {kind!r} is not save, state or memcard"))
            continue
        parts = path.split("/")
        if len(parts) < 3 or parts[1] != kind:
            refused.add(path)
            refusals.append(
                _manifest_invalid(path, f"declared {kind} but the path is not under .import/{kind}/")
            )
            continue
        if path not in present:
            refused.add(path)
            refusals.append(_manifest_invalid(path, "declared but not in the archive"))
            continue
        origin = entry.get("origin", "unknown")
        if not isinstance(origin, str) or origin not in _ORIGINS:
            log.info("imports: %s declares unknown origin %r, reading it as unknown", path, origin)
            origin = "unknown"
        entries[path] = ManifestEntry(path, kind, origin)
    for name in import_names:
        if name not in seen:
            seen.add(name)
            refusals.append(_manifest_invalid(name, "not declared in the manifest"))
    return entries, refusals


_UTF8_FLAG = 0x800
"""Zip general-purpose flag bit saying the entry name is UTF-8."""
_SAFE_EXPECTED = "a relative path of plain names: no hidden, system or oversized components"
"""The `expected` text on every hygiene refusal."""


def _name_problem(name: str) -> Optional[str]:
    """Check a whole name for characters no save path may hold.

    Args:
        name: A member name or a single component.

    Returns:
        What is wrong, or None.
    """
    if "\\" in name:
        return "backslash in name"
    if ":" in name:
        return "colon in name"
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in name):
        return "control character in name"
    return None


def _component_problem(part: str, max_component_bytes: int) -> Optional[str]:
    """Check one path component.

    Args:
        part: The component.
        max_component_bytes: The longest component allowed, in UTF-8 bytes.

    Returns:
        What is wrong, or None.
    """
    if part in ("", ".", ".."):
        return f"empty or relative component {part!r}"
    if part.startswith("."):
        return f"hidden component {part!r}"
    if part == "__MACOSX":
        return "__MACOSX component"
    if "/" in part:
        return f"slash in component {part!r}"
    if len(part.encode("utf-8")) > max_component_bytes:
        return f"component longer than {max_component_bytes} bytes"
    return None


def normalise_member(
    info: zipfile.ZipInfo,
    entry: ManifestEntry,
    *,
    zf: Optional[zipfile.ZipFile],
    max_component_bytes: int = 255,
) -> Union[ImportMember, ImportRefusal]:
    """Apply every `unsafe_path` rule to one member, in one pass.

    This is the only hygiene check: placement hooks can rely on a member's
    parts being plain names.

    Args:
        info: The member's zip entry.
        entry: Its declaration, whose kind has already been matched to the path.
        zf: The open archive, for `ImportMember.head`; None in tests.
        max_component_bytes: The longest component the emulator's filesystem takes.

    Returns:
        The member, or an `unsafe_path` refusal.
    """
    name = info.filename

    def unsafe(detail: str) -> ImportRefusal:
        """Build this member's `unsafe_path` refusal.

        Args:
            detail: What is wrong.

        Returns:
            The refusal.
        """
        return ImportRefusal("unsafe_path", name, _SAFE_EXPECTED, detail=detail)

    if not name.isascii() and not info.flag_bits & _UTF8_FLAG:
        return unsafe("non-ASCII name without the zip UTF-8 flag")
    problem = _name_problem(name)
    if problem:
        return unsafe(problem)
    prefix = f"{saves.IMPORT_PREFIX}{entry.kind}/"
    if not name.startswith(prefix) or len(name) == len(prefix):
        return unsafe(f"nothing below {prefix}")
    tail = name[len(prefix):].split("/")
    for part in tail:
        problem = _component_problem(part, max_component_bytes)
        if problem:
            return unsafe(problem)
    return ImportMember(
        name=name,
        kind=entry.kind,
        origin=entry.origin,
        rel=PurePosixPath(*tail),
        parts=tuple(tail),
        size=info.file_size,
        info=info,
        _zf=zf,
    )


def _accepted_kinds(spec: ImportSpec) -> str:
    """Name the kinds an emulator takes, for `expected`.

    Args:
        spec: The emulator's spec.

    Returns:
        A comma-separated list, or `no imports`.
    """
    return ", ".join(s.kind for s in spec.kinds) or "no imports"


def gate_kind(member: ImportMember, spec: ImportSpec, ctx: ImportCtx) -> Optional[ImportRefusal]:
    """Refuse a member whose kind the emulator does not take, before any hook sees it.

    Args:
        member: The member.
        spec: The emulator's spec for this platform.
        ctx: The launch context.

    Returns:
        `state_uses_push`, `kind_not_accepted` or `resume_slot_required`, or
        None when the member may go on to `place_import`.
    """
    kind_spec = spec.kind(member.kind)
    if kind_spec is None:
        if member.kind == "state" and spec.state_channel == "push":
            return ImportRefusal(
                "state_uses_push",
                member.name,
                "a state PUT to /api/session/state-file after activate, with resume_slot set",
            )
        return ImportRefusal("kind_not_accepted", member.name, _accepted_kinds(spec))
    if kind_spec.requires_resume_slot and ctx.resume_slot is None:
        return ImportRefusal(
            "resume_slot_required",
            member.name,
            "save.resume_slot set on the activate",
            detail="without it the state is never loaded, and the exit dump overwrites it",
        )
    return None


LIBRETRO_STATE_RE = re.compile(r"^.+\.state(\d+|\.auto)$", re.I)
"""A RetroArch numbered or auto state name (`.state3`, `.state.auto`), which no standalone loads.

A bare `.state` is not matched: flycast takes that name as its own.
"""


def place_single_file(
    member: ImportMember,
    *,
    subtree: str,
    pattern: re.Pattern[str],
    rename: Callable[[str], Optional[str]],
    expected: str,
    allow_wrappers: tuple[str, ...] = (),
    nonempty: bool = False,
    refuse_libretro_states: bool = True,
    max_component_bytes: int = 255,
) -> Union[PurePosixPath, ImportRefusal]:
    """Place a member that is one file in one directory.

    Args:
        member: The member.
        subtree: The directory it lands in, relative to `save_root`.
        pattern: What the file's name must `fullmatch`.
        rename: The emulator's own pure renamer, called with pre-launch inputs only.
            It returns None for a name it does not recognise.
        expected: The accepted shape, in words.
        allow_wrappers: Leading folders (slash-joined) to strip, at most one.
        nonempty: Whether an empty file is an `incomplete_unit`.
        refuse_libretro_states: Whether a RetroArch state name is `source_incompatible`.
        max_component_bytes: Longest name the destination filesystem takes.

    Returns:
        The destination, or a refusal.
    """
    parts = member.parts
    for wrapper in allow_wrappers:
        head = tuple(wrapper.split("/"))
        if parts[: len(head)] == head and len(parts) > len(head):
            parts = parts[len(head):]
            break
    if len(parts) != 1:
        return ImportRefusal("unrecognised_layout", member.name, expected, detail="expected a single file")
    leaf = parts[0]
    if refuse_libretro_states and LIBRETRO_STATE_RE.fullmatch(leaf):
        return ImportRefusal(
            "source_incompatible", member.name, expected, detail="a RetroArch (libretro) state"
        )
    if not pattern.fullmatch(leaf):
        return ImportRefusal("unrecognised_layout", member.name, expected)
    if nonempty and member.size == 0:
        return ImportRefusal("incomplete_unit", member.name, expected, detail="the file is empty")
    new = rename(leaf)
    if new is None:
        return ImportRefusal(
            "unrecognised_layout", member.name, expected, detail="name not recognised by the emulator"
        )
    problem = _name_problem(new) or _component_problem(new, max_component_bytes)
    if problem:
        return ImportRefusal("unsafe_path", member.name, expected, detail=f"renamed to {new!r}: {problem}")
    return PurePosixPath(subtree) / new


@dataclass(frozen=True)
class AnchoredMatch:
    """A member path split at its id levels.

    Attributes:
        wrapper: The leading folders that were stripped.
        ids: One component per id level.
        tail: Everything below the ids.
    """

    wrapper: tuple[str, ...]
    ids: tuple[str, ...]
    tail: tuple[str, ...]


def match_anchored(
    parts: Sequence[str],
    *,
    wrappers: Sequence[tuple[str, ...]],
    levels: Sequence[re.Pattern[str]],
    min_tail: int = 1,
) -> Optional[AnchoredMatch]:
    """Match a tree whose id folders sit at a fixed depth.

    Exactly one leading wrapper is stripped: the first in `wrappers` that
    matches, so callers list them longest first and `()` last. There is no
    fall-through and no substring search.

    Args:
        parts: The member's components.
        wrappers: Candidate leading folders.
        levels: One pattern per id level, each `fullmatch`ed.
        min_tail: Fewest components required below the ids.

    Returns:
        The split, or None when the path does not fit.
    """
    for wrapper in wrappers:
        if tuple(parts[: len(wrapper)]) == tuple(wrapper):
            rest = tuple(parts[len(wrapper):])
            break
    else:
        return None
    if len(rest) < len(levels) + min_tail:
        return None
    ids = rest[: len(levels)]
    if not all(p.fullmatch(i) for p, i in zip(levels, ids)):
        return None
    return AnchoredMatch(tuple(wrapper), ids, rest[len(levels):])


def build_dest(
    subtree: str,
    ids: Sequence[str],
    tail: Sequence[str],
    *,
    member: ImportMember,
    expected: str,
    max_component_bytes: int = 255,
) -> Union[PurePosixPath, ImportRefusal]:
    """Join rewritten ids and a tail under a subtree, re-checking every component.

    Args:
        subtree: The destination subtree, relative to `save_root`.
        ids: The id components, already rewritten by the caller.
        tail: The components below them.
        member: The member, for the refusal.
        expected: The accepted shape, in words.
        max_component_bytes: Longest name the destination filesystem takes.

    Returns:
        The destination, or an `unsafe_path` refusal.
    """
    for part in (*ids, *tail):
        problem = _name_problem(part) or _component_problem(part, max_component_bytes)
        if problem:
            return ImportRefusal("unsafe_path", member.name, expected, detail=problem)
    return PurePosixPath(subtree, *ids, *tail)


_PS_DASHED = re.compile(r"([A-Z]{4})[-_ ]?(\d{3})\.?(\d{2})", re.I)
"""A PlayStation serial in any of its spellings: `SLUS-20001`, `SLUS_200.01`, `slus20001`."""
_PS_NODASH = re.compile(r"([A-Za-z]{4})[-_ ]?(\d{5})")
"""A PSP/PS3 serial with or without its separator."""
_HEX8 = re.compile(r"(?:0x)?([0-9A-Fa-f]{8})")
"""An eight-digit hex title id, optionally `0x`-prefixed."""
_XBOX_CODE = re.compile(r"([A-Za-z]{2})-(\d{3})")
"""An Xbox publisher-code-and-number id such as `MS-100`."""
_GAME_ID = re.compile(r"[A-Za-z0-9]{4}(?:[A-Za-z0-9]{2})?")
"""A GameCube/Wii game id: four characters, plus two for the maker."""
_HEX16 = re.compile(r"(?:0x)?([0-9A-Fa-f]{16})")
"""A sixteen-digit hex title id, as the Switch writes it."""


def _ps_serial_dashed(raw: str) -> Optional[str]:
    """Normalise a PlayStation serial to `XXXX-NNNNN`.

    Args:
        raw: The id as found.

    Returns:
        The canonical id, or None.
    """
    m = _PS_DASHED.fullmatch(raw.strip())
    return f"{m[1].upper()}-{m[2]}{m[3]}" if m else None


def _ps_serial_nodash(raw: str) -> Optional[str]:
    """Normalise a PSP/PS3 serial to `XXXXNNNNN`.

    Args:
        raw: The id as found.

    Returns:
        The canonical id, or None.
    """
    m = _PS_NODASH.fullmatch(raw.strip())
    return f"{m[1].upper()}{m[2]}" if m else None


def _hex8(raw: str) -> Optional[str]:
    """Normalise an eight-digit hex id to upper case, without `0x`.

    Args:
        raw: The id as found.

    Returns:
        The canonical id, or None.
    """
    m = _HEX8.fullmatch(raw.strip())
    return m[1].upper() if m else None


def _xbox(raw: str) -> Optional[str]:
    """Normalise an Xbox title id: hex, or a publisher code like `MS-100`.

    Args:
        raw: The id as found.

    Returns:
        The canonical eight-digit hex id, or None.
    """
    hexed = _hex8(raw)
    if hexed:
        return hexed
    m = _XBOX_CODE.fullmatch(raw.strip())
    if not m:
        return None
    a, b = m[1].upper()
    return f"{ord(a):02X}{ord(b):02X}{int(m[2]):04X}"


def _gc_wii_disc(raw: str) -> Optional[str]:
    """Normalise a GameCube/Wii id to the hex of its four-character game code.

    Args:
        raw: The id as found: a game id like `GZLE01`, or its hex.

    Returns:
        The canonical eight-digit hex id, or None.
    """
    hexed = _hex8(raw)
    if hexed:
        return hexed
    value = raw.strip()
    if not _GAME_ID.fullmatch(value):
        return None
    return value[:4].upper().encode("ascii").hex().upper()


def _hex16(raw: str) -> Optional[str]:
    """Normalise a sixteen-digit hex id, dropping any `/` separators.

    Args:
        raw: The id as found.

    Returns:
        The canonical id, or None.
    """
    m = _HEX16.fullmatch(raw.strip().replace("/", ""))
    return m[1].upper() if m else None


def _dc_product(raw: str) -> Optional[str]:
    """Dreamcast product numbers are not unique enough to compare.

    Args:
        raw: The id as found.

    Returns:
        None, always: identity is not checked for this family.
    """
    return None


NORMALISERS: dict[str, Callable[[str], Optional[str]]] = {
    "ps_serial_dashed": _ps_serial_dashed,
    "ps2_card_dir": lambda raw: raw.strip().upper() or None,
    "ps_serial_nodash": _ps_serial_nodash,
    "hex8": _hex8,
    "xbox": _xbox,
    "gc_wii_disc": _gc_wii_disc,
    "hex16": _hex16,
    "dc_product": _dc_product,
    "scummvm_target": lambda raw: raw.strip().casefold() or None,
}
"""One normaliser per `IdFamily`: each brings every notation of an id to one form."""

IdentityPolicy = Literal["none", "advisory", "strict", "required"]
"""How hard a hook holds a member to the session's id."""


@dataclass(frozen=True)
class IdentitySource:
    """Where an emulator's session identity comes from.

    Attributes:
        family: The id family the session id is normalised in.
        rom_reader: Reads the id off the rom file, or None when it cannot.
        use_save_target: Whether RomM's value is `save_target` rather than `title_id`.
        romm_family: The family RomM's value is written in, when it differs from `family`.
    """

    family: IdFamily
    rom_reader: Optional[Callable[[Path], Optional[str]]] = None
    use_save_target: bool = False
    romm_family: Optional[IdFamily] = None


def resolve_session_identity(
    ctx: ImportCtx,
    *,
    family: IdFamily,
    rom_reader: Optional[Callable[[Path], Optional[str]]] = None,
    use_save_target: bool = False,
    romm_family: Optional[IdFamily] = None,
) -> SessionIdentity:
    """Work out the game id the session runs as, once per preflight.

    The rom wins over RomM: RomM's id is metadata a user can get wrong,
    while the id read off the rom is what the emulator will actually use.

    Args:
        ctx: The launch context; its `memo` caches the answer.
        family: The id family to normalise into.
        rom_reader: Reads the id off `ctx.rom_file`, or None.
        use_save_target: Whether RomM's value is `save_target` rather than `title_id`.
        romm_family: The family RomM's value is written in, when it differs.

    Returns:
        The identity, with the source it came from.
    """
    key = ("identity", family, use_save_target)
    cached = ctx.memo.get(key)
    if isinstance(cached, SessionIdentity):
        return cached
    normalise = NORMALISERS[family]
    from_rom: Optional[str] = None
    if rom_reader is not None and ctx.rom_file is not None:
        try:
            raw = rom_reader(ctx.rom_file)
        except Exception as exc:
            log.warning("imports: could not read an id off %s: %s", ctx.rom_file, exc)
            raw = None
        from_rom = normalise(raw) if raw else None
    from_romm: Optional[str] = None
    raw_romm = (ctx.rom.save_target if use_save_target else ctx.rom.title_id) if ctx.rom else None
    if raw_romm:
        from_romm = NORMALISERS[romm_family or family](raw_romm)
        if from_romm is None:
            log.info("imports: RomM id %r is not a %s id, ignoring it", raw_romm, romm_family or family)
    if from_rom and from_romm and from_rom != from_romm:
        log.warning("imports: the rom says %s but RomM says %s; going with the rom", from_rom, from_romm)
    if from_rom:
        identity = SessionIdentity(from_rom, "rom")
    elif from_romm:
        identity = SessionIdentity(from_romm, "romm")
    else:
        identity = SessionIdentity(None, "none")
    ctx.memo[key] = identity
    return identity


def check_member_identity(
    member: ImportMember,
    member_id: Optional[str],
    session: SessionIdentity,
    *,
    family: IdFamily,
    policy: IdentityPolicy,
    expected: str,
    keyed: bool = True,
) -> Optional[ImportRefusal]:
    """Hold one member's id to the session's, under the hook's policy.

    Args:
        member: The member.
        member_id: The id the hook read off the member, already normalised, or None.
        session: The session's identity.
        family: The id family; `ps2_card_dir` matches by prefix.
        policy: `none` never refuses; `advisory` logs a mismatch; `strict`
            refuses one; `required` also refuses when the session has no id.
        expected: The accepted shape, in words.
        keyed: Whether the layout carries an id at all; an unkeyed layout is
            never refused for lacking one.

    Returns:
        A refusal, or None.
    """
    if policy == "none":
        return None
    if session.value is None:
        if policy == "required":
            return ImportRefusal(
                "identity_unknown",
                member.name,
                expected,
                detail="neither the rom nor RomM says which game this is",
            )
        return None
    if member_id is None:
        if keyed and policy in ("strict", "required"):
            return ImportRefusal(
                "unrecognised_layout", member.name, expected, detail="no game id in the path"
            )
        return None
    if family == "ps2_card_dir":
        matches = member_id.startswith(session.value)
    else:
        matches = member_id == session.value
    if matches:
        return None
    detail = f"member {member_id}, session {session.value} (from {session.source})"
    if policy == "advisory":
        log.info("imports: %s id mismatch, allowed: %s", member.name, detail)
        return None
    if session.source == "romm":
        detail += " - fix via PUT /api/roms/{id}/identity if RomM is wrong"
    return ImportRefusal("identity_mismatch", member.name, expected, detail=detail)


def identity_for(emulator: "Emulator", ctx: ImportCtx) -> SessionIdentity:
    """Resolve the session identity through the emulator's declared source.

    Args:
        emulator: The emulator.
        ctx: The launch context.

    Returns:
        The identity, or `none` when the emulator declares no source.
    """
    source = emulator.identity_source()
    if source is None:
        return SessionIdentity(None, "none")
    return resolve_session_identity(
        ctx,
        family=source.family,
        rom_reader=source.rom_reader,
        use_save_target=source.use_save_target,
        romm_family=source.romm_family,
    )


def resolve_activate_identity(
    emulator: "Emulator", rom_file: Optional[Path], rom: Optional[RomRef]
) -> SessionIdentity:
    """Resolve the identity for a launch that ran no preflight.

    Args:
        emulator: The emulator.
        rom_file: The resolved bootable file, or None.
        rom: The activate body's rom, or None.

    Returns:
        The identity.
    """
    ctx = ImportCtx(rom_file=rom_file, rom=rom, memory_card_synced=False, excluded=(), resume_slot=None)
    return identity_for(emulator, ctx)
