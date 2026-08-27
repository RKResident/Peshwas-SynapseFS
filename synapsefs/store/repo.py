"""Repository-level layout: `.synapse/`, HEAD, and refs.

`Repo` sits one layer above `ObjectStore` -- it knows about the `.synapse/`
directory structure, `HEAD`, and `refs/heads/<branch>`, none of which
`ObjectStore` should know about. Refs are a different kind of thing than an
object: small, named, *mutable* pointers, not immutable hash-addressed
content, so they deliberately don't live inside the object store's put/get
model even though writing them safely uses the exact same primitive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from synapsefs.errors import NoRepoError, UsageError
from synapsefs.store.atomic import atomic_write
from synapsefs.store.objectstore import ObjectStore

SYNAPSE_DIRNAME = ".synapse"
_HEAD_REF_PREFIX = "ref: "
_REFS_HEADS_PREFIX = "refs/heads/"
_HEX_DIGITS = frozenset("0123456789abcdef")
_MIN_ABBREV_HASH_LEN = 6
_FULL_HASH_LEN = 64


def _is_valid_branch_name(name: str) -> bool:
    """Structural validity for a branch name used as a path component under
    `refs/heads/<name>` -- says nothing about whether that branch actually
    exists.

    A branch name becomes a filesystem path, so this is a path-traversal
    guard (reject `..`, `/`, a leading `-` that argparse-adjacent tooling
    could mistake for a flag, and the empty string), not cosmetics.
    """
    if not name:
        return False
    if name.startswith("-"):
        return False
    if "/" in name or "\\" in name:
        return False
    if ".." in name:
        return False
    return True


def _validate_branch_name(name: str) -> None:
    """Raise UsageError unless `name` is safe to use under `refs/heads/`."""
    if not _is_valid_branch_name(name):
        raise UsageError(f"invalid branch name: {name!r}")


class Repo:
    """A single SynapseFS repository rooted at `self.root`."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        self.synapse_dir = self.root / SYNAPSE_DIRNAME
        self.objects_dir = self.synapse_dir / "objects"
        self.refs_heads_dir = self.synapse_dir / "refs" / "heads"
        self.head_path = self.synapse_dir / "HEAD"
        self._store: Optional[ObjectStore] = None

    @property
    def store(self) -> ObjectStore:
        """Lazily-constructed ObjectStore over this repo's objects/ dir.

        Lazy so that merely *referencing* a Repo (e.g. `Repo.find()` just to
        check one exists) doesn't pay for a tmp/ GC scan every time.
        """
        if self._store is None:
            self._store = ObjectStore(self.objects_dir)
        return self._store

    @classmethod
    def init_at(cls, path: Union[str, Path], branch: str = "main") -> "Repo":
        """Create a new repository at `path` (CLI.md ~2).

        Creates `objects/`, `objects/tmp/`, `objects/pack/`, `refs/heads/`,
        and an attached HEAD pointing at `branch`. No branch ref file and no
        commit are created here -- `refs/heads/<branch>` only starts
        existing once the first commit lands, matching CLI.md's `branch`
        semantics: a freshly-initialized repo has no branches yet, only a
        HEAD that names one it *will* point at.

        Directory creation needs no atomicity by itself -- a partially
        created directory tree is incomplete, never corrupt, and could
        always be finished safely by re-running init or a repair step.
        HEAD's *content*, on the other hand, does need it:
        `ref: refs/heads/<branch>\\n` is a multi-byte write, and a crash
        mid-write could otherwise leave a truncated, ambiguous HEAD that
        every later command would have to defensively special-case. HEAD is
        written through the same `atomic_write` primitive objects use --
        not a second, bespoke implementation -- so there is exactly one
        write path to trust and test.

        Raises UsageError (CLI.md exit 2) if `.synapse/` already exists, or if
        `branch` is not a valid branch name.

        `branch` is validated up front, *before* any directory is created.
        `set_head_branch` below validates it too, but relying on that alone
        would be a trap: a rejected `--branch` would abort after the mkdirs
        had already run, leaving a `.synapse/` that has every directory and
        no HEAD -- and every retry would then hit the "already exists" check
        above and refuse, so a single typo would wedge the directory until
        someone deleted `.synapse/` by hand. Validating first makes a bad
        branch name a total no-op on disk.
        """
        _validate_branch_name(branch)

        root = Path(path).resolve()
        synapse_dir = root / SYNAPSE_DIRNAME
        if synapse_dir.exists():
            raise UsageError(f"'{synapse_dir}' already exists")

        objects_dir = synapse_dir / "objects"
        tmp_dir = objects_dir / "tmp"
        (objects_dir / "pack").mkdir(parents=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (synapse_dir / "refs" / "heads").mkdir(parents=True)

        repo = cls(root)
        repo.set_head_branch(branch)
        return repo

    @classmethod
    def find(cls, start: Union[str, Path] = ".") -> "Repo":
        """Resolve `-C/--repo` (CLI.md ~1.1): search upward from `start` for
        the nearest `.synapse/` directory, the way `git` searches upward
        for `.git/`.

        Raises NoRepoError (CLI.md exit 3) if no `.synapse/` is found before
        reaching the filesystem root.
        """
        current = Path(start).resolve()
        while True:
            if (current / SYNAPSE_DIRNAME).is_dir():
                return cls(current)
            if current.parent == current:
                raise NoRepoError(
                    f"not a synapsefs repository (or any parent up to /): {start}"
                )
            current = current.parent

    def read_head(self) -> Tuple[Optional[str], Optional[str]]:
        """Parse HEAD and resolve it one level, returning `(branch, commit_hash)`.

        Three cases, matching HEAD's two on-disk shapes:

        - Attached HEAD (`ref: refs/heads/<name>\\n`) whose branch ref file
          doesn't exist yet -- "unborn", the state right after `init` before
          any commit lands: `(name, None)`.
        - Attached HEAD whose branch ref file exists: `(name, hash)`.
        - Detached HEAD (HEAD itself holds a raw hex hash, no `ref:` line):
          `(None, hash)`.

        Callers are expected to already hold a `Repo` from `find()` or
        `init_at()`, both of which guarantee HEAD exists -- a missing HEAD
        here means a corrupted `.synapse/` dir, not a normal condition, so
        it's left to surface as a plain FileNotFoundError rather than
        papered over with a third typed case.
        """
        text = self.head_path.read_text(encoding="utf-8").strip()
        if text.startswith(_HEAD_REF_PREFIX):
            ref = text[len(_HEAD_REF_PREFIX):].strip()
            branch = (
                ref[len(_REFS_HEADS_PREFIX):]
                if ref.startswith(_REFS_HEADS_PREFIX)
                else ref
            )
            ref_path = self.refs_heads_dir / branch
            if ref_path.is_file():
                return branch, ref_path.read_text(encoding="utf-8").strip()
            return branch, None
        # No "ref: " prefix -- detached HEAD, the text itself is the hash.
        return None, text

    def _lookup_hash(self, ref: str) -> str:
        """Resolve a full or abbreviated hex hash to the full hash stored
        under `objects/<hh>/<hash>`.

        Raises UsageError if nothing matches (unknown ref) or more than one
        object matches an abbreviated prefix (ambiguous ref, per CLI.md
        ~1.4's "must be unambiguous").
        """
        lowered = ref.lower()
        if len(lowered) == _FULL_HASH_LEN:
            if not (self.objects_dir / lowered[:2] / lowered).is_file():
                raise UsageError(f"unknown ref: {ref!r} (no such object)")
            return lowered

        shard = self.objects_dir / lowered[:2]
        matches = (
            [p.name for p in shard.iterdir() if p.is_file() and p.name.startswith(lowered)]
            if shard.is_dir()
            else []
        )
        if not matches:
            raise UsageError(f"unknown ref: {ref!r} (no object matches this prefix)")
        if len(matches) > 1:
            raise UsageError(
                f"ambiguous ref: {ref!r} matches {len(matches)} objects, "
                f"need more characters"
            )
        return matches[0]

    def resolve_ref(self, ref: str) -> Optional[str]:
        """Resolve `ref` (CLI.md ~1.4) to a commit hash.

        Accepts `"HEAD"`, a branch name, or a full/abbreviated (>= 6 chars)
        hex commit hash, in that priority order -- a branch that happens to
        be named like a hex string still resolves as a branch first, mirroring
        real git's resolution order for an ambiguous name.

        Returns the resolved commit hash, or `None` if `ref` is `"HEAD"` and
        HEAD is unborn (no commit reachable from it yet) -- that is the one
        case where "doesn't exist" is legitimate rather than an error, per
        CLI.md ~3's "`--base` ... Ignored on the root commit."

        Raises UsageError if `ref` is malformed, names a branch that doesn't
        exist, or is an unknown/ambiguous hash.
        """
        if not ref:
            raise UsageError("empty ref")

        if ref == "HEAD":
            _, commit_hash = self.read_head()
            return commit_hash

        if _is_valid_branch_name(ref):
            branch_path = self.refs_heads_dir / ref
            if branch_path.is_file():
                return branch_path.read_text(encoding="utf-8").strip()

        lowered = ref.lower()
        looks_like_hash = (
            _MIN_ABBREV_HASH_LEN <= len(lowered) <= _FULL_HASH_LEN
            and all(c in _HEX_DIGITS for c in lowered)
        )
        if looks_like_hash:
            return self._lookup_hash(lowered)

        raise UsageError(f"unknown ref: {ref!r}")

    def update_ref(self, branch: str, commit_hash: str) -> None:
        """Atomically point `refs/heads/<branch>` at `commit_hash`.

        The single write path `commit`, `merge`, and (fast-forward)
        `checkout` all funnel through to advance a branch -- written once
        here rather than inlined per-command, same reasoning as
        `set_head_branch` below. Goes through `atomic_write` like every
        other durable write in this codebase; a torn ref file would be at
        least as bad as a torn object.

        Raises UsageError if `branch` isn't a safe `refs/heads/` path
        component (see `_is_valid_branch_name`).
        """
        _validate_branch_name(branch)
        target = self.refs_heads_dir / branch
        atomic_write(
            target,
            f"{commit_hash}\n".encode("utf-8"),
            tmp_dir=self.objects_dir / "tmp",
        )

    def set_head_branch(self, branch: str) -> None:
        """Atomically attach HEAD to `refs/heads/<branch>`.

        Writes `ref: refs/heads/<branch>\\n` through `atomic_write`, exactly
        as `init_at` used to inline -- this is now the one HEAD-writing path,
        used by both `init_at` (initial attach) and `checkout <branch>`
        (switching branches, CLI.md ~4).

        Raises UsageError if `branch` isn't a safe `refs/heads/` path
        component (see `_is_valid_branch_name`).
        """
        _validate_branch_name(branch)
        atomic_write(
            self.head_path,
            f"ref: refs/heads/{branch}\n".encode("utf-8"),
            tmp_dir=self.objects_dir / "tmp",
        )

    def set_head_detached(self, commit_hash: str) -> None:
        """Atomically point HEAD directly at `commit_hash`, detaching it.

        The other half of `checkout` (CLI.md ~4): `checkout <branch>` attaches
        HEAD to a ref, `checkout <commit>` writes the raw hash instead. The
        absence of a `ref: ` prefix is the *only* on-disk difference between
        the two states, which is what `read_head` keys off.
        """
        atomic_write(
            self.head_path,
            f"{commit_hash}\n".encode("utf-8"),
            tmp_dir=self.objects_dir / "tmp",
        )

    def branch_exists(self, name: str) -> bool:
        """Whether `refs/heads/<name>` exists.

        Structurally invalid names return False rather than raising: callers
        use this to *decide* whether an argument is a branch or a commit-ish
        (`checkout` does exactly that), and a name like `9f2c1a` must be
        answerable without an exception.
        """
        return _is_valid_branch_name(name) and (self.refs_heads_dir / name).is_file()

    def list_branches(self) -> dict:
        """`{branch name: commit hash}` for every ref under `refs/heads/`.

        Sorted by name so `branch`'s listing is stable between runs rather
        than following directory order. Subdirectories are ignored: branch
        names are validated to contain no `/`, so a directory here is not a
        ref this implementation ever wrote.
        """
        if not self.refs_heads_dir.is_dir():
            return {}
        return {
            path.name: path.read_text(encoding="utf-8").strip()
            for path in sorted(self.refs_heads_dir.iterdir())
            if path.is_file()
        }

    def delete_branch(self, name: str) -> str:
        """Remove `refs/heads/<name>`, returning the hash it pointed at.

        Refusing to delete the branch HEAD is attached to (CLI.md ~5,
        "Deleting the current branch fails with 2") is the caller's job, not
        this method's -- `Repo` reports state, the command decides policy.
        Deleting a ref is a plain unlink: no atomicity dance is needed because
        a ref either exists or it doesn't, and unlink is already atomic.
        """
        _validate_branch_name(name)
        path = self.refs_heads_dir / name
        if not path.is_file():
            raise UsageError(f"branch not found: {name!r}")
        commit_hash = path.read_text(encoding="utf-8").strip()
        path.unlink()
        return commit_hash

    def rename_branch(self, old: str, new: str) -> str:
        """Move `refs/heads/<old>` to `refs/heads/<new>`, re-attaching HEAD if
        it was pointing at `old`.

        Written as create-then-delete rather than `Path.rename`, so that a
        crash between the two steps leaves *both* refs rather than neither --
        two names for one commit is a cosmetic problem, a lost branch head is
        not. HEAD is re-attached last, for the same reason `commit` moves the
        ref last: until that write, HEAD still names a ref that exists.
        """
        _validate_branch_name(old)
        _validate_branch_name(new)
        source = self.refs_heads_dir / old
        if not source.is_file():
            raise UsageError(f"branch not found: {old!r}")
        if old != new and (self.refs_heads_dir / new).is_file():
            raise UsageError(f"branch already exists: {new!r}")

        commit_hash = source.read_text(encoding="utf-8").strip()
        self.update_ref(new, commit_hash)
        if old != new:
            source.unlink()
            branch, _ = self.read_head()
            if branch == old:
                self.set_head_branch(new)
        return commit_hash
