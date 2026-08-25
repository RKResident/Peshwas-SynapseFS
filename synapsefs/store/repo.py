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
from typing import Optional, Union

from synapsefs.errors import NoRepoError, UsageError
from synapsefs.store.atomic import atomic_write
from synapsefs.store.objectstore import ObjectStore

SYNAPSE_DIRNAME = ".synapse"


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

        Raises UsageError (CLI.md exit 2) if `.synapse/` already exists.
        """
        root = Path(path).resolve()
        synapse_dir = root / SYNAPSE_DIRNAME
        if synapse_dir.exists():
            raise UsageError(f"'{synapse_dir}' already exists")

        objects_dir = synapse_dir / "objects"
        tmp_dir = objects_dir / "tmp"
        (objects_dir / "pack").mkdir(parents=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        (synapse_dir / "refs" / "heads").mkdir(parents=True)

        head_path = synapse_dir / "HEAD"
        atomic_write(
            head_path,
            f"ref: refs/heads/{branch}\n".encode("utf-8"),
            tmp_dir=tmp_dir,
        )

        return cls(root)

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
