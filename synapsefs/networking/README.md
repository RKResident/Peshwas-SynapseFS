# Transfer

`push` and `pull` between two SynapseFS repositories over TCP, and a `serve`
mode that answers both.

```
make network                       # builds synapsefs/networking/spp

cd path/to/repo/.synapse && spp serve 9000
cd other/repo/.synapse   && spp pull 127.0.0.1 9000 main
cd other/repo/.synapse   && spp push 127.0.0.1 9000 main
```

**Run it from inside `.synapse/`.** Every path it touches is relative
(`objects/...`, `refs/heads/...`), so the working directory *is* how it finds
the repository. Running it from the repo root silently finds nothing.

## How it works

The side holding the branch walks the object graph from the branch tip and
computes the closure: commit -> checkpoint manifest -> header, config and
tensor manifests -> permutations and chunks, following `parents` and
`base_tensor_manifest` recursively. It sends that list, the peer replies with
a byte per hash saying which it already has, and only the missing objects go
over the wire. That is git's negotiation, and it means a second push of a
25-epoch run transfers almost nothing.

Objects are received into `<path>.tmp` and `rename`d into place, so an
interrupted transfer leaves either a complete object or none.

## What it does not do

**It does not verify what it receives.** Bytes are written to the path named by
the hash the sender claimed, and never re-hashed. The trust chain in
`ARCHITECTURE.md` 4.5 rests on every hash being checked against the value its
*parent* named, and this layer bypasses that entirely.

Hashes off the wire are validated as 64 lowercase hex characters, which is what
stops a hostile peer turning a hash into `../../etc/something` -- but that is a
path-traversal guard, not an integrity check.

**Run `synapsefs verify` after a pull.** It is the check this layer skips, it
is ref-anchored, and it runs at 544 MiB/s. A corrupted or substituted object is
caught; a silently truncated graph is caught too -- the missing-`config.json`
bug this code shipped with produced a repo where `log` worked and `verify` did
not.
