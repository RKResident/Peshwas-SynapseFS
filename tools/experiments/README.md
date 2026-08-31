# Measurement scripts

Every number quoted in `docs/ARCHITECTURE.md` comes from one of these. They run
against real checkpoints rather than synthetic data, because several
conclusions **changed** when tested on different data: the monotone key wins at
gap 1 and loses from gap 2 onward, and its verdict also flipped when the zstd
level changed from 3 to 1. A single-pair measurement would have been wrong.

Run from the repository root:

```
PYTHONPATH=. python tools/experiments/<script>.py
```

| script | reproduces | answers |
|---|---|---|
| `codec_ablation.py` | §4.1 | which transforms pay — subtract vs XOR, crossed with shuffle / bit-shuffle / PFor, at each star gap |
| `codec_split_point.py` | §4.1.1 | the k=8 cliff; why padded field planes and naive bitpacking both lose |
| `run_length.py` | §4.1.1 | why bit-level transforms fail: sign runs average 3.06 elements, bit-packing needs 8 |
| `codec_level_sweep.py` | §4.1.2 | zstd level is non-monotone; lzma is smaller but 9× slower to decompress |
| `storage_layout.py` | §5.4 | loose chunks vs a packfile, warm and cold, sequential and threaded |
| `memory_profile.py` | §5.4 | import baselines and codec temporaries — what a C++ port would and would not buy |
| `align_gpu.py` | §8 | GPU moves the matmuls; the LAP solve then dominates |
| `align_scaling.py` | §4.6.3 | cost against layer width and against noise; why it is n^2.5 and not n³, and where recovery breaks |

All default to `tools/checkpoints/` (the STL-10 run). Point them elsewhere with
`--checkpoints`, and note that **every result here is from one architecture** —
a 3.2M CNN trained with Adam. The margins are 0.2–1.2 percentage points, small
enough that a different model could reorder several of them.

`align_scaling.py` is the exception to both sentences: it needs no checkpoints
at all, because it *generates* MLPs with a known ground-truth permutation, and
it is the only script here that measures behaviour outside the 92M benchmark's
1792-unit width. Its weights are synthetic rather than trained, which is the
trade it makes to reach n = 10,000 on a machine with 15 GB of RAM.

## Two traps these scripts were themselves caught by

**`subprocess` forks, so a child's `ru_maxrss` includes the parent's footprint
at fork time.** `memory_profile.py` imports numpy lazily for exactly this
reason; importing it at module scope inflated every baseline by ~17 MiB and
made them all report the same number.

**Concatenating tensors before compressing flatters any scheme that benefits
from long runs.** `_common.load_streams` keeps tensors separate for schemes
where that matters.
