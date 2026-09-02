# SynapseFS

**Permutation-aware, cryptographically verifiable version control and virtual filesystem for neural network checkpoints.**

SynapseFS is a purpose-built version control system and virtual file system designed specifically for machine learning models stored in `.safetensors` format. It solves the massive storage and bandwidth overheads of deep learning checkpoints by combining **Git Re-Basin permutation alignment**, **lossless integer-domain residual codecs**, **content-addressed packfile storage**, and a **read-only POSIX FUSE engine** that serves arbitrary checkpoints on demand with zero disk pre-materialization.

---

## Table of Contents

- [Core Principles & Design Goals](#core-principles--design-goals)
- [Architecture & Key Innovations](#architecture--key-innovations)
  - [1. Permutation-Aware Alignment Engine](#1-permutation-aware-alignment-engine)
  - [2. Lossless Integer-Domain Residual Codec](#2-lossless-integer-domain-residual-codec)
  - [3. Star Commit Topology ($N=4$)](#3-star-commit-topology-n4)
  - [4. Content-Addressed On-Disk Storage & Packs](#4-content-addressed-on-disk-storage--packs)
  - [5. Cryptographic Verification & Multi-Tier Trust Model](#5-cryptographic-verification--multi-tier-trust-model)
  - [6. Zero-Materialization Read-Only POSIX FUSE Mount](#6-zero-materialization-read-only-posix-fuse-mount)
- [On-Disk Repository Layout](#on-disk-repository-layout)
- [CLI Reference](#cli-reference)
  - [Global Conventions & Flags](#global-conventions--flags)
  - [Commands](#commands)
- [Design Decisions & Benchmark Trade-offs](#design-decisions--benchmark-trade-offs)
- [Installation & Quick Start](#installation--quick-start)
  - [Prerequisites](#prerequisites)
  - [1. Installation](#1-installation)
  - [1a. Verifying the install](#1a-verifying-the-install)
  - [1b. The C++ transfer tool](#1b-the-c-transfer-tool)
  - [2. Workflow Walkthrough](#2-workflow-walkthrough)
- [Examples & Interactive Tutorials](#examples--interactive-tutorials)
- [Development & Testing](#development--testing)

---

## Core Principles & Design Goals

1. **Byte-for-Byte Exactness by Construction**: Never re-serialize or normalize data that can be stored verbatim. The 8-byte length prefix and exact JSON `.safetensors` header (including `__metadata__`, key ordering, and whitespace padding) are preserved bit-for-bit. Reconstruction yields byte-identical files to the originals.
2. **Zero Floating-Point Drift**: All delta encoding, residual calculation, and reconstruction happen purely in the integer domain on raw bit patterns mod $2^n$. No float rounding, subnormal flushing, or NaN/signed-zero loss occurs anywhere in storage or retrieval.
3. **Identity Fast Path**: Fine-tuning checkpoints often share neuron order. The alignment solver, codec, reconstructor, and FUSE mount all short-circuit when the permutation is the identity matrix, avoiding unnecessary $O(n^3)$ LAP sweeps and enabling contiguous page cache reads.
4. **Crash Safety by Immutability**: All storage objects are immutable and written via an atomic write-and-rename sequence (`tmp/<uuid>` $\to$ `fsync` $\to$ `rename` $\to$ `fsync dir`). Ref pointers update last. In-flight crashes can leave orphaned objects, but never a corrupted or inconsistent state.
5. **Bounded Peak Memory (RSS)**: Indices are memory-mapped and binary-searched using parallel structured arrays. Large checkpoints are streamed row-batch by row-batch. PyTorch is strictly a test-time dependency and is never imported in the daemon hot path.

---

## Architecture & Key Innovations

```
                                 ┌─────────────────────────┐
                                 │  Source .safetensors    │
                                 └────────────┬────────────┘
                                              │
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
         ┌─────────────────────────┐                     ┌─────────────────────────┐
         │ Raw Verbatim Header     │                     │ Weight Tensors          │
         │ (JSON + metadata blob)  │                     └────────────┬────────────┘
         └────────────┬────────────┘                                  │
                      │                                               ▼
                      │                                  ┌─────────────────────────┐
                      │                                  │ Permutation Alignment   │
                      │                                  │ (Git Re-Basin / LAP)    │
                      │                                  └────────────┬────────────┘
                      │                                               │
                      │                                               ▼
                      │                                  ┌─────────────────────────┐
                      │                                  │ Lossless Integer Codec  │
                      │                                  │ (Bit-pattern $\to$ zstd)│
                      │                                  └────────────┬────────────┘
                      │                                               │
                      └───────────────────────┬───────────────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │  Content-Addressed Objects  │
                               │  (Loose Manifests & Packs)  │
                               └──────────────┬──────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │ CLI Checkout / Restore  │                       │ POSIX FUSE Filesystem   │
        │ (Byte-Exact Generator)  │                       │ (Zero-Materialization)  │
        └─────────────────────────┘                       └─────────────────────────┘
```

### 1. Permutation-Aware Alignment Engine
Neural networks exhibit permutation symmetry: hidden units within a layer can be permuted arbitrarily without changing the function, causing naive weight deltas between fine-tuned checkpoints to appear artificially large.
- **Formulation**: Formulates weight alignment as a Linear Assignment Problem (LAP) maximizing $\langle W_A, P W_B \rangle$ per layer, coordinated across layers via coordinate descent.
- **Coupled Permutations**: Properly matches output permutations of layer $l$ to input permutations of layer $l+1$.
- **Convolutional Geometry (`col_block_size`)**: Handles conv-to-linear flattening by grouping columns in blocks of size $k_h \cdot k_w$ (or $H \cdot W$), avoiding channel corruption.
- **Not-Alignable Fallback**: Evaluates relative residual norms before and after alignment; if alignment does not meaningfully reduce the difference, SynapseFS gracefully falls back to raw storage and reports it explicitly.

### 2. Lossless Integer-Domain Residual Codec
- **Modular Integer Arithmetic**: Rather than subtracting floats, raw bit patterns are mapped to order-preserving integer domains:
  $$\Delta = \text{key}(B) - \text{key}(A) \pmod{2^n}$$
  Reconstruction is exact: $(\Delta + \text{key}(A)) \equiv \text{key}(B) \pmod{2^n}$.
- **Byte Shuffle & zstd**: Applies a deterministic byte-shuffle on the resulting residual byte stream to cluster high-order zeros, then compresses with zstandard level 1.
- **Special Values Preserved**: Distinguishes `+0.0` (`0x0000`) and `-0.0` (`0x8000`), NaNs, and infinities without edge-case branches.

### 3. Star Commit Topology ($N=4$)
Rather than creating deep linear delta chains ($C_4 \to C_3 \to C_2 \to C_1$) where reconstruction latency and memory scale linearly with depth, SynapseFS implements a **star topology**:
- Every delta commit diffs directly against the nearest anchor (hub) commit.
- Every $N=4$ commits (or when residual degradation triggers a re-base), a fresh full anchor commit is stored.
- **Result**: Maximum decode depth is bounded to **2 decodes**, delivering **$2.87\times$ faster reconstruction** and strictly bounded daemon memory overhead at a negligible $\approx 1.71\text{ pp}$ storage trade-off.

### 4. Content-Addressed On-Disk Storage & Packs
- **BLAKE3-256 Addressing**: All object hashes cover uncompressed canonical content, ensuring deduplication remains completely immune to compression level changes or library upgrades.
- **Self-Describing `.pack` Files**: Chunks are packed into sealed, immutable binary containers. Each record contains its content hash and plain/stored lengths, enabling $100\%$ index recovery from linear pack scans.
- **56-Byte `.idx` Memory-Mapped Indices**: Fanout table (256 entries) + parallel flat binary arrays for hashes, offsets, lengths, and checksums. No Python dictionary allocations in the lookup path.

### 5. Cryptographic Verification & Multi-Tier Trust Model
Trust is rooted at locally accepted branch references (`refs/heads/*`):
- **`--shallow`**: Walks and re-hashes loose metadata DAG links (commit $\to$ checkpoint-manifest $\to$ tensor-manifest).
- **`--fast`**: Verifies 8-byte stored-payload checksums across all referenced pack chunks without decompressing (rot/truncation scan).
- **Default (`--deep`)**: Decompresses every chunk and validates its 32-byte content hash against the authoritatively signed tensor-manifest, detecting crafted payload injections.
- **`--content`**: Fully reconstructs tensors end-to-end to verify final model tensor checksums.

### 6. Zero-Materialization Read-Only POSIX FUSE Mount
- **Virtual Directory Layout**:
  ```
  <mountpoint>/
  ├── <branch>/model.safetensors
  └── commits/<hash>/model.safetensors
  ```
- **On-Demand Slicing**: Converts arbitrary `pread(offset, size)` system calls into header byte slices and intersecting tensor row ranges.
- **Async Trio Loop & Worker Offloading**: `pyfuse3` single-threaded event loop delegates heavy zstd decompressions to worker threads (`trio.to_thread.run_sync`), preventing POSIX reader stalls.
- **LRU Chunk Cache**: Thread-safe byte-capped memory cache (`--cache-size`, default 512 MiB) prevents RSS blowup under concurrent read patterns.
- **Non-Root Access**: Fully functional in user space without `sudo` (mounts on any user-owned directory).

---

## On-Disk Repository Layout

```
.synapse/
├── objects/
│   ├── tmp/                          # Ephemeral staging directory (cleaned on startup)
│   ├── <hh>/<hash>                   # Loose objects (commits, manifests, headers, permutations)
│   └── pack/
│       ├── pack-<hash>.pack          # Sealed chunk packfiles
│       ├── pack-<hash>.idx           # Fanout binary indices
│       └── order                     # Newest-first pack search order
├── refs/
│   └── heads/<branch>                # Branch head tip (text file holding 64-char commit hash)
└── HEAD                              # Symbolic ref ("ref: refs/heads/main") or detached commit hash
```

---

## CLI Reference

### Global Conventions & Flags

Global options can appear before or after the subcommand:
- `-C, --repo <path>`: Operate on repository at `<path>` (defaults to searching upward for `.synapse/`).
- `--json`: Output machine-readable JSON on `stdout`.
- `-q, --quiet`: Suppress progress output on `stderr`.
- `-v, --verbose`: Enable detailed tensor-level logs.
- `--no-color`: Disable ANSI color codes.

#### Standard Exit Codes
- `0`: OK / Success
- `1`: General Error
- `2`: Usage / Argument Syntax Error
- `3`: Not a SynapseFS Repository
- `4`: **Integrity Failure** (hash mismatch, corrupted pack, tamper detection)
- `5`: Not Alignable (under `--strict`)
- `6`: Merge Conflict
- `7`: Network / Protocol Failure
- `8`: Mount / FUSE Daemon Failure

---

### Commands

#### `init`
Initializes a new, empty SynapseFS repository.
```bash
synapsefs init [<path>] [--branch <name>]
```

#### `commit`
Ingests a `.safetensors` checkpoint, aligns it against the base commit, computes integer residuals, and writes a new commit object.
```bash
synapsefs commit <model.safetensors> -m "Commit message" \
                 [--config <config.json>] [--base <ref>] \
                 [--no-align] [--chunk-size <bytes>] [--strict]
```
*(Note: `--config` is required on the first root commit to establish the model topology).*

#### `checkout`
Switches branches or reconstructs a commit's checkpoint byte-identically into the working tree.
```bash
synapsefs checkout <branch|commit> [--out <path>] [--no-materialize]
```

#### `branch`
Lists, creates, deletes, or renames repository branches.
```bash
synapsefs branch [-a] [-d <branch>] [-m <old> <new>] [<new-branch>]
```

#### `log`
Displays commit history graph, messages, timestamps, and commit hashes.
```bash
synapsefs log [<ref>] [-n <count>] [--oneline]
```

#### `verify`
Cryptographically verifies DAG integrity and chunk content hashes.
```bash
synapsefs verify [<ref>] [--shallow] [--fast] [--deep] [--content] [--full]
```

#### `merge`
Performs a three-way merge between the current branch and another branch tip.
```bash
synapsefs merge <branch> [-m <msg>] [--no-commit]
```

#### `mount`
Mounts a read-only virtual filesystem exposing commits and branches as virtual `.safetensors` files without writing them to disk.
```bash
synapsefs mount <mountpoint> [--ref <ref>] [--foreground] \
              [--cache-size <bytes>] [--allow-other] [--debug-fuse]
```

#### `unmount`
Unmounts a mounted virtual filesystem, cleanly stopping the background daemon with fallback to `fusermount3 -u`.
```bash
synapsefs unmount <mountpoint>
```

#### `restore`
Directly reconstructs a commit to a destination file and optionally validates it against a reference file.
```bash
synapsefs restore <ref> --out <dest.safetensors> [--reference <ref.safetensors>]
```

---

## Design Decisions & Benchmark Trade-offs

| Design Decision | Chosen Implementation | Rejected Alternative | Why |
|---|---|---|---|
| **Header Handling** | Stored raw and replayed verbatim | Re-serialized from JSON dict | Re-serialization alters key order, space-padding, and metadata, destroying byte-exactness. |
| **Delta Domain** | Modular integer subtraction on raw bits | Float subtraction ($\Delta = B - A$) | Float arithmetic loses bits to rounding and subnormal handling, making exact reconstruction impossible. |
| **Commit Topology** | Star topology ($N=4$) | Deep linear delta chain | Star guarantees $2$-hop decodes ($2.87\times$ faster reconstruction) at only $1.71\text{ pp}$ storage overhead. |
| **Index Representation** | 56-byte fanout structured binary array | Python `dict[hex_str, ...]` | Python dictionaries consume $50\text{--}80\text{ MB}$ RSS at scale; mmap array uses near-zero resident memory. |
| **Chunk Hashing** | BLAKE3 of uncompressed content | BLAKE3 of compressed payload | Hashing compressed payloads causes deduplication to fail whenever zstd levels or dictionaries change. |
| **Daemon Architecture** | Non-root `pyfuse3` + Trio + Threadpool | PyTorch in daemon | Importing PyTorch adds hundreds of MB to daemon RSS; pure numpy + C FUSE keeps RSS minimal and avoids root requirement. |

---

## Installation & Quick Start

Every step below was verified from an empty virtualenv on a clean interpreter.

### Prerequisites

**Python 3.11 or newer.** Ubuntu 22.04 ships 3.10 as `python3`, which `pip`
rejects outright (`Package 'synapsefs' requires a different Python`). Check with
`python3 -V` and use `python3.12`/`python3.13`/`python3.14` explicitly if the
default is older.

**System packages.** Two extension modules are compiled during install --
`pyfuse3` and this project's own Cython codec kernel -- so a compiler and
headers are required, not just a Python environment:

```bash
# Debian / Ubuntu
sudo apt install build-essential pkg-config libfuse3-dev fuse3 \
                 python3-dev            # must match your interpreter,
                                        # e.g. python3.12-dev for python3.12

# Fedora
sudo dnf install gcc gcc-c++ pkgconf-pkg-config fuse3-devel fuse3 python3-devel

# Arch
sudo pacman -S base-devel pkgconf fuse3 python
```

Omitting `python3-dev` is the most common failure: the build gets as far as the
compiler and stops at `fatal error: Python.h: No such file or directory`, for
both `pyfuse3` and the codec kernel. `libfuse3-dev` and `pkg-config` are needed
by `pyfuse3`'s build, and `fuse3` provides the `fusermount3` binary that
`mount`/`unmount` shell out to.

**CPU.** The codec kernel is compiled with `-march=x86-64-v3` (see `setup.py`),
which requires AVX2 -- Intel Haswell / AMD Excavator, 2013 or later. On an older
CPU or a non-x86 machine, drop that flag from `setup.py`; the build then falls
back to portable C and everything still works, just slower on the unshuffle.

### 1. Installation

```bash
git clone https://github.com/Peshwas-SynapseFS/Peshwas-SynapseFS.git
cd Peshwas-SynapseFS

python3.12 -m venv .venv          # or any interpreter >= 3.11
source .venv/bin/activate

pip install -e .
```

**Do not pass `--no-build-isolation`.** `setup.py` imports `Cython` and `numpy`
at build time; with isolation, pip provides them from `[build-system] requires`,
and without it they have to be in the venv already -- which on a fresh venv they
are not, so the install dies with `ModuleNotFoundError: No module named
'Cython'` before it reads a single dependency.

For the test suite and the torch-based examples:

```bash
pip install -e ".[dev]"           # adds pytest, torch and matplotlib
make fixtures                     # generate the tiny fixtures the tests need
make test
```

### 1a. Verifying the install

```bash
synapsefs --help                                        # CLI imports cleanly
python -c "from synapsefs.codec.chunk import HAS_FAST_CHUNK; print(HAS_FAST_CHUNK)"
```

`HAS_FAST_CHUNK` printing `True` means the Cython kernel compiled and is in use.
`False` is not fatal -- `chunk.py` falls back to a pure-numpy path -- but the
unshuffle is roughly 6x slower, so it is worth fixing rather than ignoring. The
compiled artifact is `synapsefs/codec/fast_chunk*.so`; it is gitignored and
rebuilt per machine.

### 1b. The C++ transfer tool

Push/pull between peers is a standalone C++ binary, deliberately not part of the
Python package, so a machine can host a repository without a Python environment:

```bash
make network                      # builds synapsefs/networking/spp
```

A prebuilt `spp` is committed to the repository as an **x86-64 ELF binary**.
Rebuild it on any new machine rather than trusting the checked-in one -- it will
not run on a different architecture, and it is not guaranteed to match the
current `spp.cpp`. It is not installed onto `PATH`; invoke it by path, and
**run it from inside `.synapse/`** -- every path it uses is relative to the
repository's internal directory, so from the repo root it silently finds
nothing. See `synapsefs/networking/README.md` for the protocol and its
deliberate omissions (it does not verify what it receives -- run
`synapsefs verify` after a pull).

### 2. Workflow Walkthrough

```bash
# 1. Initialize a repo
synapsefs init my_model_repo
cd my_model_repo

# 2. Make the initial root commit (providing config.json topology)
synapsefs commit /path/to/base_model.safetensors \
                 --config /path/to/config.json \
                 -m "Base model initialization"

# 3. Commit a fine-tuned checkpoint
synapsefs commit /path/to/finetuned_model.safetensors \
                 -m "Epoch 1 fine-tuning"

# 4. Verify cryptographic integrity
synapsefs verify

# 5. Mount virtual filesystem (non-root)
mkdir -p /tmp/model_mount
synapsefs mount /tmp/model_mount

# 6. Read virtual models directly with torch/safetensors
python3 -c "
from safetensors.torch import load_file
tensors = load_file('/tmp/model_mount/main/model.safetensors')
print('Loaded tensors:', len(tensors))
"

# 7. Unmount when finished
synapsefs unmount /tmp/model_mount
```

---

## Examples & Interactive Tutorials

We provide ready-to-run Jupyter notebooks in the [`examples/`](/examples) directory demonstrating SynapseFS workflows:

### [MNIST Training & FUSE Mount Demo](/examples/mnist-demo.ipynb) (`examples/mnist-demo.ipynb`)
An end-to-end tutorial showing:
- **PyTorch CNN Training**: Trains a convolutional classifier on MNIST.
- **Continuous Checkpoint Commits**: Automatically tracks each training epoch as a `.safetensors` file with integer-domain delta compression.
- **DAG & Residual Inspection**: Inspects the resulting star-topology commit graph and compression ratios with `synapsefs log`.
- **Deep Cryptographic Verification**: Validates repository integrity against malicious chunk modifications with `synapsefs verify --deep`.
- **Zero-Disk FUSE Mount**: Mounts the repository as a read-only POSIX virtual filesystem, directly loads checkpoints into PyTorch (`safetensors.torch.load_file`), evaluates test accuracy, and unmounts cleanly without writing any file to disk.

To run the examples:
```bash
pip install -e ".[examples]"
jupyter notebook examples/mnist-demo.ipynb
```

---

## Development & Testing

Run the test suite and code verification:

```bash
# Run compiler checks and test suite
make lint
make test

# Run pytest directly with coverage/verbose options
.venv/bin/python -m pytest
```