# SynapseFS Networking

TCP networking for synchronizing objects and branches between two SynapseFS
repositories.

The networking executable, `spp`, provides three operations:

- `serve` — listen for incoming requests.
- `push` — send a local branch to a remote server.
- `pull` — fetch a remote branch.

## Quick start

Build the networking executable from the project root:

```bash
make network
```

This produces:

```text
synapsefs/networking/spp
```

The networking commands must be run **from inside a SynapseFS `.synapse/`
directory**, because all repository paths are relative to the current working
directory.

Start a server:

```bash
cd /path/to/repository/.synapse
/path/to/synapsefs/networking/spp serve 9000
```

From another repository, pull a branch:

```bash
cd /path/to/other/.synapse
/path/to/synapsefs/networking/spp pull <server-ip> 9000 main
```

Or push the local branch:

```bash
/path/to/synapsefs/networking/spp push <server-ip> 9000 main
```

For local testing, use `127.0.0.1` as the server address.

---

## Commands

### `serve`

```text
spp serve <port> [ro=<0|1>]
```

Starts a TCP server listening on all IPv4 interfaces.

Examples:

```bash
spp serve 9000
spp serve 9000 ro=1
```

The optional `ro` flag controls whether the server accepts pushes:

- `ro=0` — normal read/write server; pushes and pulls are accepted.
- `ro=1` — read-only server; pulls are accepted but pushes are rejected.

The server continues accepting connections until it is terminated.

### `push`

```text
spp push <ip> <port> <branch>
```

Sends the requested local branch to the remote server.

Example:

```bash
spp push 192.168.1.10 9000 main
```

The sender walks the branch's reachable object graph. The receiver reports
which objects it already has, and only missing objects are transferred.

### `pull`

```text
spp pull <ip> <port> <branch>
```

Fetches the requested branch from the remote server.

Example:

```bash
spp pull 192.168.1.10 9000 main
```

After all required objects have been received, the local
`refs/heads/<branch>` reference is updated to the branch tip.

---

## How synchronization works

SynapseFS stores repository data as content-addressed objects. A branch
references a commit, and commits and manifests reference other objects.

When serving a branch, the networking layer recursively walks the reachable
object graph:

```text
branch
└── commit
    ├── checkpoint manifest
    │   ├── header object
    │   ├── topology config
    │   └── tensor manifests
    │       ├── base tensor manifest
    │       ├── row permutation
    │       ├── column permutation
    │       └── chunks
    └── parent commits
        └── ...
```

The exact traversal is implemented by:

- `branch_get_objects()`
- `commit_get_objects()`
- `checkpoint_get_objects()`
- `tensor_get_objects()`

Referenced parents and base objects are followed recursively.

`HashList` maintains an ordered list of objects while preventing duplicates.

### Object negotiation

The sender first sends the hashes of all required objects:

```text
[uint32 object count]
[64-byte hash]
[64-byte hash]
...
```

The receiver responds with one status byte per hash indicating whether that
object is already present.

The sender then transmits only the missing objects.

Consequently, synchronizing a repository that already shares most of its object
graph requires little additional data.

---

## TCP protocol

The protocol uses raw TCP sockets. TCP provides the reliable ordered byte
stream; the application protocol defines the structure of the messages sent
over that stream.

### Connection setup

The client:

1. Connects to the server.
2. Sends the requested operation.
3. Sends the branch name.
4. Receives an acceptance or rejection status.
5. Performs the push or pull protocol.

Operation values are defined in `network_common.hpp`.

### Strings

Strings are transmitted as:

```text
[uint32 length]
[length bytes of string data]
```

The length is transmitted in network byte order.

### Hashes

A `Hash` is transmitted as its 64-character hexadecimal representation,
occupying exactly 64 bytes.

Valid hashes consist only of lowercase hexadecimal characters:

```text
0123456789abcdef
```

### Objects

An object is transmitted as:

```text
[uint32 size]
[size bytes of object data]
```

The size is transmitted in network byte order.

The current implementation buffers an entire object in memory while sending
or receiving it. This is appropriate for the project's current object sizes.

### Errors

The high 16 bits of the 32-bit object-size field are reserved for an error
marker:

```text
0xFFFF0000 | error code
```

This allows the receiver to distinguish an error response from a normal
object length.

The error codes are defined by the `Error` enum in `network_common.hpp`.

---

## Repository layout

The networking code expects the standard SynapseFS layout:

```text
.synapse/
├── objects/
│   ├── <hash prefix>/
│   │   └── ...
│   └── tmp/
└── refs/
    └── heads/
        └── <branch>
```

Objects are located from their hashes. Branch references are stored under
`refs/heads/`.

The networking layer does not define or modify the internal contents of
SynapseFS objects; it transfers them according to the object relationships
understood by the main SynapseFS implementation.

---

## Safe writes

Received objects are not written directly to their final paths.

Instead, the receiver:

1. Creates the destination directories if necessary.
2. Writes the complete object to a temporary file.
3. Closes the file and checks that the write succeeded.
4. Renames the temporary file to the final object path.

The branch reference is updated using the same temporary-file-and-rename
approach.

This prevents an interrupted transfer from leaving a partially written object
or branch reference at its final path.

---

## Validation

### Branch names

Branch names may contain:

- letters,
- digits,
- `_`,
- `-`,
- `/`.

Slashes cannot appear at the beginning or end of a branch name, and consecutive
slashes are rejected.

This validation is important because branch names are used to construct
filesystem paths.

### Hashes

Hashes received from a peer are checked to ensure that they are exactly
64 lowercase hexadecimal characters before being used as object paths.

This prevents malformed hashes from being interpreted as filesystem paths.

**Hash-format validation is not content-integrity verification.**

The networking layer does not currently re-compute the cryptographic hash of
every received object. A peer can therefore send valid-looking hash names
containing incorrect bytes.

If repository integrity must be checked, use the SynapseFS verification
facilities after synchronization.

---

## Read-only mode

A server started with:

```bash
spp serve 9000 ro=1
```

rejects `push` requests before any objects are transferred.

`pull` requests remain available, allowing the server to act as a read-only
source of repository data.

---

## IPv4 and networking requirements

The client currently accepts IPv4 addresses such as:

```text
127.0.0.1
192.168.1.10
```

The networking executable does not currently provide hostname or IPv6 handling
through its command-line interface.

For two machines on a network, ensure that:

- the server is running,
- both sides use the same TCP port,
- the client can reach the server's IPv4 address, and
- any firewall permits the selected port.

---

## Source files

| File | Responsibility |
|---|---|
| `network_common.hpp` | Shared protocol definitions, hashing/path helpers, validation, TCP helpers, and file transfer |
| `push.cpp` | Resolves a branch's object graph and sends required objects |
| `pull.cpp` | Negotiates and receives required objects, then updates the branch |
| `serve.cpp` | Accepts TCP connections and dispatches push/pull requests |
| `spp.cpp` | Command-line interface and client connection setup |

---

## Typical workflows

### Pull from a repository

On the source machine:

```bash
spp serve 9000
```

On the destination machine:

```bash
spp pull <server-ip> 9000 main
```

### Push to a repository

On the destination machine:

```bash
spp serve 9000
```

On the source machine:

```bash
spp push <server-ip> 9000 main
```

### Test on one machine

Terminal 1:

```bash
cd /path/to/server/.synapse
/path/to/synapsefs/networking/spp serve 9000
```

Terminal 2:

```bash
cd /path/to/client/.synapse
/path/to/synapsefs/networking/spp pull 127.0.0.1 9000 main
```

---

## Limitations

The current implementation is intentionally small and has several limitations:

- TCP over IPv4 only.
- No encryption.
- No peer authentication.
- No authentication-based authorization.
- Read-only mode is the only server-side access-control mechanism.
- Received object contents are not re-hashed by the networking layer.
- Transfers cannot currently be resumed after a connection failure.
- Object lengths are limited to 32-bit unsigned values.
- Objects are buffered in memory during transfer.
- The server handles connections sequentially.
- The protocol is specific to the current SynapseFS object model.

These are protocol/implementation limitations rather than requirements of TCP
itself.
