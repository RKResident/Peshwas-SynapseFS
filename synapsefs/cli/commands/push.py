"""
call
> synapsefs/networking/spp serve <port>
"""
import argparse
import subprocess
from pathlib import Path

from synapsefs.errors import NetworkError, UsageError
from synapsefs.store.repo import Repo


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "push",
        parents=[global_parser],
        help="pulls from server",
    )
    parser.add_argument(
        "ip",
        help="Ip address of the server"
    )
    parser.add_argument(
        "port",
        help="Port from which data will be transferred"
    )
    parser.add_argument(
        "branch",
        help="Branch to push tree"
    )
    parser.set_defaults(func=run, format_human=format_human)


def run(args: argparse.Namespace) -> dict:
    # `spp` resolves every path relative to its working directory --
    # `objects/...`, `refs/heads/...` -- so it has to run inside `.synapse/`.
    # Locating the repo here means the user runs this from the working tree,
    # like every other subcommand, instead of having to cd into `.synapse`
    # first. Without it the peer reports a filesystem error and the reason
    # ("Cannot open file refs/heads/main") is buried in the child's stderr.
    repo = Repo.find(args.repo)
    spp_path = Path(__file__).resolve().parent.parent.parent / "networking/spp"
    if not spp_path.is_file():
        raise UsageError(
            f"transfer helper not built: {spp_path} is missing. Run `make network`."
        )

    process = subprocess.Popen(
        [str(spp_path), "push", args.ip, args.port, args.branch],
        cwd=str(repo.synapse_dir),
        stdout=subprocess.PIPE,
        # Merged so the child's diagnostics interleave in order and neither
        # pipe can fill while the other is being drained.
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    for line in process.stdout:
        print(line, end="")
        lines.append(line.rstrip("\n"))
    code = process.wait()
    if code != 0:
        # CLI.md 1.3 reserves 7 for transport failures. Without this the
        # wrapper returned a dict carrying the failure and still exited 0, so
        # `synapsefs push ... && next-step` treated a refused connection as
        # success.
        raise NetworkError(
            f"push failed (spp exit {code})"
            + (f": {lines[-1]}" if lines else "")
        )

    return {
        "ip": args.ip,
        "port": args.port,
        "branch": args.branch,
        "repo": str(repo.root),
        "output": lines,
        "exit_code": code,
    }


def format_human(result: dict) -> str:
    return (f"Pushed '{result['branch']}' from {result['repo']} "
            f"to {result['ip']}:{result['port']}")
