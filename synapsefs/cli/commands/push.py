"""
call
> synapsefs/networking/spp serve <port>
"""
import argparse
import subprocess
from pathlib import Path


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
    current_dir = Path(__file__).resolve().parent
    spp_path = current_dir.parent.parent / "networking/spp"

    process = subprocess.Popen(
        [str(spp_path), "push", args.ip, args.port, args.branch],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="")

    for line in process.stderr:
        print(line, end="")

    return {
        "port": args.port,
        "result": process.stdout,
        "error_code": process.stderr
    }

def format_human(result: dict) -> str:
    """Matches CLI.md ~7's worked example on the success path."""
    string = ""
    string += f"Serving on port {result['port']}\n"
    string += f"Output: {result['result']}"
    return string