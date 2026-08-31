"""
call
> synapsefs/networking/spp serve <port>
"""
import argparse
import subprocess
from pathlib import Path


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "pull",
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
        help="Branch to pull tree"
    )
    parser.set_defaults(func=run, format_human=format_human)


def run(args: argparse.Namespace) -> dict:
    current_dir = Path(__file__).resolve().parent
    spp_path = current_dir.parent.parent / "networking/spp"

    result = subprocess.run(
        [str(spp_path), "pull", args.ip, args.port, args.branch],
        capture_output=True,
        text=True,
        check=True,
    )

    return {
        "port": args.port,
        "result": result.stdout,
        "error_code": result.stderr
    }

def format_human(result: dict) -> str:
    """Matches CLI.md ~7's worked example on the success path."""
    string = ""
    string += f"Serving on port {result['port']}\n"
    string += f"Output: {result['result']}"
    return string