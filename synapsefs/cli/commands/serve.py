"""
call
> synapsefs/networking/spp serve <port>
"""
import argparse
import subprocess
from pathlib import Path


def add_subparser(subparsers, global_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "serve",
        parents=[global_parser],
        help="Starts the sync listener so another instance can push/pull against this repo.",
    )
    parser.add_argument(
        "port",
        help="Literally what the name says"
    )
    parser.set_defaults(func=run, format_human=format_human)


def run(args: argparse.Namespace) -> dict:
    current_dir = Path(__file__).resolve().parent
    spp_path = current_dir.parent.parent / "networking/spp"
    # print(spp_path)

    result = subprocess.run(
        [str(spp_path), "serve", args.port],
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