"""Reviewer CLI for Groundwire tool-call approvals.

Lists the calls an AI agent is waiting on, shows exactly what would run, and
approves or rejects them. Talks to the gateway's ``/v1/approvals`` API with the
reviewer token from ``GROUNDWIRE_APPROVER_TOKEN`` (or ``--token``).

Usage::

    python scripts/approvals.py list                     # pending requests
    python scripts/approvals.py list --status approved
    python scripts/approvals.py show apr_...
    python scripts/approvals.py approve apr_... --reviewer anita --note "customer verified"
    python scripts/approvals.py reject  apr_... --reviewer anita --note "duplicate claim"

Exit codes: 0 done, 1 the gateway refused (unknown id, already decided,
expired, bad token), 2 bad usage or gateway unreachable.
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import sys
from typing import Any

import httpx

logger = logging.getLogger("approvals")

TOKEN_ENV = "GROUNDWIRE_APPROVER_TOKEN"  # noqa: S105 - env var name, not a secret


class ApiError(Exception):
    """The gateway answered with an error."""

    def __init__(self, status: int, message: str) -> None:
        """Create the error.

        Args:
            status: HTTP status.
            message: Gateway error message.
        """
        super().__init__(f"{status}: {message}")


def _request(client: httpx.Client, method: str, path: str,
             body: dict[str, Any] | None = None) -> Any:
    """Call the approvals API and return the decoded JSON.

    Args:
        client: Authenticated client.
        method: HTTP method.
        path: Path under the gateway.
        body: JSON body.

    Returns:
        Decoded response.

    Raises:
        ApiError: For a non-2xx response.
        httpx.HTTPError: On transport errors.
    """
    response = client.request(method, path, json=body)
    try:
        data = response.json()
    except ValueError:
        data = {}
    if response.is_error:
        error = data.get("error", {}) if isinstance(data, dict) else {}
        raise ApiError(response.status_code, str(error.get("message") or response.text))
    return data


def _row(record: dict[str, Any]) -> str:
    """One line per approval for ``list``."""
    arguments = json.dumps(record.get("arguments", {}), separators=(",", ":"))
    return (f"{record['id']}  {record['status']:<9} {record['tool']:<16} "
            f"rule={record.get('rule') or '-'}  user={record.get('scope')}  "
            f"created={record.get('created_at')}  args={arguments}")


def _show(record: dict[str, Any]) -> str:
    """Multi-line detail for ``show`` and decisions."""
    return json.dumps(record, indent=2, ensure_ascii=False)


def run(args: argparse.Namespace, client: httpx.Client) -> int:
    """Execute one subcommand.

    Args:
        args: Parsed arguments.
        client: Authenticated client.

    Returns:
        Exit code.

    Raises:
        ApiError: The gateway refused.
        httpx.HTTPError: On transport errors.
    """
    if args.command == "list":
        data = _request(client, "GET", f"/v1/approvals?status={args.status}&limit={args.limit}")
        records = data.get("data", [])
        if not records:
            logger.info("no %s approvals", args.status)
        for record in records:
            logger.info(_row(record))
        return 0
    if args.command == "show":
        logger.info(_show(_request(client, "GET", f"/v1/approvals/{args.id}")))
        return 0
    body = {"reviewer": args.reviewer, "note": args.note}
    record = _request(client, "POST", f"/v1/approvals/{args.id}/{args.command}", body)
    logger.info("%s %s by %s", record["id"], record["status"], record.get("reviewer"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface definition."""
    parser = argparse.ArgumentParser(description="Review AI-agent tool calls held for approval.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="gateway base URL")
    parser.add_argument("--token", default=None, help=f"reviewer token (default: ${TOKEN_ENV})")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="list approvals")
    listing.add_argument("--status", default="pending",
                         choices=["pending", "approved", "rejected", "expired", "consumed"])
    listing.add_argument("--limit", type=int, default=50)
    show = sub.add_parser("show", help="show one approval, including its arguments")
    show.add_argument("id")
    for name in ("approve", "reject"):
        decide = sub.add_parser(name, help=f"{name} a pending call")
        decide.add_argument("id")
        decide.add_argument("--reviewer", default=getpass.getuser(),
                            help="your name for the audit trail (default: OS user)")
        decide.add_argument("--note", default=None, help="comment or reason")
    return parser


def configure_output() -> None:
    """Send results to stdout (pipeable: ``ID=$(... list | ...)``) and problems to stderr."""
    results = logging.StreamHandler(sys.stdout)
    results.setLevel(logging.INFO)
    results.addFilter(lambda record: record.levelno < logging.WARNING)
    problems = logging.StreamHandler(sys.stderr)
    problems.setLevel(logging.WARNING)
    for handler in (results, problems):
        handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers[:] = [results, problems]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code.
    """
    configure_output()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    token = (args.token or os.environ.get(TOKEN_ENV, "")).strip()
    if not token:
        logger.error("no reviewer token: set $%s or pass --token", TOKEN_ENV)
        return 2
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with httpx.Client(base_url=args.url, headers=headers, timeout=15.0) as client:
            return run(args, client)
    except ApiError as exc:
        logger.error("gateway refused: %s", exc)
        return 1
    except httpx.ConnectError:
        logger.error("cannot reach the gateway at %s. Start it first: make run", args.url)
        return 2
    except httpx.HTTPError as exc:
        logger.error("request failed: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
