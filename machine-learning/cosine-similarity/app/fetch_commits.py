#!/usr/bin/env python3
"""Clone a GitHub repo (blobless) and extract commits in a time window to JSONL.

This script performs a fast, lightweight *blobless partial clone*
(``--filter=blob:none --no-checkout``) into a temporary directory. That fetches
every commit and tree object but no file contents, so commit metadata and
changed-file *names* are available immediately without ever downloading blobs.
The temporary clone is removed automatically when the script finishes.

Standard library only — no third-party packages and no GitHub API calls.

A ``GITHUB_TOKEN`` environment variable, if set, is used to authenticate to
private repos. The token is passed as an HTTP ``Authorization`` header via
``git -c http.extraHeader=...`` and is *never* embedded in the clone URL.

Output is one JSON object per line (JSONL) with keys ``hash``, ``author``,
``date``, ``subject``, ``body``, ``files`` — matching a downstream ``Commit``
dataclass.

Usage example::

    python fetch_commits.py owner/name --since 2025-01-01 --until 2025-06-01
    python fetch_commits.py https://github.com/owner/name.git \\
        --since 2025-01-01 --until 2025-06-01 --branch develop --out out.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import List, Optional

# ASCII control characters used as field/record separators in the git log
# format. RS (record separator) delimits commits; US (unit separator) delimits
# fields within a commit. Both are extremely unlikely to appear in commit text.
RS = "\x1e"  # record separator (between commits)
US = "\x1f"  # unit separator (between fields)

# Placing RS at the START of the format keeps each commit's --name-only file
# lines attached to its own record. The trailing US after %b bounds the body on
# both sides so multi-line bodies never leak into the file list.
LOG_FORMAT = f"format:{RS}%H{US}%an{US}%aI{US}%s{US}%b{US}"


class GitError(RuntimeError):
    """Raised when an underlying git command fails; carries git's stderr."""


def build_clone_url(repo: str) -> str:
    """Return a clone URL from ``owner/name`` shorthand or a full git URL."""
    # A bare ``owner/name`` has exactly one slash, no scheme, and no host. Any
    # of those markers means the caller already gave us a usable URL/path.
    if "://" in repo or repo.startswith("git@") or repo.endswith(".git"):
        return repo
    if repo.count("/") == 1 and ":" not in repo:
        return f"https://github.com/{repo}.git"
    return repo


def git_auth_args(token: Optional[str]) -> List[str]:
    """Return ``-c http.extraHeader=...`` args to authenticate, or ``[]``.

    The token is supplied as a header so it never lands in the URL (and thus
    never in the repo's stored remote config or in process listings of a URL).
    """
    if not token:
        return []
    return ["-c", f"http.extraHeader=Authorization: Bearer {token}"]


def _run_git(args: List[str], *, what: str) -> str:
    """Run ``git <args>`` and return stdout, raising GitError on failure."""
    try:
        proc = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # git not installed
        raise GitError("git executable not found on PATH") from exc
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "(no stderr)"
        raise GitError(f"{what} failed (exit {proc.returncode}):\n{stderr}")
    return proc.stdout


def clone_repo(url: str, dest: str, token: Optional[str]) -> None:
    """Blobless, no-checkout clone of ``url`` into ``dest``.

    ``--filter=blob:none`` skips file contents; ``--no-checkout`` avoids
    materializing a working tree. Together they keep the clone fast and ensure
    no blob downloads are triggered by later metadata-only ``git log`` work.
    """
    auth = git_auth_args(token)
    _run_git(
        [*auth, "clone", "--filter=blob:none", "--no-checkout", url, dest],
        what="git clone",
    )


def read_commits(
    repo_dir: str,
    since: str,
    until: str,
    branch: Optional[str],
) -> List[dict]:
    """Extract commits in [since, until] via a single ``git log`` call.

    Targets HEAD (the default branch) unless ``branch`` is given, in which case
    ``origin/<branch>`` is used. Merge commits are excluded.
    """
    ref = f"origin/{branch}" if branch else "HEAD"
    log_args = [
        "-C",
        repo_dir,
        "log",
        ref,
        "--no-merges",
        f"--since={since}",
        f"--until={until}",
        "--name-only",
        f"--pretty={LOG_FORMAT}",
    ]
    try:
        output = _run_git(log_args, what="git log")
    except GitError as exc:
        # A bad/missing branch surfaces here as a git error referencing the ref.
        if branch:
            raise GitError(
                f"could not read branch 'origin/{branch}'. "
                f"Does it exist on the remote?\n\n{exc}"
            ) from exc
        raise
    return parse_log(output)


def parse_log(output: str) -> List[dict]:
    """Parse the RS/US-delimited ``git log`` output into commit dicts."""
    commits: List[dict] = []
    # Split on RS; the first chunk is empty because the format starts with RS.
    for record in output.split(RS):
        if not record.strip():
            continue
        # First five fields are the commit metadata; the body's trailing US
        # means the 6th segment holds the --name-only file list (one per line).
        parts = record.split(US)
        if len(parts) < 6:
            continue  # malformed / unexpected; skip defensively
        hash_, author, date, subject, body = parts[:5]
        files_blob = parts[5]
        files = [line for line in files_blob.splitlines() if line.strip()]
        commits.append(
            {
                "hash": hash_,
                "author": author,
                "date": date,
                "subject": subject,
                "body": body.strip("\n"),
                "files": files,
            }
        )
    return commits


def write_jsonl(commits: List[dict], out_path: str) -> None:
    """Write commits as one JSON object per line."""
    with open(out_path, "w", encoding="utf-8") as fh:
        for commit in commits:
            fh.write(json.dumps(commit, ensure_ascii=False))
            fh.write("\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone a GitHub repo (blobless) and extract commits in a time "
            "window to JSONL, ready for an embeddings pipeline."
        )
    )
    parser.add_argument(
        "repo",
        help="owner/name shorthand or a full git URL.",
    )
    parser.add_argument(
        "--since",
        required=True,
        help="ISO-8601 date/datetime — start of the window (inclusive).",
    )
    parser.add_argument(
        "--until",
        required=True,
        help="ISO-8601 date/datetime — end of the window.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Branch to scan; defaults to the repo's default branch (HEAD).",
    )
    parser.add_argument(
        "--out",
        default="commits.jsonl",
        help="Output JSONL path (default: commits.jsonl).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    token = os.environ.get("GITHUB_TOKEN") or None
    url = build_clone_url(args.repo)

    try:
        # TemporaryDirectory cleans up the clone automatically, even on error.
        with tempfile.TemporaryDirectory(prefix="fetch_commits_") as tmpdir:
            clone_repo(url, tmpdir, token)
            commits = read_commits(tmpdir, args.since, args.until, args.branch)
    except GitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    write_jsonl(commits, args.out)

    branch_label = args.branch or "default branch (HEAD)"
    print(f"Extracted {len(commits)} commit(s) from {branch_label}")
    print(f"  date range: {args.since} .. {args.until}")
    print(f"  output:     {args.out}")
    if not commits:
        print("  note: no commits matched this window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
