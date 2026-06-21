#!/usr/bin/env python3
"""Incrementally sync a repo's commits into a per-repo SQLite index + JSONL.

This ties together the two earlier stages:

* ``fetch_commits.py`` — blobless clone + ``git log`` extraction in a window.
* ``commit_index.py``  — BGE-M3 embeddings stored in SQLite, cosine + rerank.

Each repository gets its own JSONL + SQLite pair, named from a lowercase repo
slug: ``eBay/evo-web`` -> ``ebay-evo-web.jsonl`` / ``ebay-evo-web.db``. The JSONL
filename is the single hint that names the db (``<stem>.jsonl`` -> ``<stem>.db``),
so passing ``--jsonl other.jsonl`` writes to ``other.db``. The SQLite index is the
source of truth for "what we already have": the newest stored commit date marks
where we left off.

Two operations
--------------
1. **sync** (default) — read the newest commit date already in the repo's
   SQLite db, blobless-clone, ``git log`` everything from that date up to today,
   APPEND the new commits to ``commits.jsonl`` (deduped by hash), and (unless
   ``--no-index``) embed + insert just those new commits into the db.

2. **migrate** (``--migrate``) — load an existing ``commits.jsonl`` into the
   repo's SQLite db (embedding any commits not already stored). Use this to
   bootstrap the db from a JSONL you already fetched, or to fold a JSONL grown
   by ``--no-index`` syncs into the index. Does not touch the network.

Usage examples
--------------
    # First time: bootstrap ebay-evo-web.db from ebay-evo-web.jsonl
    python sync_commits.py eBay/evo-web --migrate

    # Bring in everything new since the last stored commit, up to today
    python sync_commits.py eBay/evo-web

    # JSONL-only sync (no model load), index it later with --migrate
    python sync_commits.py eBay/evo-web --no-index

    # Custom locations / branch / first-pull start date
    python sync_commits.py eBay/evo-web --db-dir ./indexes \\
        --jsonl ebay.jsonl --branch main --since 2025-01-01
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import date, datetime, timedelta
from typing import List, Optional, Set

from commit_index import (
    CommitIndex,
    Commit,
    db_path_for_jsonl,
    load_commits_jsonl,
)
from fetch_commits import GitError, build_clone_url, clone_repo, read_commits

# Used as the window start for a first-ever pull when the db is empty and no
# --since was supplied. git accepts this and it simply means "all history".
DEFAULT_SINCE = "1970-01-01"

# When resuming from the last stored commit, step back this many days before
# querying git. git --since filters on *commit* date while we store *author*
# date, and offsets differ across commits; the small overlap (deduped by hash)
# guarantees we never skip a commit that straddles the boundary.
RESUME_BUFFER_DAYS = 2


# --------------------------------------------------------------------------- #
# Per-repo paths
# --------------------------------------------------------------------------- #
def repo_slug(repo: str) -> str:
    """Turn ``owner/name`` or a git URL into a lowercase, dash-joined slug.

    ``eBay/evo-web``                            -> ``ebay-evo-web``
    ``https://github.com/eBay/evo-web.git``     -> ``ebay-evo-web``
    ``git@github.com:eBay/evo-web.git``         -> ``ebay-evo-web``
    """
    s = repo.strip()
    s = re.sub(r"\.git$", "", s)
    s = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://[^/]+/", "", s)  # scheme://host/
    s = re.sub(r"^[^@/]+@[^:]+:", "", s)  # git@host:
    slug = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return slug or "repo"


def default_jsonl_path(repo: str, db_dir: str) -> str:
    """Default JSONL for a repo: ``<db_dir>/<slug>.jsonl`` (the db follows it)."""
    return os.path.join(db_dir, f"{repo_slug(repo)}.jsonl")


# --------------------------------------------------------------------------- #
# JSONL helpers
# --------------------------------------------------------------------------- #
def read_jsonl_hashes(path: str) -> Set[str]:
    """Collect the commit hashes already present in a JSONL file."""
    hashes: Set[str] = set()
    if not os.path.exists(path):
        return hashes
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                hashes.add(json.loads(line)["hash"])
            except (json.JSONDecodeError, KeyError):
                continue
    return hashes


def append_jsonl(path: str, commits: List[Commit]) -> None:
    """Append commits to a JSONL file (one JSON object per line)."""
    with open(path, "a", encoding="utf-8") as fh:
        for commit in commits:
            fh.write(json.dumps(commit, ensure_ascii=False))
            fh.write("\n")


# --------------------------------------------------------------------------- #
# Window resolution + fetch
# --------------------------------------------------------------------------- #
def resume_since(latest_iso: str, buffer_days: int = RESUME_BUFFER_DAYS) -> str:
    """Window start = the last stored date minus a small safety buffer."""
    try:
        dt = datetime.fromisoformat(latest_iso)
    except ValueError:
        return latest_iso
    return (dt - timedelta(days=buffer_days)).date().isoformat()


def fetch_window(
    repo: str, since: str, until: str, branch: Optional[str]
) -> List[Commit]:
    """Blobless-clone ``repo`` into a temp dir and ``git log`` the window."""
    token = os.environ.get("GITHUB_TOKEN") or None
    url = build_clone_url(repo)
    with tempfile.TemporaryDirectory(prefix="sync_commits_") as tmpdir:
        clone_repo(url, tmpdir, token)
        return read_commits(tmpdir, since, until, branch)


# --------------------------------------------------------------------------- #
# Operations
# --------------------------------------------------------------------------- #
def sync(
    repo: str,
    db_path: str,
    jsonl: str,
    branch: Optional[str],
    since_override: Optional[str],
    do_index: bool,
) -> int:
    """Fetch new commits since the db's newest, append to JSONL, index them."""
    index = CommitIndex(db_path)

    latest = index.latest_commit_date()
    if since_override:
        since = since_override
        start_note = f"--since override ({since})"
    elif latest:
        since = resume_since(latest)
        start_note = f"resuming after last stored commit {latest}"
    else:
        since = DEFAULT_SINCE
        start_note = "empty db -> full history"

    # +1 day so commits made earlier *today* are included (git --until is a bound).
    until = (date.today() + timedelta(days=1)).isoformat()

    print(f"Repo:   {repo}")
    print(f"DB:     {db_path}")
    print(f"JSONL:  {jsonl}")
    print(f"Window: {since} .. today   ({start_note})")

    try:
        fetched = fetch_window(repo, since, until, branch)
    except GitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # New-to-JSONL (dedup against what the file already holds).
    seen = read_jsonl_hashes(jsonl)
    new_for_jsonl = [c for c in fetched if c["hash"] not in seen]
    append_jsonl(jsonl, new_for_jsonl)

    # New-to-DB: index.add() dedups against the db itself and embeds only the
    # genuinely new commits (and won't load the model if there are none).
    added_to_db = 0
    if do_index:
        added_to_db = index.add(fetched)

    print(
        f"\nFetched {len(fetched)} commit(s) in window; "
        f"{len(new_for_jsonl)} new to {jsonl}, "
        f"{added_to_db} new to index"
        + ("" if do_index else " (indexing skipped: --no-index)")
    )
    if do_index and added_to_db:
        print(f"New newest stored commit: {index.latest_commit_date()}")
    return 0


def migrate(db_path: str, jsonl: str) -> int:
    """Load an existing JSONL into its SQLite db (embed new commits)."""
    if not os.path.exists(jsonl):
        print(f"error: {jsonl} not found", file=sys.stderr)
        return 1
    commits = load_commits_jsonl(jsonl)
    index = CommitIndex(db_path)

    print(f"Migrating {len(commits)} commit(s) from {jsonl} -> {db_path} ...")
    added = index.add(commits)
    print(
        f"Inserted {added} new commit(s) "
        f"({len(commits) - added} already present)."
    )
    newest = index.latest_commit_date()
    if newest:
        print(f"Newest stored commit: {newest}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
class _HelpfulParser(argparse.ArgumentParser):
    """ArgumentParser that prints FULL help (not a one-line error) on misuse."""

    def error(self, message: str) -> "NoReturn":  # type: ignore[name-defined]
        sys.stderr.write(f"error: {message}\n\n")
        self.print_help(sys.stderr)
        raise SystemExit(2)


_EXAMPLES = """\
examples:
  # First time: bootstrap ebay-evo-web.db from ebay-evo-web.jsonl
  python sync_commits.py eBay/evo-web --migrate

  # Bring in everything new since the last stored commit, up to today
  python sync_commits.py eBay/evo-web

  # JSONL-only sync (no model load); index it later with --migrate
  python sync_commits.py eBay/evo-web --no-index

  # Custom locations / branch / first-pull start date
  python sync_commits.py eBay/evo-web --db-dir ./indexes \\
      --jsonl ebay.jsonl --branch main --since 2025-01-01
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = _HelpfulParser(
        prog="sync_commits.py",
        description=(
            "Incrementally sync a repo's commits into a per-repo SQLite index "
            "and a JSONL log."
        ),
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("repo", help="owner/name shorthand or a full git URL.")
    parser.add_argument(
        "--db-dir",
        default="data",
        help="Directory for the per-repo JSONL + SQLite files (default: data/).",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="JSONL log to append to / migrate from. Default: "
        "<db-dir>/<repo-slug>.jsonl (e.g. ebay-evo-web.jsonl). The matching "
        "db is the same path with a .db extension.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Branch to scan; defaults to the repo's default branch (HEAD).",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Override the window start (otherwise: resume from the db, or "
        "full history on first pull).",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Load --jsonl into the repo's SQLite db and exit (no fetching).",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="During sync, only append to JSONL; do not embed/insert into the "
        "db (skips loading the model).",
    )
    args = parser.parse_args(argv)

    os.makedirs(args.db_dir, exist_ok=True)

    # The JSONL filename is the hint that names the db: <stem>.jsonl -> <stem>.db.
    # Default the JSONL itself from the repo slug when not given explicitly.
    jsonl = args.jsonl or default_jsonl_path(args.repo, args.db_dir)
    db_path = db_path_for_jsonl(jsonl)

    if args.migrate:
        return migrate(db_path, jsonl)
    return sync(
        repo=args.repo,
        db_path=db_path,
        jsonl=jsonl,
        branch=args.branch,
        since_override=args.since,
        do_index=not args.no_index,
    )


if __name__ == "__main__":
    raise SystemExit(main())
