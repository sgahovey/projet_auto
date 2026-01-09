#!/usr/bin/env python3
"""Generate Kanban artifacts from the latest git commit."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().with_name("kanban.config.json")
CARDS_PATH = REPO_ROOT / "kanban_cards.json"
CSV_PATH = REPO_ROOT / "kanban_export.csv"


CONVENTIONAL_RE = re.compile(
    r"^(?P<type>[a-z]+)(\((?P<scope>[^)]+)\))?(?P<breaking>!)?: (?P<subject>.+)$"
)


def run_git_command(args: List[str]) -> str:
    """Run a git command and return its stdout."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(f"[kanban] git {' '.join(args)} failed: {exc.stderr.strip()}\n")
        raise
    return completed.stdout.rstrip("\n")


def load_config() -> Dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Cannot find config file at {CONFIG_PATH}")
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_conventional(subject: str) -> Tuple[str, str | None, str]:
    """Parse a conventional commit subject line."""
    match = CONVENTIONAL_RE.match(subject)
    if not match:
        return "chore", None, subject
    commit_type = match.group("type")
    scope = match.group("scope")
    message = match.group("subject").strip()
    if match.group("breaking"):
        message = f"{message} (breaking)"
    return commit_type, scope, message


def format_description(
    commit_hash: str,
    commit_type: str,
    scope: str | None,
    message: str,
    body: str,
    files: List[str],
) -> str:
    """Compose the description text for the Kanban card."""
    conventional = f"{commit_type}"
    if scope:
        conventional += f"({scope})"
    conventional += f": {message}"

    lines = [
        f"Commit: {commit_hash}",
        f"Conventional: {conventional}",
    ]

    cleaned_body = body.strip()
    if cleaned_body:
        lines.append("Body:")
        lines.append(cleaned_body)
    else:
        lines.append("Body: (none)")

    if files:
        lines.append("Files:")
        lines.extend(f"- {path}" for path in files)
    else:
        lines.append("Files: (none)")

    return "\n".join(lines)


def ensure_unique(values: List[str]) -> List[str]:
    seen = set()
    unique_values: List[str] = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def load_existing_cards() -> List[Dict]:
    if not CARDS_PATH.exists():
        return []
    try:
        with CARDS_PATH.open(encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, list):
                return data
    except json.JSONDecodeError:
        sys.stderr.write("[kanban] Unable to parse existing kanban_cards.json, starting fresh\n")
    return []


def save_cards(cards: List[Dict]) -> None:
    with CARDS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(cards, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def export_csv(cards: List[Dict]) -> None:
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Title", "Description", "Labels", "Start Date", "Due Date", "DoD"]
        )
        for card in cards:
            writer.writerow(
                [
                    card.get("title", ""),
                    card.get("description", ""),
                    "; ".join(card.get("labels", [])),
                    card.get("start_date", ""),
                    card.get("due_date", ""),
                    "; ".join(card.get("dod", [])),
                ]
            )


def derive_labels(config: Dict, commit_type: str) -> List[str]:
    labels = list(config.get("default_labels", []))
    mapping = config.get("labels_map", {})
    labels.extend(mapping.get(commit_type, []))
    if not mapping.get(commit_type):
        # Provide a catch-all label when no specific mapping exists.
        labels.extend(mapping.get("chore", []))
    return ensure_unique(labels)


def determine_dod(config: Dict, commit_type: str) -> List[str]:
    templates = config.get("dod_templates", {})
    if commit_type in templates:
        return templates[commit_type][:6]
    default = templates.get("default", [])
    return default[:6]


def main() -> int:
    try:
        config = load_config()
        log_output = run_git_command(
            ["log", "-1", "--date=short", "--pretty=format:%H%x1f%cd%x1f%s%x1f%b", "HEAD"]
        )
        parts = log_output.split("\x1f", 3)
        if len(parts) != 4:
            raise ValueError("Unexpected git log output")
        commit_hash, commit_date, subject, body = parts[0], parts[1], parts[2], parts[3]
        files_output = run_git_command(
            ["show", "--name-only", "--pretty=format:", "--no-renames", "HEAD"]
        )
        files = [line.strip() for line in files_output.splitlines() if line.strip()]
        commit_type, scope, message = parse_conventional(subject)
        description = format_description(
            commit_hash=commit_hash,
            commit_type=commit_type,
            scope=scope,
            message=message,
            body=body,
            files=files,
        )
        labels = derive_labels(config, commit_type)
        dod = determine_dod(config, commit_type)
        card = {
            "title": message or subject or commit_hash[:7],
            "description": description,
            "labels": labels,
            "start_date": commit_date,
            "due_date": commit_date,
            "dod": dod,
        }

        cards = load_existing_cards()
        cards.append(card)
        save_cards(cards)
        export_csv(cards)
    except Exception as exc:
        sys.stderr.write(f"[kanban] Generation failed: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
