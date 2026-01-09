#!/usr/bin/env python3
"""Generate Kanban artifacts from the latest git commit."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().with_name("kanban.config.json")
CARDS_PATH = REPO_ROOT / "kanban_cards.json"
CSV_PATH = REPO_ROOT / "kanban_export.csv"


CONVENTIONAL_RE = re.compile(
    r"^(?P<type>[a-z]+)"
    r"(?:\((?P<scope>[^)]+)\))?"
    r"(?:\s*\[(?P<difficulty>Diff-[1-5])\])?"
    r"(?P<breaking>!)?: "
    r"(?P<subject>.+)$"
)
TYPE_OVERRIDE_RE = re.compile(
    r"\[type\s*=\s*(?P<type>feat|fix|ci|test|docs|refactor|chore|deploy)\]", re.IGNORECASE
)
SCOPE_OVERRIDE_RE = re.compile(
    r"\[scope\s*=\s*(?P<scope>frontend|backend|docker|ansible|monitoring|init|github-actions|prod|none)\]",
    re.IGNORECASE,
)
DIFF_OVERRIDE_RE = re.compile(r"\[diff-(?P<level>[1-5])\]", re.IGNORECASE)

FRONTEND_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".css", ".scss", ".html", ".htm", ".vue"}
BACKEND_EXTENSIONS = {".py", ".php", ".java", ".rb", ".go", ".cs", ".ts", ".rs"}
TEST_KEYWORDS = [
    "tests/",
    "__tests__",
    "_test.",
    "test_",
    ".spec.",
    ".test.",
    "test/",
]
DEVOPS_SCOPES = {"docker", "ansible", "monitoring", "github-actions", "prod"}


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


def normalize_override_type(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"feat", "fix", "ci", "test", "docs", "refactor", "chore", "deploy"}:
        return lowered
    return None


def normalize_override_scope(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    lowered = value.lower()
    if lowered == "none":
        return "none"
    if lowered in {
        "frontend",
        "backend",
        "docker",
        "ansible",
        "monitoring",
        "init",
        "github-actions",
        "prod",
    }:
        return lowered
    return None


def normalize_path(path: str) -> str:
    if " -> " in path:
        path = path.split(" -> ", 1)[-1]
    return path.strip()


def extract_overrides(subject: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    override_type: Optional[str] = None
    override_scope: Optional[str] = None
    override_diff: Optional[str] = None
    cleaned = subject

    match = CONVENTIONAL_RE.match(subject)
    if match:
        override_type = match.group("type")
        override_scope = match.group("scope")
        if match.group("difficulty"):
            override_diff = match.group("difficulty")
        cleaned = match.group("subject").strip()
        if match.group("breaking"):
            cleaned = f"{cleaned} (breaking)"
    else:
        cleaned = subject

    for found in TYPE_OVERRIDE_RE.finditer(subject):
        override_type = found.group("type")
    for found in SCOPE_OVERRIDE_RE.finditer(subject):
        override_scope = found.group("scope")
    diff_match = None
    for found in DIFF_OVERRIDE_RE.finditer(subject):
        diff_match = found
    if diff_match:
        override_diff = f"Diff-{diff_match.group('level')}"

    cleaned = TYPE_OVERRIDE_RE.sub("", cleaned)
    cleaned = SCOPE_OVERRIDE_RE.sub("", cleaned)
    cleaned = DIFF_OVERRIDE_RE.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -:")

    override_type = normalize_override_type(override_type)
    override_scope = normalize_override_scope(override_scope)

    return cleaned or subject.strip(), override_type, override_scope, override_diff


def format_description(
    commit_hash: str,
    original_subject: str,
    type_label: str,
    scope_label: Optional[str],
    difficulty_label: str,
    body: str,
    files: List[str],
) -> str:
    """Compose the description text for the Kanban card."""
    lines = [
        f"Commit: {commit_hash}",
        f"Subject: {original_subject}",
        f"Type: {type_label}",
        f"Scope: {scope_label or 'None'}",
        f"Difficulty: {difficulty_label}",
    ]

    cleaned_body = body.strip()
    if cleaned_body:
        lines.append("Body:")
        lines.append(cleaned_body)
    else:
        lines.append("Body: (none)")

    if files:
        lines.append("Files:")
        lines.extend(f"* {path}" for path in files)
    else:
        lines.append("Files: (none)")

    return "\n".join(lines)


def detect_scopes_for_path(path: str) -> List[str]:
    scopes: List[str] = []
    lower = path.lower()
    suffix = Path(lower).suffix

    if lower.startswith("frontend/") or suffix in FRONTEND_EXTENSIONS:
        scopes.append("frontend")
    if lower.startswith("backend/") or lower.startswith("src/") or suffix in BACKEND_EXTENSIONS:
        scopes.append("backend")
    if (
        lower == "dockerfile"
        or lower.endswith("/dockerfile")
        or lower.endswith("docker-compose.yml")
        or "/docker/" in lower
        or "/.docker/" in lower
    ):
        scopes.append("docker")
    if lower.startswith("ansible/") or ("playbook" in lower and lower.endswith(".yml")):
        scopes.append("ansible")
    if (
        lower.startswith("monitoring/")
        or "/grafana/" in lower
        or lower.startswith("grafana/")
        or "/prometheus/" in lower
        or lower.startswith("prometheus/")
    ):
        scopes.append("monitoring")
    if lower.startswith(".github/workflows/"):
        scopes.append("github-actions")
    if lower.startswith("prod/") or "deploy" in lower or "production" in lower:
        scopes.append("prod")
    if lower.startswith("init/"):
        scopes.append("init")

    return scopes


def analyze_commit_diff(output: str) -> Tuple[List[str], Counter, Dict[str, int]]:
    files: List[str] = []
    scope_weights: Counter = Counter()
    stats: Dict[str, int] = {
        "has_ci": 0,
        "has_docs": 0,
        "has_tests": 0,
        "rename_count": 0,
        "additions": 0,
        "deletions": 0,
    }

    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        add_str, del_str = parts[0], parts[1]
        raw_path = parts[2] if len(parts) == 3 else " -> ".join(parts[2:])
        path = normalize_path(raw_path)
        files.append(path)

        additions = int(add_str) if add_str.isdigit() else 0
        deletions = int(del_str) if del_str.isdigit() else 0
        stats["additions"] += additions
        stats["deletions"] += deletions
        weight = additions + deletions
        if weight <= 0:
            weight = 1

        lower = path.lower()
        if "/.github/workflows" in lower or lower.startswith(".github/workflows/"):
            stats["has_ci"] = 1
        if lower == "readme.md" or lower.startswith("docs/"):
            stats["has_docs"] = 1
        if any(keyword in lower for keyword in TEST_KEYWORDS):
            stats["has_tests"] = 1
        if len(parts) > 3 or " -> " in raw_path:
            stats["rename_count"] += 1

        for scope in detect_scopes_for_path(path):
            scope_weights[scope] += weight

    return files, scope_weights, stats


def determine_scope(
    scope_weights: Counter,
    message: str,
    override_scope: Optional[str],
) -> Optional[str]:
    if override_scope is not None:
        if override_scope == "none":
            return None
        return override_scope

    if scope_weights:
        max_weight = max(scope_weights.values())
        candidates = [scope for scope, weight in scope_weights.items() if weight == max_weight]
        priority = [
            "backend",
            "frontend",
            "docker",
            "ansible",
            "monitoring",
            "github-actions",
            "prod",
            "init",
        ]
        for preferred in priority:
            if preferred in candidates:
                return preferred
        return candidates[0]

    lowered = message.lower()
    keyword_scope_map = [
        ("frontend", "frontend"),
        ("backend", "backend"),
        ("docker", "docker"),
        ("ansible", "ansible"),
        ("monitor", "monitoring"),
        ("grafana", "monitoring"),
        ("prometheus", "monitoring"),
        ("github", "github-actions"),
        ("gha", "github-actions"),
        ("prod", "prod"),
        ("production", "prod"),
        ("deploy", "prod"),
        ("init", "init"),
        ("initial", "init"),
    ]
    for keyword, scope in keyword_scope_map:
        if keyword in lowered:
            return scope
    return None


def infer_type(
    message: str,
    override_type: Optional[str],
    scope: Optional[str],
    stats: Dict[str, int],
    total_changes: int,
) -> str:
    if override_type:
        return override_type

    lowered = message.lower()

    if stats.get("has_ci"):
        return "ci"
    if stats.get("has_docs"):
        return "docs"
    if stats.get("has_tests"):
        return "test"
    if scope == "prod" or any(token in lowered for token in ("deploy", "déploi", "production", "prod")):
        return "deploy"
    if any(token in lowered for token in ("fix", "bug", "corrige", "corrigé", "corrigée")) and total_changes <= 80:
        return "fix"
    if any(token in lowered for token in ("refactor", "refacto", "restruct", "restructuration")) or stats.get(
        "rename_count", 0
    ) > 0:
        return "refactor"
    if any(token in lowered for token in ("chore", "maintenance", "maintain", "nettoyage")):
        return "chore"
    return "feat"


def estimate_difficulty(
    total_changes: int,
    override_difficulty: Optional[str],
    commit_type: str,
    scope: Optional[str],
) -> str:
    if override_difficulty:
        return override_difficulty

    if total_changes <= 20:
        level = 1
    elif total_changes <= 80:
        level = 2
    elif total_changes <= 200:
        level = 3
    elif total_changes <= 500:
        level = 4
    else:
        level = 5

    if total_changes >= 80 and (commit_type in {"ci", "deploy"} or scope == "monitoring"):
        level = min(5, level + 1)

    return f"Diff-{level}"


def type_label_from_config(config: Dict, commit_type: str) -> str:
    labels_map = config.get("labels_map", {})
    labels = labels_map.get(commit_type)
    if labels:
        return labels[0]
    return commit_type.capitalize()


def scope_label_from_config(config: Dict, scope: Optional[str]) -> Optional[str]:
    if not scope:
        return None
    scope_map = config.get("scope_labels_map", {})
    labels = scope_map.get(scope)
    if labels:
        return labels[0]
    return scope.capitalize()


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
                normalized: List[Dict] = []
                for item in data:
                    if isinstance(item, dict):
                        description = item.get("description")
                        if isinstance(description, str):
                            item["description"] = description.replace("\n- ", "\n* ")
                        normalized.append(item)
                return normalized
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


def derive_labels(
    config: Dict, commit_type: str, scope: Optional[str], difficulty: str
) -> List[str]:
    labels = list(config.get("default_labels", []))
    mapping = config.get("labels_map", {})
    type_labels = mapping.get(commit_type)
    if type_labels:
        labels.extend(type_labels)
    else:
        labels.append(commit_type.capitalize())
    if scope:
        scope_labels = config.get("scope_labels_map", {})
        labels.extend(scope_labels.get(scope, [scope.capitalize()]))
    difficulty_labels = config.get("difficulty_labels_map", {})
    labels.extend(difficulty_labels.get(difficulty, [difficulty]))
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

        numstat_output = run_git_command(["show", "--numstat", "--pretty=format:", "HEAD"])
        files, scope_weights, stats = analyze_commit_diff(numstat_output)
        if not files:
            files_output = run_git_command(["show", "--name-only", "--pretty=format:", "HEAD"])
            files = [line.strip() for line in files_output.splitlines() if line.strip()]

        cleaned_subject, override_type, override_scope, override_diff = extract_overrides(subject)
        scope = determine_scope(scope_weights, cleaned_subject, override_scope)
        total_changes = stats.get("additions", 0) + stats.get("deletions", 0)
        commit_type = infer_type(cleaned_subject, override_type, scope, stats, total_changes)
        difficulty = estimate_difficulty(total_changes, override_diff, commit_type, scope)
        type_label = type_label_from_config(config, commit_type)
        scope_label = scope_label_from_config(config, scope)
        description = format_description(
            commit_hash=commit_hash,
            original_subject=subject,
            type_label=type_label,
            scope_label=scope_label,
            difficulty_label=difficulty,
            body=body,
            files=files,
        )
        labels = derive_labels(config, commit_type, scope, difficulty)
        dod = determine_dod(config, commit_type)
        card = {
            "title": cleaned_subject or subject or commit_hash[:7],
            "description": description,
            "labels": labels,
            "start_date": commit_date,
            "due_date": commit_date,
            "dod": dod,
        }

        cards = load_existing_cards()
        cards = [
            existing
            for existing in cards
            if not (
                isinstance(existing, dict)
                and isinstance(existing.get("description"), str)
                and f"Commit: {commit_hash}" in existing.get("description", "")
            )
        ]
        cards.append(card)
        save_cards(cards)
        export_csv(cards)
    except Exception as exc:
        sys.stderr.write(f"[kanban] Generation failed: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
