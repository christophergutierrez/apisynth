#!/usr/bin/env python3
"""Retrieval layer for code-path queries.

Builds a name→file index from source records and enriches prompts with
file-location context before they reach the LLM.  At production time the
index would come from a live code search (grep / tree-sitter); here we
build it from the same JSONL records the repo was generated from.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


BACKTICK_RE = re.compile(r"`([^`]+)`")
CLASS_CONTEXT_RE = re.compile(
    r"(?:on|of|on a|of a|on an|of an)\s+`([^`]+)`", re.IGNORECASE
)


def build_index(records: list[dict]) -> dict[str, dict[str, set[str]]]:
    """Build a lookup: name → {file → set of classes-or-'_'}.

    This captures both where a name lives and, for methods, which class
    it belongs to — so ambiguous names like __init__ can be narrowed by
    class context in the question.
    """
    index: dict[str, dict[str, set[str]]] = {}
    for r in records:
        out = r.get("output", {})
        name = out.get("name", "")
        file = out.get("file", "")
        cls = out.get("class", "")
        if not name or not file:
            continue
        entry = index.setdefault(name, {})
        entry.setdefault(file, set()).add(cls or "_")
    return index


def build_index_from_jsonl(*paths: Path) -> dict[str, dict[str, set[str]]]:
    records: list[dict] = []
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    return build_index(records)


def extract_identifiers(question: str) -> tuple[str, str | None]:
    """Return (target_name, class_context_or_None) from a question."""
    backticks = BACKTICK_RE.findall(question)
    if not backticks:
        return "", None

    class_match = CLASS_CONTEXT_RE.search(question)
    class_ctx = class_match.group(1) if class_match else None

    if class_ctx and class_ctx in backticks:
        candidates = [b for b in backticks if b != class_ctx]
        target = candidates[0] if candidates else backticks[-1]
    else:
        target = backticks[-1]

    # Handle Class.method syntax (e.g. `VaultClient.__init__`)
    if "." in target and class_ctx is None:
        parts = target.rsplit(".", 1)
        class_ctx = parts[0]
        target = parts[1]

    return target, class_ctx


def search_index(
    index: dict[str, dict[str, set[str]]],
    name: str,
    class_ctx: str | None = None,
) -> list[str]:
    """Return matching file paths, narrowed by class context if available."""
    entries = index.get(name)
    if not entries:
        return []

    if class_ctx:
        narrowed = [f for f, classes in entries.items() if class_ctx in classes]
        if narrowed:
            return sorted(narrowed)

    return sorted(entries.keys())


def retrieval_context(files: list[str], name: str) -> str:
    """Format a retrieval context block to inject into the prompt."""
    if not files:
        return ""
    if len(files) == 1:
        return f"[Code search: `{name}` found in {files[0]}]"
    listing = ", ".join(files)
    return f"[Code search: `{name}` found in {listing}]"


def enrich_messages(
    messages: list[dict],
    index: dict[str, dict[str, set[str]]],
) -> list[dict]:
    """Return a copy of messages with retrieval context prepended to the
    user turn."""
    enriched = []
    for msg in messages:
        if msg["role"] == "user":
            question = msg["content"]
            name, class_ctx = extract_identifiers(question)
            files = search_index(index, name, class_ctx)
            ctx = retrieval_context(files, name)
            if ctx:
                enriched.append({
                    "role": "user",
                    "content": f"{ctx}\n\n{question}",
                })
            else:
                enriched.append(msg)
        else:
            enriched.append(msg)
    return enriched
