#!/usr/bin/env python3
"""Reorganize local Zotero library collections and tags.

This script uses pyzotero against the local Zotero connector.
Run with --apply to persist changes; otherwise it performs a dry run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os

from pyzotero import zotero


COLLECTION_NAMES = [
    "00_Inbox",
    "01_MA Thesis",
    "02_Reading Queue",
]

TAG_RENAMES = {
    "Curriculum learning": "topic:curriculum-learning",
    "cl": "topic:curriculum-learning",
    "meta-learning": "topic:meta-learning",
    "Meta-learning": "topic:meta-learning",
    "mtl": "topic:multi-task-learning",
    "Optimization": "topic:optimization",
    "RL": "topic:reinforcement-learning",
    "NLP": "topic:nlp",
    "nlp": "topic:nlp",
}

DROP_TAGS = {
    "inbox",
    "ma",
    "magisterka",
}

STATUS_TAGS = {
    "status:to-read",
    "status:reading",
    "status:summarized",
}


@dataclass
class ChangeStats:
    collection_creates: int = 0
    item_updates: int = 0
    tags_changed: int = 0
    collections_assigned: int = 0
    status_added: int = 0


def _client(
    locale: str,
    local: bool,
    library_id: str,
    library_type: str,
    api_key: str | None,
) -> zotero.Zotero:
    return zotero.Zotero(
        library_id=library_id,
        library_type=library_type,
        api_key=api_key,
        local=local,
        locale=locale,
    )


def _get_or_create_collections(client: zotero.Zotero, apply: bool) -> tuple[dict[str, str], int]:
    existing = client.everything(client.collections())
    by_name: dict[str, str] = {c["data"]["name"]: c["data"]["key"] for c in existing}

    missing = [name for name in COLLECTION_NAMES if name not in by_name]
    if missing and apply:
        payload = [{"name": name} for name in missing]
        client.create_collections(payload)
        existing = client.everything(client.collections())
        by_name = {c["data"]["name"]: c["data"]["key"] for c in existing}

    if missing and not apply:
        for name in missing:
            by_name[name] = f"__DRYRUN__{name}"

    selected = {name: by_name[name] for name in COLLECTION_NAMES if name in by_name}
    return selected, len(missing)


def _normalize_tags(raw_tags: list[dict]) -> tuple[list[dict], int]:
    out: list[dict] = []
    seen: set[str] = set()
    changed = 0

    for tag_obj in raw_tags:
        old = (tag_obj.get("tag") or "").strip()
        if not old:
            continue

        if old in DROP_TAGS:
            changed += 1
            continue

        new = TAG_RENAMES.get(old, old)
        if new != old:
            changed += 1

        if new not in seen:
            out.append({"tag": new, "type": 1})
            seen.add(new)

    return out, changed


def reorganize(
    apply: bool,
    add_status_to_read: bool,
    locale: str,
    local: bool,
    library_id: str,
    library_type: str,
    api_key: str | None,
) -> ChangeStats:
    client = _client(locale, local, library_id, library_type, api_key)
    stats = ChangeStats()

    coll_map, missing_count = _get_or_create_collections(client, apply=apply)
    stats.collection_creates = missing_count

    inbox_key = coll_map.get("00_Inbox")
    thesis_key = coll_map.get("01_MA Thesis")

    items = client.everything(client.top())

    for item in items:
        data = item.get("data", {})
        item_type = data.get("itemType")
        if item_type in {"attachment", "note"}:
            continue

        old_tags = data.get("tags", [])
        old_tag_names = {t.get("tag", "") for t in old_tags}
        new_tags, tags_changed = _normalize_tags(old_tags)

        collections = list(data.get("collections", []))
        collections_set = set(collections)
        collections_added = 0

        if "inbox" in old_tag_names and inbox_key and inbox_key not in collections_set:
            collections.append(inbox_key)
            collections_set.add(inbox_key)
            collections_added += 1

        if (
            ({"ma", "magisterka"} & old_tag_names)
            and thesis_key
            and thesis_key not in collections_set
        ):
            collections.append(thesis_key)
            collections_set.add(thesis_key)
            collections_added += 1

        status_added = 0
        if add_status_to_read and thesis_key and thesis_key in collections_set:
            present = {t["tag"] for t in new_tags}
            if not (present & STATUS_TAGS):
                new_tags.append({"tag": "status:to-read", "type": 1})
                status_added = 1

        if tags_changed == 0 and collections_added == 0 and status_added == 0:
            continue

        data["tags"] = new_tags
        data["collections"] = collections

        stats.item_updates += 1
        stats.tags_changed += tags_changed
        stats.collections_assigned += collections_added
        stats.status_added += status_added

        if apply:
            client.update_item(item)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize local Zotero collections and tags")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist updates to Zotero. If omitted, runs dry-run only.",
    )
    parser.add_argument(
        "--add-status-to-read",
        action="store_true",
        help="Add status:to-read to MA Thesis items without status tags.",
    )
    parser.add_argument("--locale", default="en-US", help="Locale for Zotero API")
    parser.add_argument(
        "--local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use local Zotero connector (default: true).",
    )
    parser.add_argument(
        "--library-id",
        default=os.getenv("ZOTERO_LIBRARY_ID", "0"),
        help="Zotero library ID (default: env ZOTERO_LIBRARY_ID or 0).",
    )
    parser.add_argument(
        "--library-type",
        default=os.getenv("ZOTERO_LIBRARY_TYPE", "user"),
        choices=["user", "group"],
        help="Zotero library type (default: env ZOTERO_LIBRARY_TYPE or user).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ZOTERO_API_KEY"),
        help="Zotero API key (required for remote writes).",
    )
    args = parser.parse_args()

    if args.apply and args.local:
        raise SystemExit(
            "--apply with --local is not supported by Zotero connector API. "
            "Use remote Web API credentials (ZOTERO_LIBRARY_ID, "
            "ZOTERO_LIBRARY_TYPE, ZOTERO_API_KEY) and run with --no-local."
        )

    stats = reorganize(
        apply=args.apply,
        add_status_to_read=args.add_status_to_read,
        locale=args.locale,
        local=args.local,
        library_id=args.library_id,
        library_type=args.library_type,
        api_key=args.api_key,
    )

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Mode: {mode}")
    print(f"Missing collections (would create): {stats.collection_creates}")
    print(f"Items to update: {stats.item_updates}")
    print(f"Tag edits: {stats.tags_changed}")
    print(f"Collection assignments: {stats.collections_assigned}")
    print(f"status:to-read additions: {stats.status_added}")


if __name__ == "__main__":
    main()
