#!/usr/bin/env python3
"""Build a generative-QA JSONL from full NarrativeQA stories.

The converter intentionally uses full story text, not Wikipedia summaries and
not retrieval snippets. Optional length filters only decide which complete
documents are eligible for the sampled subset; they never trim the selected
story text.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "narrativeqa"
DEFAULT_OUTPUT = DATA_DIR / "narrativeqa.jsonl"
DEFAULT_STORIES_DIR = DATA_DIR / "stories"
RAW_BASE_URL = "https://raw.githubusercontent.com/google-deepmind/narrativeqa/master"
DEFAULT_DOCUMENTS_CSV = f"{RAW_BASE_URL}/documents.csv"
DEFAULT_QAPS_CSV = f"{RAW_BASE_URL}/qaps.csv"


def _read_text_source(source: str | Path, *, timeout: int = 30) -> str:
    source_str = str(source)
    path = Path(source_str)
    if path.exists():
        return path.read_text(encoding="utf-8")

    if "://" not in source_str:
        raise FileNotFoundError(f"CSV source not found: {source}")

    with urllib.request.urlopen(source_str, timeout=timeout) as response:
        return response.read().decode("utf-8")


def load_csv_rows(source: str | Path, *, timeout: int = 30) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(_read_text_source(source, timeout=timeout))))


def group_qaps_by_doc(qaps_rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in qaps_rows:
        grouped[str(row["document_id"])].append(row)
    return dict(grouped)


def decode_story_bytes(raw: bytes) -> str:
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_story_span(text: str, story_start: str | None, story_end: str | None) -> str:
    """Extract the official story span when NarrativeQA markers are present."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    start = (story_start or "").strip()
    end = (story_end or "").strip()

    start_idx = text.find(start) if start else -1
    if start_idx >= 0:
        text = text[start_idx:]

    end_idx = text.find(end) if end else -1
    if end_idx >= 0:
        text = text[: end_idx + len(end)]

    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _download_bytes(url: str, *, timeout: int, retries: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "lora-kv-cache/1.0"})
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - report original download failures.
            last_exc = exc
            if attempt < retries:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"Could not download {url}: {last_exc}") from last_exc


def load_story_text(
    document: dict[str, str],
    *,
    stories_dir: Path,
    download: bool,
    timeout: int = 60,
    retries: int = 2,
) -> str:
    doc_id = str(document["document_id"])
    story_path = stories_dir / f"{doc_id}.content"
    if story_path.exists():
        raw = story_path.read_bytes()
    else:
        if not download:
            raise FileNotFoundError(f"Story file missing and --no_download set: {story_path}")
        stories_dir.mkdir(parents=True, exist_ok=True)
        raw = _download_bytes(str(document["story_url"]), timeout=timeout, retries=retries)
        story_path.write_bytes(raw)

    return extract_story_span(
        decode_story_bytes(raw),
        document.get("story_start"),
        document.get("story_end"),
    )


def document_is_eligible(
    *,
    story_text: str,
    min_story_words: int | None,
    max_story_words: int | None,
) -> tuple[bool, dict[str, Any]]:
    words = len(story_text.split())
    stats: dict[str, Any] = {"story_words": words}
    if min_story_words is not None and words < min_story_words:
        stats["reason"] = f"story_words<{min_story_words}"
        return False, stats
    if max_story_words is not None and words > max_story_words:
        stats["reason"] = f"story_words>{max_story_words}"
        return False, stats

    return True, stats


def records_for_document(
    *,
    document: dict[str, str],
    story_text: str,
    qaps: list[dict[str, str]],
    subset_seed: int,
    subset_doc_rank: int,
    doc_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    doc_id = str(document["document_id"])
    records = []
    for question_idx, row in enumerate(qaps):
        answers = [
            str(row[key]).strip()
            for key in ("answer1", "answer2")
            if str(row.get(key, "")).strip()
        ]
        records.append(
            {
                "_id": f"narrativeqa_{doc_id}_q{question_idx:03d}",
                "context": story_text,
                "question": str(row["question"]).strip(),
                "answers": answers,
                "metadata": {
                    "document_id": doc_id,
                    "set": document.get("set"),
                    "kind": document.get("kind"),
                    "wiki_title": document.get("wiki_title"),
                    "story_url": document.get("story_url"),
                    "story_word_count_official": document.get("story_word_count"),
                    "subset_seed": subset_seed,
                    "subset_doc_rank": subset_doc_rank,
                    **doc_stats,
                },
            }
        )
    return records


def build_subset_records(
    *,
    documents: list[dict[str, str]],
    qaps_by_doc: dict[str, list[dict[str, str]]],
    stories_dir: Path,
    num_docs: int,
    seed: int,
    split: str,
    download: bool,
    timeout: int,
    retries: int,
    min_story_words: int | None,
    max_story_words: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [
        doc
        for doc in documents
        if (split == "all" or doc.get("set") == split)
        and str(doc.get("document_id")) in qaps_by_doc
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)

    selected_records: list[dict[str, Any]] = []
    selected_docs: list[dict[str, Any]] = []
    skipped = 0

    for doc in candidates:
        doc_id = str(doc["document_id"])
        try:
            story_text = load_story_text(
                doc,
                stories_dir=stories_dir,
                download=download,
                timeout=timeout,
                retries=retries,
            )
            if not story_text:
                raise ValueError("empty story text after marker extraction")
            qaps = qaps_by_doc[doc_id]
            eligible, stats = document_is_eligible(
                story_text=story_text,
                min_story_words=min_story_words,
                max_story_words=max_story_words,
            )
            if not eligible:
                skipped += 1
                continue
        except Exception as exc:  # noqa: BLE001 - keep sampling other docs.
            skipped += 1
            print(f"[convert_narrativeqa] Skipping {doc_id}: {exc}")
            continue

        doc_rank = len(selected_docs)
        selected_records.extend(
            records_for_document(
                document=doc,
                story_text=story_text,
                qaps=qaps,
                subset_seed=seed,
                subset_doc_rank=doc_rank,
                doc_stats=stats,
            )
        )
        selected_docs.append(
            {
                "document_id": doc_id,
                "set": doc.get("set"),
                "kind": doc.get("kind"),
                "wiki_title": doc.get("wiki_title"),
                "num_qas": len(qaps),
                **stats,
            }
        )
        print(
            f"[convert_narrativeqa] Selected {len(selected_docs)}/{num_docs}: "
            f"{doc_id} ({len(qaps)} QA, {stats.get('story_words')} words)"
        )
        if len(selected_docs) >= num_docs:
            break

    if len(selected_docs) < num_docs:
        raise RuntimeError(
            f"Only selected {len(selected_docs)} docs out of requested {num_docs} "
            f"(skipped {skipped}). Try increasing max_docs_to_try, relaxing length "
            "filters, or pre-downloading more stories."
        )

    return selected_records, selected_docs


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download/read NarrativeQA full stories and build a seeded 20-doc "
            "generative-QA evaluation JSONL."
        )
    )
    parser.add_argument("--documents_csv", default=DEFAULT_DOCUMENTS_CSV)
    parser.add_argument("--qaps_csv", default=DEFAULT_QAPS_CSV)
    parser.add_argument("--stories_dir", type=Path, default=DEFAULT_STORIES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num_docs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split",
        choices=["all", "train", "valid", "test"],
        default="all",
        help="Document split to sample from. Default all matches the 20-doc subset use case.",
    )
    parser.add_argument("--min_story_words", type=int, default=None)
    parser.add_argument("--max_story_words", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    download_group = parser.add_mutually_exclusive_group()
    download_group.add_argument("--download", dest="download", action="store_true")
    download_group.add_argument("--no_download", dest="download", action="store_false")
    parser.set_defaults(download=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    documents = load_csv_rows(args.documents_csv, timeout=args.timeout)
    qaps = load_csv_rows(args.qaps_csv, timeout=args.timeout)
    qaps_by_doc = group_qaps_by_doc(qaps)
    records, selected_docs = build_subset_records(
        documents=documents,
        qaps_by_doc=qaps_by_doc,
        stories_dir=args.stories_dir,
        num_docs=args.num_docs,
        seed=args.seed,
        split=args.split,
        download=bool(args.download),
        timeout=args.timeout,
        retries=args.retries,
        min_story_words=args.min_story_words,
        max_story_words=args.max_story_words,
    )
    write_jsonl(args.output, records)
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "num_docs": len(selected_docs),
                "num_qas": len(records),
                "seed": args.seed,
                "split": args.split,
                "min_story_words": args.min_story_words,
                "max_story_words": args.max_story_words,
                "selected_docs": selected_docs,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"[convert_narrativeqa] Wrote {len(records)} QA from {len(selected_docs)} docs -> "
        f"{args.output}"
    )
    print(f"[convert_narrativeqa] Metadata -> {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
