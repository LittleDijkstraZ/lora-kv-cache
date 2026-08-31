#!/usr/bin/env python3
"""Convert LongHealth benchmark JSON into the generative-QA JSONL format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "longhealth"
SOURCE = DATA_DIR / "benchmark_v5.json"
OUTPUT = DATA_DIR / "longhealth.jsonl"


def _sorted_text_items(texts: dict[str, str]) -> list[str]:
    def _key(name: str) -> tuple[int, str]:
        suffix = name.split("_")[-1]
        return (int(suffix) if suffix.isdigit() else 0, name)

    return [str(texts[name]).strip() for name in sorted(texts, key=_key)]


def _options_for_question(question: dict) -> dict[str, str]:
    return {
        "A": str(question.get("answer_a", "")).strip(),
        "B": str(question.get("answer_b", "")).strip(),
        "C": str(question.get("answer_c", "")).strip(),
        "D": str(question.get("answer_d", "")).strip(),
        "E": str(question.get("answer_e", "")).strip(),
    }


def _validate_sampling_args(
    *,
    doc_stride: int,
    max_docs: int | None,
    question_stride: int,
    max_questions_per_doc: int | None,
    max_records: int | None,
) -> None:
    for name, value in (
        ("doc_stride", doc_stride),
        ("question_stride", question_stride),
    ):
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}")
    for name, value in (
        ("max_docs", max_docs),
        ("max_questions_per_doc", max_questions_per_doc),
        ("max_records", max_records),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{name} must be >= 1 when set, got {value}")


def records_from_benchmark(
    benchmark: dict,
    *,
    doc_stride: int = 1,
    max_docs: int | None = None,
    question_stride: int = 1,
    max_questions_per_doc: int | None = None,
    max_records: int | None = None,
) -> list[dict]:
    _validate_sampling_args(
        doc_stride=doc_stride,
        max_docs=max_docs,
        question_stride=question_stride,
        max_questions_per_doc=max_questions_per_doc,
        max_records=max_records,
    )

    records: list[dict] = []
    patient_ids = sorted(benchmark)[::doc_stride]
    if max_docs is not None:
        patient_ids = patient_ids[:max_docs]

    for patient_id in patient_ids:
        patient = benchmark[patient_id]
        context = "\n\n".join(_sorted_text_items(patient.get("texts", {}))).strip()
        diagnosis = patient.get("diagnosis")
        questions = sorted(
            patient.get("questions", []),
            key=lambda item: int(item.get("No", 0)),
        )
        questions = questions[::question_stride]
        if max_questions_per_doc is not None:
            questions = questions[:max_questions_per_doc]

        for q in questions:
            q_no = int(q.get("No", 0))
            correct_answer = str(q.get("correct", "")).strip()
            if not correct_answer:
                continue

            options = _options_for_question(q)
            metadata = {
                "document_id": patient_id,
                "patient_id": patient_id,
                "question_no": q_no,
                "diagnosis": diagnosis,
                "name": patient.get("name"),
                "birthday": patient.get("birthday"),
                "options": options,
                "answer_text": correct_answer,
            }

            row_id = f"{patient_id}_q{q_no:02d}_{len(records) + 1:04d}"
            row = {
                "_id": row_id,
                "context": context,
                "question": str(q["question"]).strip(),
                "answers": [correct_answer],
                "metadata": metadata,
            }
            records.append(row)
            if max_records is not None and len(records) >= max_records:
                return records
    return records


def _write_jsonl(records: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as fh:
        for row in records:
            fh.write(json.dumps(row) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=1,
        help="Keep every Nth patient/document after sorting patient ids.",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Maximum number of sampled patients/documents to keep.",
    )
    parser.add_argument(
        "--question_stride",
        type=int,
        default=1,
        help="Keep every Nth question per sampled patient after sorting by No.",
    )
    parser.add_argument(
        "--max_questions_per_doc",
        type=int,
        default=None,
        help="Maximum sampled questions to keep per patient/document.",
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum total output rows to keep after document/question sampling.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    source = args.source
    if not source.exists():
        raise FileNotFoundError(f"LongHealth source file not found: {source}")

    benchmark = json.loads(source.read_text())
    sampling_kwargs = {
        "doc_stride": args.doc_stride,
        "max_docs": args.max_docs,
        "question_stride": args.question_stride,
        "max_questions_per_doc": args.max_questions_per_doc,
        "max_records": args.max_records,
    }
    qa_records = records_from_benchmark(benchmark, **sampling_kwargs)
    _write_jsonl(qa_records, args.output)
    print(f"Wrote {len(qa_records)} rows -> {args.output}")


if __name__ == "__main__":
    main()
