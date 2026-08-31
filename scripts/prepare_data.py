#!/usr/bin/env python3
"""Download and prepare the NarrativeQA and LongHealth evaluation subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


NARRATIVEQA_REVISION = "2e643e7363944af1c33a652d1c87320d0871c4e4"
NARRATIVEQA_FILES = {
    "narrativeqa_full_text.zip": (
        "https://huggingface.co/datasets/deepmind/narrativeqa/resolve/"
        f"{NARRATIVEQA_REVISION}/data/narrativeqa_full_text.zip?download=true",
        "3e179a579d348da37b4929f20ece277a721f853fdc5efc11f915904de2a71727",
    ),
    "narrativeqa-master.zip": (
        "https://huggingface.co/datasets/deepmind/narrativeqa/resolve/"
        f"{NARRATIVEQA_REVISION}/data/narrativeqa-master.zip?download=true",
        "d9fc92d5f53409f845ba44780e6689676d879c739589861b4805064513d1476b",
    ),
}

LONGHEALTH_CONTENT_REVISION = "513ab3088a5ecf91b3967f8c47ae373d1b87378f"
LONGHEALTH_URL = (
    "https://raw.githubusercontent.com/kbressem/LongHealth/"
    f"{LONGHEALTH_CONTENT_REVISION}/data/benchmark_v5.json"
)
LONGHEALTH_SHA256 = "82d34d9da47ab279d7aa89a6bdf298c0ac79f1e506e1dd0a3ea69a1ad5e2cb45"
LONGHEALTH_RECONSTRUCTION_SHA256 = (
    "a9d0052e348c85d4bd5a293138ae12035c42f17572f5dd8bd6730847f958df80"
)
NARRATIVEQA_RECONSTRUCTION_SHA256 = (
    "91bc5e49803dceea505b5d855cc9cea8a0478cb99e9aec96b7b13eb94f7c6a69"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256(destination) == expected_sha256:
        print(f"Using verified download: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "lora-kv-cache/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out)
    actual = sha256(temporary)
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 mismatch for {url}: expected {expected_sha256}, got {actual}"
        )
    temporary.replace(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def run(command: list[str], project_root: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=project_root, check=True)


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        selection = candidate / "data/selection/narrativeqa.json"
        if (candidate / "src/data_preparation").is_dir() and selection.is_file():
            return candidate
    raise RuntimeError(f"Could not locate the repository root above {start}")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def verify_longhealth(output: Path) -> None:
    rows = read_jsonl(output)
    patients = {str(row["metadata"]["patient_id"]) for row in rows}
    if len(rows) != 400 or len(patients) != 20:
        raise ValueError(
            f"LongHealth reconstruction has {len(patients)} documents/{len(rows)} QA; "
            "expected 20/400"
        )
    actual = sha256(output)
    if actual != LONGHEALTH_RECONSTRUCTION_SHA256:
        raise ValueError(
            "LongHealth reconstruction SHA256 mismatch: "
            f"expected {LONGHEALTH_RECONSTRUCTION_SHA256}, got {actual}"
        )
    print("Verified LongHealth: 20 documents / 400 questions")


def verify_narrativeqa(output: Path, project_root: Path) -> None:
    rows = read_jsonl(output)
    document_ids = {str(row["metadata"]["document_id"]) for row in rows}
    if len(rows) != 596 or len(document_ids) != 20:
        raise ValueError(
            f"NarrativeQA reconstruction has {len(document_ids)} documents/{len(rows)} QA; "
            "expected 20/596"
        )
    actual = sha256(output)
    if actual != NARRATIVEQA_RECONSTRUCTION_SHA256:
        raise ValueError(
            "NarrativeQA reconstruction SHA256 mismatch: "
            f"expected {NARRATIVEQA_RECONSTRUCTION_SHA256}, got {actual}"
        )
    generated_metadata = json.loads(output.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    reference_path = project_root / "data/selection/narrativeqa.json"
    reference_metadata = json.loads(reference_path.read_text(encoding="utf-8"))
    selected = [
        (str(item["document_id"]), int(item["num_qas"]))
        for item in generated_metadata["selected_docs"]
    ]
    expected = [
        (str(item["document_id"]), int(item["questions"]))
        for item in reference_metadata["selected_documents"]
    ]
    if selected != expected:
        raise ValueError("NarrativeQA selection does not match data/selection/narrativeqa.json")
    print("Verified NarrativeQA: 20 documents / 596 questions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Dataset output root relative to the repository root (default: data).",
    )
    parser.add_argument(
        "--download-cache",
        type=Path,
        default=Path(".cache/public_datasets"),
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "narrativeqa", "longhealth"),
        default="all",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = find_project_root(Path(__file__).resolve().parent)
    output_root = (project_root / args.output_root).resolve()
    cache = (project_root / args.download_cache).resolve()

    if args.dataset in {"all", "longhealth"}:
        raw = cache / "longhealth" / "benchmark_v5.json"
        download(LONGHEALTH_URL, raw, LONGHEALTH_SHA256)
        run(
            [
                sys.executable,
                "-m",
                "src.data_preparation.convert_longhealth",
                "--source",
                str(raw),
                "--output",
                str(output_root / "longhealth" / "longhealth.jsonl"),
            ],
            project_root,
        )
        verify_longhealth(output_root / "longhealth" / "longhealth.jsonl")

    if args.dataset in {"all", "narrativeqa"}:
        nqa_cache = cache / "narrativeqa"
        archives: dict[str, Path] = {}
        for name, (url, digest) in NARRATIVEQA_FILES.items():
            path = nqa_cache / name
            download(url, path, digest)
            archives[name] = path

        full_text = nqa_cache / "full_text"
        master = nqa_cache / "master"
        safe_extract(archives["narrativeqa_full_text.zip"], full_text)
        safe_extract(archives["narrativeqa-master.zip"], master)
        documents_csv = master / "narrativeqa-master" / "documents.csv"
        qaps_csv = master / "narrativeqa-master" / "qaps.csv"
        output = output_root / "narrativeqa" / "narrativeqa.jsonl"
        run(
            [
                sys.executable,
                "-m",
                "src.data_preparation.convert_narrativeqa",
                "--documents_csv",
                str(documents_csv),
                "--qaps_csv",
                str(qaps_csv),
                "--stories_dir",
                str(full_text),
                "--output",
                str(output),
                "--num_docs",
                "20",
                "--seed",
                "0",
                "--split",
                "all",
                "--min_story_words",
                "8000",
                "--max_story_words",
                "12000",
                "--no_download",
            ],
            project_root,
        )
        verify_narrativeqa(output, project_root)


if __name__ == "__main__":
    main()
