#!/usr/bin/env python3
"""Document loading and chunking utilities for generative QA."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


def load_cluster_chunks(cluster_dir: Path) -> List[str]:
    """
    Load all .txt chunks from a cluster directory.

    Args:
        cluster_dir: Path to cluster directory containing .txt files

    Returns:
        List of text chunks (file contents)
    """
    chunks = []
    for txt_file in sorted(cluster_dir.glob("*.txt")):
        chunks.append(txt_file.read_text())
    return chunks


@dataclass(frozen=True)
class ShatteredSection:
    """One final overlapping section emitted by ``shatter_document``."""

    text: str
    start_char: int
    end_char: int
    start_unit_idx: int
    end_unit_idx: int


@dataclass(frozen=True)
class ShatteredDocument:
    """Normalized shatter output with section span metadata."""

    normalized_text: str
    unit_chunks: List[str]
    sections: List[ShatteredSection]


@dataclass(frozen=True)
class ShatterPiece:
    """One atomic text piece plus the exact separator that followed it."""

    text: str
    trailing_separator: str = ""

    @property
    def rendered(self) -> str:
        return f"{self.text}{self.trailing_separator}"

    @property
    def word_count(self) -> int:
        return len(self.text.split())


_SHATTER_BOUNDARY_RE = re.compile(
    r"(?P<paragraph>\n\s*\n+)|(?P<sentence_punct>[.!?])(?P<sentence_sep>\s+|$)",
    re.S,
)


def _split_shatter_pieces(text: str) -> List[ShatterPiece]:
    """Split text into pieces while preserving the original split separators."""

    pieces: List[ShatterPiece] = []
    start = 0

    for match in _SHATTER_BOUNDARY_RE.finditer(text):
        if match.group("paragraph") is not None:
            piece_text = text[start:match.start()]
            trailing_separator = match.group("paragraph")
        else:
            piece_text = text[start:match.start()] + match.group("sentence_punct")
            trailing_separator = match.group("sentence_sep")

        if piece_text:
            pieces.append(
                ShatterPiece(
                    text=piece_text,
                    trailing_separator=trailing_separator,
                )
            )
        elif trailing_separator and pieces:
            prev = pieces[-1]
            pieces[-1] = ShatterPiece(
                text=prev.text,
                trailing_separator=prev.trailing_separator + trailing_separator,
            )

        start = match.end()

    tail = text[start:]
    if tail:
        pieces.append(ShatterPiece(text=tail))

    return pieces


def _build_unit_chunks(pieces: List[ShatterPiece], unit_size: int) -> List[str]:
    """Build the non-overlapping unit chunks that underpin the final overlap."""

    unit_chunks = []
    bucket: List[ShatterPiece] = []
    bucket_word_count = 0

    for piece in pieces:
        piece_word_count = piece.word_count

        if bucket and bucket_word_count + piece_word_count > unit_size:
            unit_chunks.append("".join(item.rendered for item in bucket))
            bucket = []
            bucket_word_count = 0

        bucket.append(piece)
        bucket_word_count += piece_word_count

    if bucket:
        unit_chunks.append("".join(item.rendered for item in bucket))

    return unit_chunks


def shatter_document_with_metadata(
    text: str,
    chunk_size: int,
    overlap_ratio: float,
) -> ShatteredDocument:
    """Shatter a document and preserve section spans within a normalized text."""
    pieces = _split_shatter_pieces(text)
    unit_size = max(1, int(chunk_size * (1 - overlap_ratio)))
    unit_chunks = _build_unit_chunks(pieces, unit_size)

    if not unit_chunks:
        return ShatteredDocument(
            normalized_text=text,
            unit_chunks=[],
            sections=[
                ShatteredSection(
                    text=text,
                    start_char=0,
                    end_char=len(text),
                    start_unit_idx=0,
                    end_unit_idx=0,
                )
            ],
        )

    normalized_text = "".join(unit_chunks)
    unit_spans: List[tuple[int, int]] = []
    cursor = 0
    for unit in unit_chunks:
        start = cursor
        end = start + len(unit)
        unit_spans.append((start, end))
        cursor = end

    if len(unit_chunks) == 1:
        return ShatteredDocument(
            normalized_text=normalized_text,
            unit_chunks=unit_chunks,
            sections=[
                ShatteredSection(
                    text=unit_chunks[0],
                    start_char=0,
                    end_char=len(unit_chunks[0]),
                    start_unit_idx=0,
                    end_unit_idx=0,
                )
            ],
        )

    sections: List[ShatteredSection] = []
    last_span: tuple[int, int] | None = None
    for i in range(len(unit_chunks)):
        if i == 0:
            span = (0, 1)
        elif i == len(unit_chunks) - 1:
            span = (i - 1, i)
        else:
            span = (i, i + 1)

        if span == last_span:
            continue

        start_char = unit_spans[span[0]][0]
        end_char = unit_spans[span[1]][1]
        sections.append(
            ShatteredSection(
                text=normalized_text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
                start_unit_idx=span[0],
                end_unit_idx=span[1],
            )
        )
        last_span = span

    return ShatteredDocument(
        normalized_text=normalized_text,
        unit_chunks=unit_chunks,
        sections=sections,
    )


def shatter_document(text: str, chunk_size: int, overlap_ratio: float) -> List[str]:
    """
    Shatter a document into overlapping chunks.

    Args:
        text: The document text to shatter
        chunk_size: Target word length for final chunks
        overlap_ratio: Ratio of overlap between adjacent chunks (0.0 to 1.0)

    Returns:
        List of overlapping text chunks
    """
    shattered = shatter_document_with_metadata(text, chunk_size, overlap_ratio)
    return [section.text for section in shattered.sections]
