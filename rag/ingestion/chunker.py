"""INGESTION part 2: long text -> small overlapping chunks.

The idea, in 3 steps:
  1. Cut the text into sentences.
  2. Pack sentences into a chunk until it would pass CHUNK_SIZE characters.
  3. Start the next chunk with the last few sentences of the previous one (overlap),
     so an idea that sits on a boundary is not lost.
"""
import re

from rag import config


def split_into_sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.replace("\n", " ").strip()   # PDFs break lines mid-sentence
        if paragraph:
            sentences.extend(re.split(r"(?<=[.!?])\s+", paragraph))
    return sentences


def cut_long_piece(piece: str, size: int) -> list[str]:
    """A single 'sentence' longer than size (e.g. a table): cut it by words."""
    parts, current = [], ""
    words = []
    for word in piece.split(" "):                      # a "word" longer than size
        words.extend(word[i:i + size] for i in range(0, len(word), size) or [0])
    for word in words:
        if current and len(current) + len(word) + 1 > size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_text(text: str, size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    pieces = []
    for s in split_into_sentences(text):
        pieces.extend([s] if len(s) <= size else cut_long_piece(s, size))

    chunks, current, length = [], [], 0
    for piece in pieces:
        if current and length + len(piece) + 1 > size:
            chunks.append(" ".join(current))                 # chunk is full: save it
            keep, total = [], 0                              # build the overlap
            for prev in reversed(current):
                if total + len(prev) + 1 > overlap:
                    break
                keep.insert(0, prev)
                total += len(prev) + 1
            while keep and total + len(piece) + 1 > size:    # overlap must not overflow
                total -= len(keep.pop(0)) + 1
            current, length = keep, total
        current.append(piece)
        length += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks
