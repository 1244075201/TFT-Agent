"""Heading-aware deterministic knowledge chunking."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    heading: str | None
    text: str
    token_count: int


class HeadingAwareChunker:
    """Split Markdown sections by paragraphs while retaining their heading path."""

    def __init__(self, target_tokens: int = 450, overlap_tokens: int = 40) -> None:
        if target_tokens < 50 or overlap_tokens < 0 or overlap_tokens >= target_tokens:
            raise ValueError("invalid chunk sizing")
        self._target = target_tokens
        self._overlap = overlap_tokens

    def chunk(self, text: str, document_title: str | None = None) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for heading, paragraphs in _sections(text):
            current: list[str] = []
            for paragraph in paragraphs:
                if current and _tokens("\n\n".join([*current, paragraph])) > self._target:
                    chunks.append(_make_chunk(document_title, heading, current))
                    overlap = " ".join("\n\n".join(current).split()[-self._overlap :])
                    current = [overlap] if overlap else []
                current.append(paragraph)
            if current:
                chunks.append(_make_chunk(document_title, heading, current))
        return chunks


def _sections(text: str) -> list[tuple[str | None, list[str]]]:
    path: list[str] = []
    sections: list[tuple[str | None, list[str]]] = []
    heading: str | None = None
    body: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        match = re.match(r"^(#{1,6})\s+(.+)$", block.strip())
        if match:
            if body:
                sections.append((heading, body))
                body = []
            level = len(match.group(1))
            path = path[: level - 1] + [match.group(2).strip()]
            heading = " > ".join(path)
        elif block.strip():
            body.append(block.strip())
    if body:
        sections.append((heading, body))
    return sections


def _make_chunk(
    document_title: str | None, heading: str | None, paragraphs: list[str]
) -> TextChunk:
    text = "\n\n".join(paragraphs)
    context = [value for value in (document_title, heading) if value]
    embedding_text = "\n".join([*context, text])
    return TextChunk(heading=heading, text=embedding_text, token_count=_tokens(embedding_text))


def _tokens(text: str) -> int:
    return len(re.findall(r"[\w']+|[^\w\s]", text, re.UNICODE))
