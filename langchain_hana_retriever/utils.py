"""Utility functions for tokenization and result fusion."""

from __future__ import annotations

import re

from langchain_core.documents import Document


def tokenize(text: str) -> list[str]:
    """Tokenize text by lowercasing, splitting on non-alphanumeric chars, and deduplicating.

    Simple but effective for Spanish/English text.
    """
    tokens = re.split(r"[^a-záéíóúüñA-ZÁÉÍÓÚÜÑ0-9]+", text.lower())
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if len(token) >= 2 and token not in seen:
            seen.add(token)
            result.append(token)
    return result


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    weights: list[float],
    k: int,
    rrf_k: int = 60,
) -> list[Document]:
    """Merge multiple ranked document lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: List of ranked document lists to merge.
        weights: Weight for each ranked list.
        k: Number of top results to return.
        rrf_k: RRF constant (default 60).

    Returns:
        Top-k merged documents sorted by fused score.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for weight, doc_list in zip(weights, ranked_lists):
        for rank, doc in enumerate(doc_list):
            content = doc.page_content
            score = weight / (rrf_k + rank + 1)
            scores[content] = scores.get(content, 0.0) + score
            if content not in doc_map:
                doc_map[content] = doc

    sorted_contents = sorted(scores, key=lambda c: scores[c], reverse=True)
    return [doc_map[c] for c in sorted_contents[:k]]
