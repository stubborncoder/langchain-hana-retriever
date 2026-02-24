"""Hybrid retriever combining vector search and BM25 for SAP HANA Cloud."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_hana_retriever.utils import reciprocal_rank_fusion


class HANAHybridRetriever(BaseRetriever):
    """Hybrid retriever that merges vector similarity and BM25 keyword results via RRF.

    Uses a HanaDB vector store for semantic search and HANABm25Retriever for keyword
    search, then fuses results using Reciprocal Rank Fusion.
    """

    vector_store: Any
    keyword_retriever: Any  # HANABm25Retriever (Any to allow mocking)
    alpha: float = 0.5
    k: int = 10

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        vector_results = self.vector_store.similarity_search(query, k=self.k)
        keyword_results = self.keyword_retriever.invoke(query)

        return reciprocal_rank_fusion(
            ranked_lists=[vector_results, keyword_results],
            weights=[self.alpha, 1 - self.alpha],
            k=self.k,
        )
