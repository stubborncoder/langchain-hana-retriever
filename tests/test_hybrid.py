"""Tests for HANAHybridRetriever with mocked components."""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from langchain_hana_retriever.hybrid import HANAHybridRetriever


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    return store


@pytest.fixture
def mock_keyword_retriever():
    retriever = MagicMock()
    return retriever


def _make_hybrid(vector_store, keyword_retriever, alpha=0.5, k=10):
    return HANAHybridRetriever(
        vector_store=vector_store,
        keyword_retriever=keyword_retriever,
        alpha=alpha,
        k=k,
    )


class TestHANAHybridRetriever:
    def test_merges_results_via_rrf(self, mock_vector_store, mock_keyword_retriever):
        vec_docs = [Document(page_content="vec1"), Document(page_content="shared")]
        kw_docs = [Document(page_content="shared"), Document(page_content="kw1")]
        mock_vector_store.similarity_search.return_value = vec_docs
        mock_keyword_retriever.invoke.return_value = kw_docs

        retriever = _make_hybrid(mock_vector_store, mock_keyword_retriever)
        results = retriever.invoke("test query")

        # "shared" appears in both, should rank first
        assert results[0].page_content == "shared"
        assert len(results) <= 10

    def test_alpha_zero_keyword_only(self, mock_vector_store, mock_keyword_retriever):
        vec_docs = [Document(page_content="vec_only")]
        kw_docs = [Document(page_content="kw_only")]
        mock_vector_store.similarity_search.return_value = vec_docs
        mock_keyword_retriever.invoke.return_value = kw_docs

        retriever = _make_hybrid(mock_vector_store, mock_keyword_retriever, alpha=0.0)
        results = retriever.invoke("test")

        # With alpha=0, vector weight is 0, keyword weight is 1
        assert results[0].page_content == "kw_only"

    def test_alpha_one_vector_only(self, mock_vector_store, mock_keyword_retriever):
        vec_docs = [Document(page_content="vec_only")]
        kw_docs = [Document(page_content="kw_only")]
        mock_vector_store.similarity_search.return_value = vec_docs
        mock_keyword_retriever.invoke.return_value = kw_docs

        retriever = _make_hybrid(mock_vector_store, mock_keyword_retriever, alpha=1.0)
        results = retriever.invoke("test")

        # With alpha=1, vector weight is 1, keyword weight is 0
        assert results[0].page_content == "vec_only"

    def test_deduplication(self, mock_vector_store, mock_keyword_retriever):
        same_doc = Document(page_content="duplicate content")
        mock_vector_store.similarity_search.return_value = [same_doc]
        mock_keyword_retriever.invoke.return_value = [same_doc]

        retriever = _make_hybrid(mock_vector_store, mock_keyword_retriever)
        results = retriever.invoke("test")

        contents = [r.page_content for r in results]
        assert contents.count("duplicate content") == 1

    def test_respects_k(self, mock_vector_store, mock_keyword_retriever):
        vec_docs = [Document(page_content=f"vec{i}") for i in range(10)]
        kw_docs = [Document(page_content=f"kw{i}") for i in range(10)]
        mock_vector_store.similarity_search.return_value = vec_docs
        mock_keyword_retriever.invoke.return_value = kw_docs

        retriever = _make_hybrid(mock_vector_store, mock_keyword_retriever, k=5)
        results = retriever.invoke("test")

        assert len(results) == 5

    def test_empty_results_from_both(self, mock_vector_store, mock_keyword_retriever):
        mock_vector_store.similarity_search.return_value = []
        mock_keyword_retriever.invoke.return_value = []

        retriever = _make_hybrid(mock_vector_store, mock_keyword_retriever)
        results = retriever.invoke("test")

        assert results == []
