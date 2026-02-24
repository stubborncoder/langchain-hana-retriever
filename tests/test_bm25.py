"""Tests for HANABm25Retriever with mocked HANA connection."""

from unittest.mock import MagicMock

import pytest

from langchain_hana_retriever.bm25 import HANABm25Retriever


@pytest.fixture
def mock_connection():
    conn = MagicMock()
    return conn


def _setup_cursor(conn, rows):
    """Configure mock connection to return given rows."""
    cursor = MagicMock()
    cursor.fetchall.return_value = rows
    conn.cursor.return_value = cursor
    return cursor


class TestHANABm25Retriever:
    def test_returns_documents_with_metadata(self, mock_connection):
        rows = [
            ("Document about Python programming", "source1.pdf", "chapter1"),
            ("Another document about Java", "source2.pdf", "chapter2"),
        ]
        _setup_cursor(mock_connection, rows)

        retriever = HANABm25Retriever(
            connection=mock_connection,
            table_name="TEST_TABLE",
            metadata_columns=["SOURCE", "CHAPTER"],
        )
        results = retriever.invoke("Python programming")

        assert len(results) > 0
        assert results[0].metadata["SOURCE"] is not None
        assert results[0].metadata["CHAPTER"] is not None
        assert "bm25_score" in results[0].metadata

    def test_empty_query_returns_empty(self, mock_connection):
        retriever = HANABm25Retriever(
            connection=mock_connection,
            table_name="TEST_TABLE",
        )
        results = retriever.invoke("")
        assert results == []
        mock_connection.cursor.assert_not_called()

    def test_bm25_ranks_exact_match_higher(self, mock_connection):
        rows = [
            ("The quick brown fox jumps over the lazy dog",),
            ("Python is a great programming language for data science",),
            ("Python programming with advanced techniques in Python",),
        ]
        _setup_cursor(mock_connection, rows)

        retriever = HANABm25Retriever(
            connection=mock_connection,
            table_name="TEST_TABLE",
        )
        results = retriever.invoke("Python programming")

        # Doc with more Python mentions should rank higher
        assert "Python" in results[0].page_content

    def test_respects_k_parameter(self, mock_connection):
        rows = [(f"Document number {i}",) for i in range(20)]
        _setup_cursor(mock_connection, rows)

        retriever = HANABm25Retriever(
            connection=mock_connection,
            table_name="TEST_TABLE",
            k=3,
        )
        results = retriever.invoke("Document number")
        assert len(results) == 3

    def test_handles_special_characters(self, mock_connection):
        rows = [("Información sobre diseño técnico",)]
        _setup_cursor(mock_connection, rows)

        retriever = HANABm25Retriever(
            connection=mock_connection,
            table_name="TEST_TABLE",
        )
        results = retriever.invoke("información diseño!")
        assert len(results) == 1

    def test_no_results_from_db(self, mock_connection):
        _setup_cursor(mock_connection, [])

        retriever = HANABm25Retriever(
            connection=mock_connection,
            table_name="TEST_TABLE",
        )
        results = retriever.invoke("something")
        assert results == []

    def test_sql_uses_parameterized_query(self, mock_connection):
        rows = [("test document",)]
        cursor = _setup_cursor(mock_connection, rows)

        retriever = HANABm25Retriever(
            connection=mock_connection,
            table_name="TEST_TABLE",
        )
        retriever.invoke("hello world")

        cursor.execute.assert_called_once()
        sql, params = cursor.execute.call_args[0]
        assert "?" in sql
        assert isinstance(params, list)
