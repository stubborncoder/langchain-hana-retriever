"""Integration tests requiring a live SAP HANA Cloud connection."""

import os
import uuid

import pytest

from langchain_hana_retriever.bm25 import HANABm25Retriever

pytestmark = pytest.mark.skipif(
    not os.environ.get("HANA_HOST"),
    reason="HANA_HOST not set — skipping integration tests",
)

TEST_TABLE = f"TEST_BM25_{uuid.uuid4().hex[:8].upper()}"


@pytest.fixture(autouse=True)
def setup_test_table(hana_connection):
    """Create and populate a temp table, drop it after tests."""
    cursor = hana_connection.cursor()
    cursor.execute(
        f"CREATE TABLE {TEST_TABLE} ("
        f"  VEC_TEXT NCLOB, "
        f"  SOURCE NVARCHAR(255)"
        f")"
    )
    docs = [
        ("Python es un lenguaje de programación muy popular", "doc1.pdf"),
        ("SAP HANA Cloud ofrece procesamiento en memoria", "doc2.pdf"),
        ("La inteligencia artificial transforma los negocios", "doc3.pdf"),
        ("Python y machine learning van de la mano", "doc4.pdf"),
        ("Base de datos en memoria para análisis en tiempo real", "doc5.pdf"),
    ]
    for text, source in docs:
        cursor.execute(f"INSERT INTO {TEST_TABLE} (VEC_TEXT, SOURCE) VALUES (?, ?)", [text, source])
    hana_connection.commit()
    cursor.close()

    yield

    cursor = hana_connection.cursor()
    cursor.execute(f"DROP TABLE {TEST_TABLE}")
    cursor.close()


class TestBm25Integration:
    def test_retrieves_relevant_documents(self, hana_connection):
        retriever = HANABm25Retriever(
            connection=hana_connection,
            table_name=TEST_TABLE,
            metadata_columns=["SOURCE"],
            k=3,
        )
        results = retriever.invoke("Python programación")
        assert len(results) > 0
        assert any("Python" in r.page_content for r in results)

    def test_metadata_populated(self, hana_connection):
        retriever = HANABm25Retriever(
            connection=hana_connection,
            table_name=TEST_TABLE,
            metadata_columns=["SOURCE"],
        )
        results = retriever.invoke("HANA Cloud memoria")
        assert len(results) > 0
        assert "SOURCE" in results[0].metadata
        assert results[0].metadata["SOURCE"].endswith(".pdf")

    def test_no_matches(self, hana_connection):
        retriever = HANABm25Retriever(
            connection=hana_connection,
            table_name=TEST_TABLE,
        )
        results = retriever.invoke("xyznonexistent")
        assert results == []
