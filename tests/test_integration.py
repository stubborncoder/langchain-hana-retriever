"""Integration tests requiring a live SAP HANA Cloud connection."""

import os
import uuid

import pytest

from langchain_hana_retriever.bm25 import HANABm25Retriever
from langchain_hana_retriever.hybrid import HANAHybridRetriever

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

    def test_bm25_reranking_prefers_higher_keyword_density(self, hana_connection):
        """Doc with more mentions of query terms should rank higher than one with fewer."""
        retriever = HANABm25Retriever(
            connection=hana_connection,
            table_name=TEST_TABLE,
            metadata_columns=["SOURCE"],
            k=5,
        )
        # "Python" appears in doc1 and doc4.
        # doc1: "Python es un lenguaje de programación muy popular"  (1 mention)
        # doc4: "Python y machine learning van de la mano"           (1 mention)
        # But query "Python programación" should rank doc1 higher because it also
        # matches "programación", giving it higher BM25 score.
        results = retriever.invoke("Python programación")
        assert len(results) >= 2
        # The top result should be doc1 which matches both query terms
        assert "programación" in results[0].page_content
        assert results[0].metadata["SOURCE"] == "doc1.pdf"
        # All results should have bm25_score in descending order
        scores = [r.metadata["bm25_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_candidate_limit_controls_sql_results(self, hana_connection):
        """Setting candidate_limit should limit how many rows SQL fetches."""
        retriever = HANABm25Retriever(
            connection=hana_connection,
            table_name=TEST_TABLE,
            candidate_limit=2,
            k=10,
        )
        # Even though k=10, candidate_limit=2 restricts SQL to 2 rows
        # so we get at most 2 results
        results = retriever.invoke("memoria datos")
        assert len(results) <= 2


HYBRID_TABLE = f"TEST_HYBRID_{uuid.uuid4().hex[:8].upper()}"


@pytest.fixture
def hybrid_table(hana_connection):
    """Create a table with embeddings for hybrid search tests."""
    from langchain_community.vectorstores import HanaDB
    from langchain_openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = HanaDB(
        connection=hana_connection,
        embedding=embeddings,
        table_name=HYBRID_TABLE,
    )

    docs = [
        "Python es un lenguaje de programación muy popular para ciencia de datos",
        "SAP HANA Cloud ofrece procesamiento en memoria de alto rendimiento",
        "La inteligencia artificial y el machine learning transforman los negocios",
        "Las bases de datos vectoriales permiten búsqueda semántica eficiente",
        "El procesamiento del lenguaje natural usa redes neuronales profundas",
    ]
    vector_store.add_texts(docs)

    yield vector_store

    cursor = hana_connection.cursor()
    cursor.execute(f"DROP TABLE {HYBRID_TABLE}")
    cursor.close()


class TestHybridIntegration:
    def test_hybrid_retriever_returns_results(self, hana_connection, hybrid_table):
        """Hybrid retriever should return results combining both sources."""
        keyword_retriever = HANABm25Retriever(
            connection=hana_connection,
            table_name=HYBRID_TABLE,
            k=5,
        )
        hybrid = HANAHybridRetriever(
            vector_store=hybrid_table,
            keyword_retriever=keyword_retriever,
            alpha=0.5,
            k=3,
        )
        results = hybrid.invoke("Python programación datos")
        assert len(results) > 0
        assert len(results) <= 3

    def test_hybrid_boosts_docs_found_by_both(self, hana_connection, hybrid_table):
        """A doc matching both semantic and keyword search should rank higher."""
        keyword_retriever = HANABm25Retriever(
            connection=hana_connection,
            table_name=HYBRID_TABLE,
            k=5,
        )
        hybrid = HANAHybridRetriever(
            vector_store=hybrid_table,
            keyword_retriever=keyword_retriever,
            alpha=0.5,
            k=5,
        )
        # "Python ciencia datos" should strongly match doc1 both semantically
        # and via keywords
        results = hybrid.invoke("Python ciencia datos")
        assert len(results) > 0
        assert "Python" in results[0].page_content

    def test_alpha_zero_gives_keyword_dominant(self, hana_connection, hybrid_table):
        """With alpha=0, keyword results should dominate."""
        keyword_retriever = HANABm25Retriever(
            connection=hana_connection,
            table_name=HYBRID_TABLE,
            k=5,
        )
        hybrid = HANAHybridRetriever(
            vector_store=hybrid_table,
            keyword_retriever=keyword_retriever,
            alpha=0.0,
            k=3,
        )
        # Query with a very specific keyword that only matches one doc
        results = hybrid.invoke("HANA memoria rendimiento")
        assert len(results) > 0
        # Top result should contain the keyword match
        assert "HANA" in results[0].page_content
