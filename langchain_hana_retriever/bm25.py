"""BM25 retriever for SAP HANA Cloud."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

from langchain_hana_retriever.utils import tokenize


class HANABm25Retriever(BaseRetriever):
    """BM25 keyword retriever backed by SAP HANA Cloud.

    Uses SQL LOCATE for candidate filtering, then scores with BM25Okapi in Python.
    """

    connection: Any
    table_name: str
    content_column: str = "VEC_TEXT"
    metadata_columns: list[str] = []
    k: int = 10
    candidate_limit: int = 50
    max_tokens_in_query: int = 5

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        tokens = tokenize(query)
        if not tokens:
            return []

        # Pick the longest tokens as proxy for distinctiveness
        tokens = sorted(tokens, key=len, reverse=True)[: self.max_tokens_in_query]

        # Build SQL with LOCATE conditions
        columns = [self.content_column] + self.metadata_columns
        col_list = ", ".join(columns)
        where_clauses = [
            f"LOCATE(LOWER(TO_NVARCHAR({self.content_column})), ?) > 0" for _ in tokens
        ]
        where_sql = " OR ".join(where_clauses)
        sql = (
            f"SELECT {col_list} FROM {self.table_name} "
            f"WHERE {where_sql} "
            f"LIMIT {self.candidate_limit}"
        )

        cursor = self.connection.cursor()
        try:
            cursor.execute(sql, tokens)
            rows = cursor.fetchall()
        finally:
            cursor.close()

        if not rows:
            return []

        # Score candidates with BM25
        corpus = [tokenize(row[0]) for row in rows]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(tokens)

        # Pair rows with scores and sort
        scored = sorted(zip(scores, rows), key=lambda x: x[0], reverse=True)

        results: list[Document] = []
        for score, row in scored[: self.k]:
            metadata: dict[str, Any] = {}
            for i, col_name in enumerate(self.metadata_columns):
                metadata[col_name] = row[i + 1]
            metadata["bm25_score"] = float(score)
            results.append(Document(page_content=row[0], metadata=metadata))

        return results
