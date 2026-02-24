# langchain-hana-retriever

LangChain BM25 and hybrid retrievers for SAP HANA Cloud.

SAP HANA Cloud doesn't have built-in BM25/full-text ranking. This package provides two LangChain-compatible retrievers that enable proper hybrid retrieval for RAG pipelines without requiring PAL.

## How it works

- **`HANABm25Retriever`** — Uses SQL `LOCATE` to fetch keyword-matching candidates from HANA, then scores them with [BM25Okapi](https://github.com/dorianbrown/rank_bm25) in Python.
- **`HANAHybridRetriever`** — Combines a HANA vector store (semantic search) with the BM25 retriever, merging results via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf).

## Installation

```bash
pip install langchain-hana-retriever
```

For hybrid retrieval (requires a vector store):

```bash
pip install "langchain-hana-retriever[hybrid]"
```

## Usage

### BM25 keyword retriever

```python
import hdbcli.dbapi
from langchain_hana_retriever import HANABm25Retriever

connection = hdbcli.dbapi.connect(
    address="your-host.hanacloud.ondemand.com",
    port=443,
    user="your_user",
    password="your_password",
    encrypt=True,
)

retriever = HANABm25Retriever(
    connection=connection,
    table_name="YOUR_TABLE",
    content_column="VEC_TEXT",        # column containing document text
    metadata_columns=["SOURCE"],      # optional metadata columns to return
    k=10,                             # number of results
    candidate_limit=50,               # SQL LIMIT for initial candidate fetch
)

docs = retriever.invoke("your search query")
```

### Hybrid retriever (vector + BM25)

```python
from langchain_community.vectorstores import HanaDB
from langchain_openai import OpenAIEmbeddings
from langchain_hana_retriever import HANABm25Retriever, HANAHybridRetriever

# Set up vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = HanaDB(
    connection=connection,
    embedding=embeddings,
    table_name="YOUR_TABLE",
)

# Set up keyword retriever
keyword_retriever = HANABm25Retriever(
    connection=connection,
    table_name="YOUR_TABLE",
    k=10,
)

# Combine both
hybrid = HANAHybridRetriever(
    vector_store=vector_store,
    keyword_retriever=keyword_retriever,
    alpha=0.5,  # 0 = keyword only, 1 = vector only
    k=10,
)

docs = hybrid.invoke("your search query")
```

## Parameters

### HANABm25Retriever

| Parameter | Type | Default | Description |
|---|---|---|---|
| `connection` | `hdbcli.dbapi.Connection` | required | HANA database connection |
| `table_name` | `str` | required | Table to search |
| `content_column` | `str` | `"VEC_TEXT"` | Column containing document text |
| `metadata_columns` | `list[str]` | `[]` | Additional columns to include in metadata |
| `k` | `int` | `10` | Number of results to return |
| `candidate_limit` | `int` | `50` | SQL LIMIT for candidate fetching |
| `max_tokens_in_query` | `int` | `5` | Max query tokens sent to SQL WHERE clause |

### HANAHybridRetriever

| Parameter | Type | Default | Description |
|---|---|---|---|
| `vector_store` | `HanaDB` | required | HANA vector store for semantic search |
| `keyword_retriever` | `HANABm25Retriever` | required | BM25 retriever for keyword search |
| `alpha` | `float` | `0.5` | Balance between vector (`1.0`) and keyword (`0.0`) |
| `k` | `int` | `10` | Number of results to return |

## Development

```bash
git clone https://github.com/stubborncoder/langchain-hana-retriever.git
cd langchain-hana-retriever
pip install -e ".[dev]"

# Run unit tests
pytest tests/test_utils.py tests/test_bm25.py tests/test_hybrid.py -v

# Run integration tests (requires HANA credentials in .env)
pytest tests/test_integration.py -v
```

## License

MIT
