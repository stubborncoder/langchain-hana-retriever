"""Shared fixtures for tests."""

import os

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@pytest.fixture
def hana_connection():
    """Create a real HANA connection for integration tests."""
    import hdbcli.dbapi

    conn = hdbcli.dbapi.connect(
        address=os.environ["HANA_HOST"],
        port=int(os.environ.get("HANA_PORT", "443")),
        user=os.environ["HANA_USER"],
        password=os.environ["HANA_PASSWORD"],
        encrypt=True,
    )
    yield conn
    conn.close()
