"""Tests for utility functions."""

from langchain_core.documents import Document

from langchain_hana_retriever.utils import reciprocal_rank_fusion, tokenize


class TestTokenize:
    def test_basic_english(self):
        tokens = tokenize("Hello world test")
        assert tokens == ["hello", "world", "test"]

    def test_spanish_text(self):
        tokens = tokenize("información según diseño")
        assert "información" in tokens
        assert "según" in tokens
        assert "diseño" in tokens

    def test_strips_short_tokens(self):
        tokens = tokenize("I am a big dog")
        assert "i" not in tokens
        assert "am" in tokens
        assert "a" not in tokens
        assert "big" in tokens
        assert "dog" in tokens

    def test_deduplication(self):
        tokens = tokenize("the cat and the cat")
        assert tokens.count("the") == 1
        assert tokens.count("cat") == 1

    def test_splits_on_punctuation(self):
        tokens = tokenize("hello-world, foo.bar")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens

    def test_empty_string(self):
        assert tokenize("") == []

    def test_special_characters_only(self):
        assert tokenize("!@#$%^&*()") == []


class TestReciprocalRankFusion:
    def test_basic_merge(self):
        list_a = [Document(page_content="doc1"), Document(page_content="doc2")]
        list_b = [Document(page_content="doc2"), Document(page_content="doc3")]
        result = reciprocal_rank_fusion([list_a, list_b], weights=[1.0, 1.0], k=3)
        # doc2 appears in both lists, should rank highest
        assert result[0].page_content == "doc2"
        assert len(result) == 3

    def test_respects_k(self):
        list_a = [Document(page_content=f"doc{i}") for i in range(10)]
        result = reciprocal_rank_fusion([list_a], weights=[1.0], k=3)
        assert len(result) == 3

    def test_weights_affect_ranking(self):
        list_a = [Document(page_content="a_only")]
        list_b = [Document(page_content="b_only")]
        # Heavily weight list_a
        result = reciprocal_rank_fusion([list_a, list_b], weights=[10.0, 0.1], k=2)
        assert result[0].page_content == "a_only"

    def test_deduplication(self):
        list_a = [Document(page_content="same")]
        list_b = [Document(page_content="same")]
        result = reciprocal_rank_fusion([list_a, list_b], weights=[1.0, 1.0], k=5)
        assert len(result) == 1

    def test_empty_lists(self):
        result = reciprocal_rank_fusion([], weights=[], k=5)
        assert result == []
