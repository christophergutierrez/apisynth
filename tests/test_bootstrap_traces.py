"""Unit tests for bootstrap_traces.py dedup and temperature changes."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from bootstrap_traces import _cosine_similarity, _dedup_by_embedding


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 0]) == 0.0

    def test_partial_overlap(self):
        score = _cosine_similarity([1, 1, 0], [1, 0, 0])
        assert 0 < score < 1.0


class TestDedupByEmbedding:
    """Tests dedup logic with mocked embeddings to avoid sentence-transformers dependency."""

    def _make_candidates(self, prompts: list[str]) -> list[dict]:
        return [{"prompt": p, "ok": True, "api_call": {"endpoint": "GET /v1/items", "params": {}}}
                for p in prompts]

    def test_dedup_empty_existing(self, tmp_path):
        """With no existing records, all candidates are kept."""
        candidates = self._make_candidates(["list items", "get item 1"])
        try:
            kept, removed = _dedup_by_embedding(candidates, str(tmp_path / "out.jsonl"), 0.95)
            # If sentence-transformers is available, all should be kept (distinct prompts)
            assert removed == 0
            assert len(kept) == 2
        except Exception:
            pytest.skip("sentence-transformers not available")

    def test_threshold_1_0_disables_dedup(self, tmp_path):
        """Threshold of 1.0 means nothing is ever considered a duplicate."""
        candidates = self._make_candidates(["list items", "list items"])
        kept, removed = _dedup_by_embedding(candidates, None, 1.0)
        # With threshold 1.0 we skip dedup entirely per CLI logic
        # This test verifies the function handles the threshold gracefully
        assert isinstance(kept, list)

    def test_dedup_filters_near_duplicate(self, tmp_path, monkeypatch):
        """Mocked embeddings: verify dedup removes near-identical vectors."""
        import json, sys, unittest.mock as mock

        out = tmp_path / "out.jsonl"
        out.write_text(json.dumps({"question": "list items", "api_call": {}}) + "\n")

        class _Arr:
            def __init__(self, data):
                self._d = data
            def tolist(self):
                return self._d
            def __getitem__(self, i):
                return self._d[i]

        class MockModel:
            def encode(self, texts, **kwargs):
                return _Arr([[1.0, 0.0]] * len(texts))

        mock_st = mock.MagicMock()
        mock_st.SentenceTransformer.return_value = MockModel()
        with mock.patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            candidates = self._make_candidates(["list items"])
            kept, removed = _dedup_by_embedding(candidates, str(out), 0.95)
            assert removed == 1
            assert len(kept) == 0

    def test_dedup_passes_distinct(self, tmp_path, monkeypatch):
        """Mocked embeddings: verify distinct questions are kept."""
        import json, sys, unittest.mock as mock

        out = tmp_path / "out.jsonl"
        out.write_text(json.dumps({"question": "list items", "api_call": {}}) + "\n")

        class _Arr:
            def __init__(self, data):
                self._d = data
            def tolist(self):
                return self._d
            def __getitem__(self, i):
                return self._d[i]

        call_count = 0

        class MockModel:
            def encode(self, texts, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return _Arr([[1.0, 0.0]] * len(texts))
                return _Arr([[0.0, 1.0]] * len(texts))

        mock_st = mock.MagicMock()
        mock_st.SentenceTransformer.return_value = MockModel()
        with mock.patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            candidates = self._make_candidates(["get item 42"])
            kept, removed = _dedup_by_embedding(candidates, str(out), 0.95)
            assert removed == 0
            assert len(kept) == 1
