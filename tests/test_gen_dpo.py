"""Unit tests for scripts/gen_dpo.py — DPO pair generation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from gen_dpo import (
    _generate_rejected_candidates,
    _make_dpo_record,
    _sample_question,
)


class TestGenerateRejectedCandidates:
    def _cfg(self, params=None, path_params=None):
        return {
            "endpoint": {"method": "GET", "path": "/v1/items", "name": "items",
                         "vendor": "test", "base_url": "https://api.test.com/v1/items"},
            "params": params or {},
            "path_params": path_params or {},
        }

    def test_always_generates_at_least_one_candidate(self):
        cfg = self._cfg(params={"pageSize": {"type": "integer"}})
        candidates = _generate_rejected_candidates(cfg, {"pageSize": 10})
        assert len(candidates) >= 1

    def test_spurious_param_candidate(self):
        cfg = self._cfg(params={"pageSize": {"type": "integer"}})
        candidates = _generate_rejected_candidates(cfg, {"pageSize": 10})
        spurious = [c for c in candidates if "_invalid_param_xyz" in c]
        assert len(spurious) >= 1

    def test_zero_id_candidate_for_path_param(self):
        cfg = self._cfg(path_params={"id": {"type": "integer"}})
        candidates = _generate_rejected_candidates(cfg, {"id": 42})
        zero_id = [c for c in candidates if c.get("id") == 0]
        assert len(zero_id) >= 1

    def test_negative_page_size_candidate(self):
        cfg = self._cfg(params={"pageSize": {"type": "integer"}})
        candidates = _generate_rejected_candidates(cfg, {"pageSize": 10})
        bad_ps = [c for c in candidates if c.get("pageSize", 0) <= 0]
        assert len(bad_ps) >= 1

    def test_missing_path_param_candidate(self):
        cfg = self._cfg(path_params={"id": {"type": "integer"}})
        candidates = _generate_rejected_candidates(cfg, {"id": 42})
        missing_id = [c for c in candidates if "id" not in c]
        assert len(missing_id) >= 1

    def test_candidates_differ_from_chosen(self):
        cfg = self._cfg(params={"pageSize": {"type": "integer"}})
        chosen = {"pageSize": 10}
        candidates = _generate_rejected_candidates(cfg, chosen)
        for c in candidates:
            assert c != chosen


class TestMakeDpoRecord:
    def _cfg(self):
        return {
            "endpoint": {"method": "GET", "path": "/v1/items", "name": "items",
                         "vendor": "test", "base_url": "https://api.test.com/v1/items"},
            "params": {"pageSize": {"type": "integer"}},
            "path_params": {},
        }

    def test_record_format(self):
        cfg = self._cfg()
        record = _make_dpo_record(cfg, "list items", {"pageSize": 10}, {"pageSize": 0})
        assert "question" in record
        assert "chosen" in record
        assert "rejected" in record
        assert "schema" in record
        assert "intent_category" in record

    def test_chosen_and_rejected_differ(self):
        cfg = self._cfg()
        record = _make_dpo_record(cfg, "list", {"pageSize": 10}, {"pageSize": 0})
        assert record["chosen"] != record["rejected"]

    def test_endpoint_in_chosen_and_rejected(self):
        cfg = self._cfg()
        record = _make_dpo_record(cfg, "list", {}, {"bad": 1})
        assert "endpoint" in record["chosen"]
        assert "endpoint" in record["rejected"]
        assert record["chosen"]["endpoint"] == record["rejected"]["endpoint"]

    def test_schema_is_string(self):
        cfg = self._cfg()
        record = _make_dpo_record(cfg, "list", {}, {})
        assert isinstance(record["schema"], str)
        assert "GET /v1/items" in record["schema"]


class TestSampleQuestion:
    def _cfg(self, name="items", path_params=None):
        return {
            "endpoint": {"method": "GET", "path": "/v1/items", "name": name,
                         "vendor": "test", "base_url": "https://api.test.com/v1/items"},
            "params": {"pageSize": {"type": "integer"}},
            "path_params": path_params or {},
        }

    def test_no_params_returns_list_question(self):
        cfg = self._cfg()
        q = _sample_question(cfg, {})
        assert "list" in q.lower() or "items" in q.lower()

    def test_path_param_returns_by_id_question(self):
        cfg = self._cfg(name="item", path_params={"id": {"type": "integer"}})
        q = _sample_question(cfg, {"id": 42})
        assert "42" in q

    def test_filter_params_in_question(self):
        cfg = self._cfg()
        q = _sample_question(cfg, {"pageSize": 10})
        assert "pageSize" in q or "10" in q
