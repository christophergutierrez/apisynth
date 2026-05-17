"""Unit tests for scripts/eval.py — 3-tier evaluation rubric."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from eval import format_score, param_f1, score_record, _band


class TestFormatScore:
    def test_valid_single_step(self):
        assert format_score({"endpoint": "GET /v1/items", "params": {}}) == 1.0

    def test_valid_with_params(self):
        assert format_score({"endpoint": "GET /v1/items", "params": {"pageSize": 10}}) == 1.0

    def test_valid_chained(self):
        api_call = {"steps": [
            {"endpoint": "GET /v1/items", "params": {}},
            {"endpoint": "GET /v1/items/{id}", "params": {"id": "{{steps.0.id}}"}},
        ]}
        assert format_score(api_call) == 1.0

    def test_missing_endpoint(self):
        assert format_score({"params": {}}) == 0.0

    def test_missing_params(self):
        assert format_score({"endpoint": "GET /v1/items"}) == 0.0

    def test_not_a_dict(self):
        assert format_score("not a dict") == 0.0
        assert format_score(None) == 0.0
        assert format_score([]) == 0.0

    def test_params_not_dict(self):
        assert format_score({"endpoint": "GET /v1/items", "params": "oops"}) == 0.0

    def test_chained_empty_steps(self):
        assert format_score({"steps": []}) == 0.0

    def test_chained_step_missing_endpoint(self):
        assert format_score({"steps": [{"params": {}}]}) == 0.0


class TestParamF1:
    def test_exact_match(self):
        assert param_f1(
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10, "networkId": 5}},
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10, "networkId": 5}},
        ) == 1.0

    def test_no_params_both(self):
        assert param_f1(
            {"endpoint": "GET /v1/me", "params": {}},
            {"endpoint": "GET /v1/me", "params": {}},
        ) == 1.0

    def test_partial_overlap(self):
        # predicted: {pageSize}, expected: {pageSize, networkId}
        # precision = 1/1, recall = 1/2, F1 = 2/3
        score = param_f1(
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10}},
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10, "networkId": 5}},
        )
        assert abs(score - 2/3) < 0.001

    def test_no_overlap(self):
        score = param_f1(
            {"endpoint": "GET /v1/items", "params": {"x": 1}},
            {"endpoint": "GET /v1/items", "params": {"y": 2}},
        )
        assert score == 0.0

    def test_extra_params_predicted(self):
        # predicted has extra param: precision < 1
        score = param_f1(
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10, "extra": 1}},
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10}},
        )
        assert 0 < score < 1.0

    def test_chained_union_of_steps(self):
        predicted = {"steps": [
            {"endpoint": "GET /v1/items", "params": {}},
            {"endpoint": "GET /v1/items/{id}", "params": {"id": 1}},
        ]}
        expected = {"steps": [
            {"endpoint": "GET /v1/items", "params": {}},
            {"endpoint": "GET /v1/items/{id}", "params": {"id": 1}},
        ]}
        assert param_f1(predicted, expected) == 1.0


class TestScoreRecord:
    def test_perfect_score(self):
        api_call = {"endpoint": "GET /v1/items", "params": {"pageSize": 10}}
        result = score_record(api_call, api_call)
        assert result["format_score"] == 1.0
        assert result["param_f1"] == 1.0
        assert result["composite_score"] == 1.0
        assert result["band"] == "GOLD"

    def test_format_fail_short_circuits(self):
        result = score_record(None, {"endpoint": "GET /v1/items", "params": {}})
        assert result["format_score"] == 0.0
        assert result["param_f1"] == 0.0
        assert result["band"] == "FAIL"

    def test_no_executable_by_default(self):
        api_call = {"endpoint": "GET /v1/items", "params": {}}
        result = score_record(api_call, api_call)
        assert result["executable"] is None


class TestBand:
    def test_gold(self):
        assert _band(0.95) == "GOLD"
        assert _band(1.0) == "GOLD"

    def test_silver(self):
        assert _band(0.75) == "SILVER"

    def test_bronze(self):
        assert _band(0.5) == "BRONZE"

    def test_fail(self):
        assert _band(0.3) == "FAIL"
        assert _band(0.0) == "FAIL"
