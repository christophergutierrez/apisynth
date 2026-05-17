"""Unit tests for scripts/utils.py — extract_schema and infer_intent."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from utils import (
    extract_schema,
    infer_intent,
    INTENT_BARE_LIST,
    INTENT_PAGINATED,
    INTENT_FILTERED,
    INTENT_BY_ID,
    INTENT_CHAINED,
    INTENT_NO_PARAM,
)


# ── extract_schema ─────────────────────────────────────────────────────────

class TestExtractSchema:
    def _cfg(self, path, params=None, path_params=None):
        cfg = {"endpoint": {"method": "GET", "path": path}}
        if params:
            cfg["params"] = params
        if path_params:
            cfg["path_params"] = path_params
        return cfg

    def test_list_endpoint_params_only(self):
        cfg = self._cfg(
            "/v1/widgets",
            params={"pageSize": {"type": "integer"}, "name": {"type": "string"}},
        )
        result = extract_schema(cfg)
        assert "GET /v1/widgets" in result
        assert "pageSize (integer)" in result
        assert "name (string)" in result
        assert "path params" not in result

    def test_path_param_endpoint(self):
        cfg = self._cfg(
            "/v1/widgets/{id}",
            path_params={"id": {"type": "integer"}},
        )
        result = extract_schema(cfg)
        assert "GET /v1/widgets/{id}" in result
        assert "path params: id (integer)" in result
        assert "params: (none)" in result

    def test_mixed_params_and_path_params(self):
        cfg = self._cfg(
            "/v1/episodes/{episodeId}",
            params={"lang": {"type": "string"}},
            path_params={"episodeId": {"type": "integer"}},
        )
        result = extract_schema(cfg)
        assert "GET /v1/episodes/{episodeId}" in result
        assert "lang (string)" in result
        assert "episodeId (integer)" in result

    def test_no_params_endpoint(self):
        cfg = self._cfg("/v1/me")
        result = extract_schema(cfg)
        assert "GET /v1/me" in result
        assert "params: (none)" in result

    def test_base_url_extracts_path(self):
        cfg = {
            "endpoint": {
                "method": "GET",
                "base_url": "https://api.example.com/v1/widgets",
                "path": "/v1/widgets",
            }
        }
        result = extract_schema(cfg)
        assert "GET /v1/widgets" in result

    def test_missing_type_defaults_to_string(self):
        cfg = self._cfg("/v1/items", params={"q": {}})
        result = extract_schema(cfg)
        assert "q (string)" in result


# ── infer_intent ───────────────────────────────────────────────────────────

class TestInferIntent:
    def test_chained(self):
        api_call = {"steps": [{"endpoint": "GET /v1/items", "params": {}}]}
        assert infer_intent(api_call) == INTENT_CHAINED

    def test_by_id(self):
        api_call = {"endpoint": "GET /v1/items/{id}", "params": {"id": 42}}
        path_params_cfg = {"id": {"type": "integer"}}
        assert infer_intent(api_call, path_params_cfg) == INTENT_BY_ID

    def test_no_param(self):
        api_call = {"endpoint": "GET /v1/me", "params": {}}
        assert infer_intent(api_call) == INTENT_NO_PARAM

    def test_filtered(self):
        api_call = {"endpoint": "GET /v1/items", "params": {"networkId": 5}}
        assert infer_intent(api_call) == INTENT_FILTERED

    def test_paginated(self):
        api_call = {"endpoint": "GET /v1/items", "params": {"pageSize": 10}}
        assert infer_intent(api_call) == INTENT_PAGINATED

    def test_bare_list_empty_params(self):
        api_call = {"endpoint": "GET /v1/items", "params": {}}
        assert infer_intent(api_call) == INTENT_NO_PARAM

    def test_filtered_with_pagesize(self):
        api_call = {"endpoint": "GET /v1/items", "params": {"pageSize": 10, "networkId": 3}}
        assert infer_intent(api_call) == INTENT_FILTERED

    def test_by_id_takes_priority_over_filtered(self):
        api_call = {"endpoint": "GET /v1/items/{id}", "params": {"id": 1, "lang": "en"}}
        path_params_cfg = {"id": {"type": "integer"}}
        assert infer_intent(api_call, path_params_cfg) == INTENT_BY_ID
