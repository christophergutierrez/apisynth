"""Tests for run.py --offline generation.

Offline mode must produce well-formed records WITHOUT any network access, and
must not change the default (live-verified) behaviour. The `no_network` fixture
makes any call to urlopen fail the test, proving the offline path never hits it.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run
from run import run_one, gen_questions, _CHAINED
from utils import FIELD_QUESTION, FIELD_API_CALL

APIS = Path(__file__).parent.parent / "apis" / "example"


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test if anything tries to touch the network."""
    def boom(*a, **k):
        raise AssertionError("offline mode must not touch the network")
    monkeypatch.setattr(run.urllib.request, "urlopen", boom)


def _load(name):
    with open(APIS / name / "config.yaml") as f:
        return yaml.safe_load(f)


def test_offline_standard_record(tmp_path, no_network):
    cfg = _load("episodes")
    out = tmp_path / "training.jsonl"
    ok, msg = run_one(cfg, "", out, "Get 10 episodes", {"pageSize": 10}, offline=True)
    assert ok
    assert "unverified" in msg.lower()
    rec = json.loads(out.read_text().strip())
    assert rec[FIELD_QUESTION] == "Get 10 episodes"
    assert rec[FIELD_API_CALL]["endpoint"] == "GET /external/v1/content/episodes"
    assert rec[FIELD_API_CALL]["params"] == {"pageSize": 10}
    assert rec["schema"]
    assert rec["intent_category"] == "paginated"


def test_offline_chained_record(tmp_path, no_network):
    cfg = _load("episode")
    out = tmp_path / "training.jsonl"
    ok, _ = run_one(cfg, "", out, "show me an episode", {_CHAINED: True}, offline=True)
    assert ok
    rec = json.loads(out.read_text().strip())
    steps = rec[FIELD_API_CALL]["steps"]
    assert len(steps) == 2
    assert steps[0]["endpoint"] == "GET /external/v1/content/episodes"
    assert steps[1]["params"]["episodeId"] == "{{steps.0.id}}"
    assert rec["intent_category"] == "chained"


def test_offline_byid_record(tmp_path, no_network):
    cfg = _load("episode")
    out = tmp_path / "training.jsonl"
    ok, _ = run_one(cfg, "", out, "Get episode 101", {"episodeId": 101}, offline=True)
    assert ok
    rec = json.loads(out.read_text().strip())
    assert rec[FIELD_API_CALL]["endpoint"] == "GET /external/v1/content/episodes/{episodeId}"
    assert rec[FIELD_API_CALL]["params"] == {"episodeId": 101}
    assert rec["intent_category"] == "by-id"


def test_offline_generates_full_variant(tmp_path, no_network):
    """Generate the whole question set for a filtered variant, fully offline."""
    cfg = _load("episodes")
    out = tmp_path / "training.jsonl"
    questions = gen_questions(cfg, cfg.get("status") or {}, {"networkId": 3}, target=25)
    assert questions, "expected generated questions"
    for q, p in questions:
        run_one(cfg, "", out, q, p, offline=True)
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == len(questions)
    for rec in lines:
        assert rec[FIELD_QUESTION]
        assert rec[FIELD_API_CALL]["endpoint"].startswith("GET ")
        assert "intent_category" in rec


def test_live_path_still_validates(tmp_path, monkeypatch):
    """Guard: without --offline, run_one must still validate via the API."""
    cfg = _load("episodes")
    out = tmp_path / "training.jsonl"
    calls = {"n": 0}

    def fake_validate(*a, **k):
        calls["n"] += 1
        return True

    monkeypatch.setattr(run, "api_validate", fake_validate)
    ok, _ = run_one(cfg, "tok", out, "Get 10 episodes", {"pageSize": 10}, offline=False)
    assert ok
    assert calls["n"] == 1
