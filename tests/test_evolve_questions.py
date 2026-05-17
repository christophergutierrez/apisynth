"""Unit tests for scripts/evolve_questions.py — Evol-Instruct question mutation."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from evolve_questions import _evolve_record, _AXIS_MODELS, _PROMPTS, evolve_file


class TestMutationPromptConstruction:
    def test_all_axes_have_prompts(self):
        assert "constraint" in _PROMPTS
        assert "context" in _PROMPTS
        assert "complexity" in _PROMPTS

    def test_prompt_includes_question(self):
        for axis, template in _PROMPTS.items():
            assert "{question}" in template, f"{axis} prompt missing {{question}}"

    def test_prompt_includes_api_call(self):
        for axis, template in _PROMPTS.items():
            assert "{api_call}" in template, f"{axis} prompt missing {{api_call}}"


class TestModelSelection:
    def test_simple_axes_use_haiku(self):
        assert "haiku" in _AXIS_MODELS["constraint"].lower()
        assert "haiku" in _AXIS_MODELS["context"].lower()

    def test_complexity_uses_sonnet(self):
        assert "sonnet" in _AXIS_MODELS["complexity"].lower()


class TestEvolveRecord:
    def _record(self):
        return {
            "question": "list all episodes",
            "api_call": {"endpoint": "GET /v1/episodes", "params": {}},
            "schema": "GET /v1/episodes\nparams: pageSize (integer)",
        }

    def test_evolved_record_has_source_tag(self):
        record = self._record()
        with patch("evolve_questions._call_claude", return_value="show me all episodes please"):
            evolved = _evolve_record(record, "constraint")
        assert evolved is not None
        assert evolved["source"] == "evol"

    def test_evolved_record_has_axis_tag(self):
        record = self._record()
        with patch("evolve_questions._call_claude", return_value="list episodes (top 10)"):
            evolved = _evolve_record(record, "constraint")
        assert evolved["evol_axis"] == "constraint"

    def test_evolved_record_has_seed_question(self):
        record = self._record()
        with patch("evolve_questions._call_claude", return_value="get all episodes now"):
            evolved = _evolve_record(record, "context")
        assert evolved["evol_seed"] == "list all episodes"

    def test_evolved_record_preserves_api_call(self):
        record = self._record()
        with patch("evolve_questions._call_claude", return_value="fetch episodes for Q3"):
            evolved = _evolve_record(record, "context")
        assert evolved["api_call"] == record["api_call"]

    def test_returns_none_on_claude_error(self):
        record = self._record()
        with patch("evolve_questions._call_claude", side_effect=Exception("API error")):
            evolved = _evolve_record(record, "constraint")
        assert evolved is None

    def test_returns_none_if_unchanged(self):
        record = self._record()
        with patch("evolve_questions._call_claude", return_value="list all episodes"):
            evolved = _evolve_record(record, "constraint")
        assert evolved is None


class TestEvolveFile:
    def _write_records(self, path: Path, records: list[dict]):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_evolve_file_writes_evolved_records(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        records = [
            {"question": "list episodes", "api_call": {"endpoint": "GET /v1/eps", "params": {}}},
        ]
        self._write_records(input_path, records)

        with patch("evolve_questions._call_claude", return_value="fetch all episodes"):
            count = evolve_file(input_path, input_path, per_record=1, dry_run=False, sample=None)

        assert count >= 1
        lines = [l for l in input_path.read_text().splitlines() if l.strip()]
        # Original + at least one evolved
        assert len(lines) >= 2
        evolved = [json.loads(l) for l in lines if json.loads(l).get("source") == "evol"]
        assert len(evolved) >= 1

    def test_dry_run_writes_nothing(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        records = [{"question": "list items", "api_call": {"endpoint": "GET /v1/items", "params": {}}}]
        self._write_records(input_path, records)
        original_content = input_path.read_text()

        with patch("evolve_questions._call_claude", return_value="show all items"):
            evolve_file(input_path, input_path, per_record=1, dry_run=True)

        assert input_path.read_text() == original_content

    def test_skips_already_evolved_records(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        records = [
            {"question": "list items", "api_call": {}, "source": "evol"},
        ]
        self._write_records(input_path, records)

        with patch("evolve_questions._call_claude", return_value="show all items") as mock_claude:
            count = evolve_file(input_path, input_path, per_record=1)

        assert count == 0
        assert not mock_claude.called
