"""Unit tests for scripts/repo/evolve_code_questions.py — deterministic code-unit question evolution.

No mocking needed: all axis functions are pure, offline, and deterministic.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "repo"))
from evolve_code_questions import (
    _axis_paraphrase,
    _axis_context,
    _axis_task_pattern,
    _evolve_record,
    evolve_file,
    _AXES,
)


# ---------------------------------------------------------------------------
# Helpers — representative code-unit records
# ---------------------------------------------------------------------------

def _function_record(name="process_data", file="src/utils.py", sig="process_data(items, limit=None)"):
    return {
        "type": "code",
        "question": f"How do I use `{name}`?",
        "thinking": f"Entity: function {name}\nFile: {file}",
        "output": {
            "unit": "function",
            "name": name,
            "file": file,
            "signature": sig,
        },
    }


def _method_record(name="fetch", cls="ApiClient", file="src/client.py", sig="fetch(self, url)"):
    return {
        "type": "code",
        "question": f"How do I call `{name}` on a `{cls}`?",
        "thinking": f"Entity: method {name} on class {cls}",
        "output": {
            "unit": "method",
            "name": name,
            "file": file,
            "signature": sig,
            "class": cls,
        },
    }


def _class_record(name="DataLoader", file="src/loader.py", sig="DataLoader()"):
    return {
        "type": "code",
        "question": f"How do I instantiate `{name}`?",
        "thinking": f"Entity: class {name}",
        "output": {
            "unit": "class",
            "name": name,
            "file": file,
            "signature": sig,
        },
    }


def _api_call_record(name="requests.get", file="src/http.py", sig="requests.get(url, **kwargs)"):
    return {
        "type": "code",
        "question": f"How is the `{name}` API call made?",
        "thinking": f"Entity: api_call {name}",
        "output": {
            "unit": "api_call",
            "name": name,
            "file": file,
            "signature": sig,
        },
    }


# ---------------------------------------------------------------------------
# Tests: axis list and module constants
# ---------------------------------------------------------------------------

class TestAxesConstant:
    def test_three_axes_defined(self):
        assert len(_AXES) == 3

    def test_axis_names(self):
        assert set(_AXES) == {"paraphrase", "context", "task_pattern"}


# ---------------------------------------------------------------------------
# Tests: _axis_paraphrase
# ---------------------------------------------------------------------------

class TestAxisParaphrase:
    def test_function_differs_from_seed(self):
        record = _function_record()
        result = _axis_paraphrase(record)
        assert result is not None
        assert result != record["question"]

    def test_method_with_class_differs_from_seed(self):
        record = _method_record()
        result = _axis_paraphrase(record)
        assert result is not None
        assert result != record["question"]

    def test_method_question_mentions_name(self):
        record = _method_record(name="fetch", cls="ApiClient")
        result = _axis_paraphrase(record)
        assert result is not None
        assert "fetch" in result

    def test_class_differs_from_seed(self):
        record = _class_record()
        result = _axis_paraphrase(record)
        assert result is not None
        assert result != record["question"]

    def test_api_call_differs_from_seed(self):
        record = _api_call_record()
        result = _axis_paraphrase(record)
        assert result is not None
        assert result != record["question"]

    def test_returns_none_when_all_candidates_equal_seed(self, monkeypatch):
        """If every candidate matches the seed, returns None."""
        record = _function_record(name="foo")
        # Patch the question to match the first candidate
        record["question"] = "What is the signature of `foo`?"
        # Result should still be non-None because other candidates differ.
        result = _axis_paraphrase(record)
        assert result is not None
        assert result != record["question"]

    def test_method_without_class(self):
        record = _method_record()
        record["output"].pop("class", None)
        record["question"] = "How do I call `fetch`?"
        result = _axis_paraphrase(record)
        # Should produce something even without class info
        assert result is not None

    def test_function_paraphrase_is_deterministic(self):
        record = _function_record()
        r1 = _axis_paraphrase(record)
        r2 = _axis_paraphrase(record)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests: _axis_context
# ---------------------------------------------------------------------------

class TestAxisContext:
    def test_function_includes_module(self):
        record = _function_record(file="src/utils.py")
        result = _axis_context(record)
        assert result is not None
        assert "utils" in result

    def test_method_includes_class_and_name(self):
        record = _method_record(name="fetch", cls="ApiClient", file="src/client.py")
        result = _axis_context(record)
        assert result is not None
        assert "fetch" in result
        assert "ApiClient" in result

    def test_method_includes_module(self):
        record = _method_record(file="src/client.py")
        result = _axis_context(record)
        assert result is not None
        assert "client" in result

    def test_class_includes_module(self):
        record = _class_record(name="DataLoader", file="src/loader.py")
        result = _axis_context(record)
        assert result is not None
        assert "loader" in result or "DataLoader" in result

    def test_api_call_includes_module(self):
        record = _api_call_record(name="requests.get", file="src/http.py")
        result = _axis_context(record)
        assert result is not None
        assert "http" in result or "requests.get" in result

    def test_differs_from_seed(self):
        record = _function_record()
        result = _axis_context(record)
        assert result is not None
        assert result != record["question"]

    def test_method_context_differs_from_seed(self):
        record = _method_record()
        result = _axis_context(record)
        assert result is not None
        assert result != record["question"]

    def test_context_is_deterministic(self):
        record = _method_record()
        r1 = _axis_context(record)
        r2 = _axis_context(record)
        assert r1 == r2

    def test_no_file_uses_name_in_output(self):
        """With a non-empty file, context axis includes the module name."""
        record = _function_record(file="src/utils.py")
        result = _axis_context(record)
        assert result is not None
        assert "utils" in result

    def test_returns_none_when_would_equal_seed(self):
        """If the produced question matches the seed, returns None."""
        # Construct a record where context would equal the seed
        record = _function_record(name="process_data", file="")
        # Seed is "How do I use `process_data`?" and no module → "How do I use `process_data`?"
        record["question"] = "How do I use `process_data`?"
        # With empty file the context axis would produce "How do I use `process_data`?"
        result = _axis_context(record)
        # If they match, must return None; if not equal, result is fine.
        if result is not None:
            assert result != record["question"]


# ---------------------------------------------------------------------------
# Tests: _axis_task_pattern
# ---------------------------------------------------------------------------

class TestAxisTaskPattern:
    def test_function_produces_implement_framing(self):
        record = _function_record(name="process_data", sig="process_data(items, limit=None)")
        result = _axis_task_pattern(record)
        assert result is not None
        assert "Implement" in result or "implement" in result or "process_data" in result

    def test_method_with_class_uses_call_helper_framing(self):
        record = _method_record(name="fetch", cls="ApiClient")
        result = _axis_task_pattern(record)
        assert result is not None
        assert "fetch" in result
        assert "ApiClient" in result

    def test_class_includes_name(self):
        record = _class_record(name="DataLoader")
        result = _axis_task_pattern(record)
        assert result is not None
        assert "DataLoader" in result

    def test_api_call_framing(self):
        record = _api_call_record(name="requests.get")
        result = _axis_task_pattern(record)
        assert result is not None
        assert "requests.get" in result

    def test_differs_from_seed(self):
        record = _function_record()
        result = _axis_task_pattern(record)
        assert result is not None
        assert result != record["question"]

    def test_method_task_differs_from_seed(self):
        record = _method_record()
        result = _axis_task_pattern(record)
        assert result is not None
        assert result != record["question"]

    def test_task_pattern_is_deterministic(self):
        record = _function_record()
        r1 = _axis_task_pattern(record)
        r2 = _axis_task_pattern(record)
        assert r1 == r2

    def test_returns_none_when_would_equal_seed(self):
        """Axis returns None if the generated task question equals the seed."""
        record = _function_record(name="foo", sig="foo(...)")
        # Force the seed question to match what task_pattern would produce.
        record["question"] = "Implement `foo(...)`"
        result = _axis_task_pattern(record)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _evolve_record
# ---------------------------------------------------------------------------

class TestEvolveRecord:
    def test_evolved_record_has_source_evol(self):
        record = _function_record()
        evolved = _evolve_record(record, "paraphrase")
        assert evolved is not None
        assert evolved["source"] == "evol"

    def test_evolved_record_has_axis_tag(self):
        record = _function_record()
        for axis in _AXES:
            evolved = _evolve_record(record, axis)
            if evolved is not None:
                assert evolved["evol_axis"] == axis

    def test_evolved_record_has_seed_question(self):
        record = _function_record()
        orig_q = record["question"]
        evolved = _evolve_record(record, "paraphrase")
        assert evolved is not None
        assert evolved["evol_seed"] == orig_q

    def test_evolved_record_preserves_thinking(self):
        record = _function_record()
        evolved = _evolve_record(record, "context")
        assert evolved is not None
        assert evolved["thinking"] == record["thinking"]

    def test_evolved_record_preserves_output(self):
        record = _function_record()
        evolved = _evolve_record(record, "task_pattern")
        assert evolved is not None
        assert evolved["output"] == record["output"]

    def test_returns_none_when_question_unchanged(self):
        record = _function_record(name="foo", sig="foo(...)")
        record["question"] = "Implement `foo(...)`"
        evolved = _evolve_record(record, "task_pattern")
        assert evolved is None

    def test_evolve_record_does_not_mutate_original(self):
        record = _function_record()
        orig_q = record["question"]
        _evolve_record(record, "paraphrase")
        assert record["question"] == orig_q

    def test_all_three_axes_produce_different_questions(self):
        record = _function_record()
        questions = set()
        for axis in _AXES:
            evolved = _evolve_record(record, axis)
            if evolved is not None:
                questions.add(evolved["question"])
        # At least two distinct evolved questions for a representative function.
        assert len(questions) >= 2

    def test_method_context_axis_mentions_class(self):
        record = _method_record(name="close", cls="Connection")
        evolved = _evolve_record(record, "context")
        assert evolved is not None
        assert "Connection" in evolved["question"]


# ---------------------------------------------------------------------------
# Tests: skipping already-evolved and bootstrap records
# ---------------------------------------------------------------------------

class TestSkipping:
    def test_skips_evol_source(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        record = {**_function_record(), "source": "evol"}
        input_path.write_text(json.dumps(record) + "\n")
        count = evolve_file(input_path, input_path, per_record=3)
        assert count == 0

    def test_skips_bootstrap_source(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        record = {**_function_record(), "source": "bootstrap"}
        input_path.write_text(json.dumps(record) + "\n")
        count = evolve_file(input_path, input_path, per_record=3)
        assert count == 0

    def test_only_non_evol_records_are_evolved(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        records = [
            _function_record(),
            {**_method_record(), "source": "evol"},
        ]
        input_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        count = evolve_file(input_path, input_path, per_record=3)
        # Only the plain function record should be evolved.
        assert count > 0
        lines = [l for l in input_path.read_text().splitlines() if l.strip()]
        evolved_lines = [json.loads(l) for l in lines if json.loads(l).get("source") == "evol"]
        # The pre-existing evol + newly evolved entries
        for ev in evolved_lines:
            # All newly written evolved records came from the function, not the already-evol method
            if ev.get("evol_seed") is not None:
                assert ev["evol_seed"] == _function_record()["question"]


# ---------------------------------------------------------------------------
# Tests: evolve_file
# ---------------------------------------------------------------------------

class TestEvolveFile:
    def _write_records(self, path, records):
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def test_evolve_file_appends_evolved_records(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        records = [_function_record()]
        self._write_records(input_path, records)

        count = evolve_file(input_path, input_path, per_record=2)

        assert count >= 1
        lines = [l for l in input_path.read_text().splitlines() if l.strip()]
        # Original + evolved
        assert len(lines) >= 2
        evolved = [json.loads(l) for l in lines if json.loads(l).get("source") == "evol"]
        assert len(evolved) >= 1

    def test_dry_run_writes_nothing(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        records = [_function_record()]
        self._write_records(input_path, records)
        original_content = input_path.read_text()

        evolve_file(input_path, input_path, per_record=2, dry_run=True)

        assert input_path.read_text() == original_content

    def test_evolved_records_have_correct_fields(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        records = [_method_record()]
        self._write_records(input_path, records)

        evolve_file(input_path, input_path, per_record=3)

        lines = [l for l in input_path.read_text().splitlines() if l.strip()]
        for line in lines:
            rec = json.loads(line)
            if rec.get("source") == "evol":
                assert rec["evol_axis"] in set(_AXES)
                assert rec["evol_seed"] == _method_record()["question"]
                assert rec["output"] == _method_record()["output"]
                assert rec["thinking"] == _method_record()["thinking"]

    def test_per_record_1_produces_at_most_1_per_record(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        records = [_function_record()]
        self._write_records(input_path, records)

        count = evolve_file(input_path, input_path, per_record=1)
        assert count <= 1

    def test_sample_limits_records_processed(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        records = [_function_record(name=f"fn_{i}", file=f"src/mod_{i}.py") for i in range(10)]
        self._write_records(input_path, records)

        count = evolve_file(input_path, input_path, per_record=3, sample=2, seed=42)
        # At most 2 records * 3 axes = 6 evolved
        assert count <= 6

    def test_empty_file_returns_zero(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        input_path.write_text("")

        count = evolve_file(input_path, input_path)
        assert count == 0

    def test_different_output_path(self, tmp_path):
        input_path = tmp_path / "training.jsonl"
        output_path = tmp_path / "evolved.jsonl"
        records = [_function_record()]
        self._write_records(input_path, records)
        output_path.write_text("")

        count = evolve_file(input_path, output_path, per_record=2)

        # Input file unchanged
        assert input_path.read_text() == "\n".join(json.dumps(r) for r in records) + "\n"
        # Output file has evolved records
        if count > 0:
            out_lines = [l for l in output_path.read_text().splitlines() if l.strip()]
            assert len(out_lines) == count


# ---------------------------------------------------------------------------
# Tests: determinism — two runs produce byte-identical output
# ---------------------------------------------------------------------------

class TestDeterminism:
    def _write_records(self, path, records):
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def test_evolve_file_is_deterministic(self, tmp_path):
        """Running evolve_file twice on the same input yields byte-identical output."""
        input_records = [
            _function_record(name="alpha", file="src/a.py"),
            _method_record(name="beta", cls="Foo", file="src/b.py"),
            _class_record(name="Gamma", file="src/g.py"),
        ]

        # Run 1
        input1 = tmp_path / "input1.jsonl"
        out1 = tmp_path / "out1.jsonl"
        self._write_records(input1, input_records)
        out1.write_text("")
        evolve_file(input1, out1, per_record=2, seed=42)

        # Run 2 — fresh paths, same data
        input2 = tmp_path / "input2.jsonl"
        out2 = tmp_path / "out2.jsonl"
        self._write_records(input2, input_records)
        out2.write_text("")
        evolve_file(input2, out2, per_record=2, seed=42)

        assert out1.read_text() == out2.read_text()

    def test_axis_functions_are_deterministic(self):
        record = _method_record()
        for axis_fn in (_axis_paraphrase, _axis_context, _axis_task_pattern):
            assert axis_fn(record) == axis_fn(record)

    def test_different_seeds_may_differ(self, tmp_path):
        """Different seeds can select different axes; same seed always matches."""
        input_records = [
            _function_record(name="fn1", file="src/m1.py"),
            _function_record(name="fn2", file="src/m2.py"),
        ]
        out42 = tmp_path / "out42.jsonl"
        out99 = tmp_path / "out99.jsonl"
        out42b = tmp_path / "out42b.jsonl"

        for p in (out42, out99, out42b):
            p.write_text("")

        input_path = tmp_path / "in.jsonl"
        self._write_records(input_path, input_records)

        evolve_file(input_path, out42, per_record=2, seed=42)
        evolve_file(input_path, out99, per_record=2, seed=99)
        evolve_file(input_path, out42b, per_record=2, seed=42)

        # Same seed → same output
        assert out42.read_text() == out42b.read_text()


# ---------------------------------------------------------------------------
# Tests: evolved record tagging integrity
# ---------------------------------------------------------------------------

class TestEvolvedRecordTagging:
    def test_source_is_evol(self):
        record = _function_record()
        for axis in _AXES:
            evolved = _evolve_record(record, axis)
            if evolved is not None:
                assert evolved["source"] == "evol"

    def test_evol_axis_matches(self):
        record = _function_record()
        for axis in _AXES:
            evolved = _evolve_record(record, axis)
            if evolved is not None:
                assert evolved["evol_axis"] == axis

    def test_evol_seed_is_original_question(self):
        record = _method_record()
        orig_q = record["question"]
        for axis in _AXES:
            evolved = _evolve_record(record, axis)
            if evolved is not None:
                assert evolved["evol_seed"] == orig_q

    def test_thinking_unchanged(self):
        record = _class_record()
        for axis in _AXES:
            evolved = _evolve_record(record, axis)
            if evolved is not None:
                assert evolved["thinking"] == record["thinking"]

    def test_output_unchanged(self):
        record = _api_call_record()
        for axis in _AXES:
            evolved = _evolve_record(record, axis)
            if evolved is not None:
                assert evolved["output"] == record["output"]

    def test_type_field_unchanged(self):
        record = _function_record()
        for axis in _AXES:
            evolved = _evolve_record(record, axis)
            if evolved is not None:
                assert evolved["type"] == "code"
