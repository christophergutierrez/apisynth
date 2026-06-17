"""Tests for Milestone 2.1 — QOC and enriched linear thinking traces.

Covers:
- QOC style produces non-empty, deterministic traces for all four unit types
- QOC trace contains the unit name and structural markers
- Linear remains the default and contains "Entity:" / "Use:"
- Cross-process determinism for QOC (subprocess with different PYTHONHASHSEED)
- Classless method under QOC must not contain "None"
- generate_code_thinking fallback for unknown style
- generate_from_code style="qoc" yields QOC records; default yields linear
- config.thinking_style="hybrid" → QOC; "deterministic" → linear
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.repo.loader import RepoConfig
from scripts.repo.generate_from_code import (
    generate_code_thinking,
    generate_from_code,
    generate_from_repo,
    _make_thinking,
    _make_record,
    _split_records,
    _style_from_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_unit(unit_type="function", name="do_thing", file="mod.py", **kwargs):
    """Construct a minimal scanner unit dict."""
    u = {"type": unit_type, "name": name, "file": file, "lineno": 10}
    u.update(kwargs)
    return u


def _make_config(thinking_style="deterministic"):
    """Build a minimal RepoConfig without touching disk (uses /tmp as path sentinel)."""
    import tempfile, pathlib
    # Need a valid directory for RepoConfig — use a known temp dir
    return RepoConfig(
        name="test-thinking",
        path="/tmp",
        include=["**/*.py"],
        exclude=[],
        thinking_style=thinking_style,
    )


_ALL_UNIT_TYPES = [
    _make_unit("function", "my_func", file="pkg/utils.py"),
    _make_unit("method", "my_method", file="pkg/cls.py", **{"class": "MyClass"}),
    _make_unit("class", "MyClass", file="pkg/cls.py"),
    _make_unit("api_call", "requests.get", file="pkg/api.py"),
]


# ---------------------------------------------------------------------------
# QOC: non-empty traces for all four unit types
# ---------------------------------------------------------------------------

class TestQOCNonEmpty:
    @pytest.mark.parametrize("unit", _ALL_UNIT_TYPES, ids=lambda u: u["type"])
    def test_qoc_nonempty(self, unit):
        trace = generate_code_thinking(unit, style="qoc")
        assert trace, f"QOC trace is empty for {unit['type']}"

    @pytest.mark.parametrize("unit", _ALL_UNIT_TYPES, ids=lambda u: u["type"])
    def test_qoc_contains_unit_name(self, unit):
        trace = generate_code_thinking(unit, style="qoc")
        assert unit["name"] in trace, (
            f"Unit name {unit['name']!r} not found in QOC trace for {unit['type']}:\n{trace}"
        )


# ---------------------------------------------------------------------------
# QOC: structural markers present
# ---------------------------------------------------------------------------

class TestQOCStructuralMarkers:
    @pytest.mark.parametrize("unit", _ALL_UNIT_TYPES, ids=lambda u: u["type"])
    def test_qoc_has_question_marker(self, unit):
        trace = generate_code_thinking(unit, style="qoc")
        assert "Question:" in trace, (
            f"'Question:' marker missing from QOC trace for {unit['type']}:\n{trace}"
        )

    @pytest.mark.parametrize("unit", _ALL_UNIT_TYPES, ids=lambda u: u["type"])
    def test_qoc_has_option_marker(self, unit):
        trace = generate_code_thinking(unit, style="qoc")
        assert "Option" in trace, (
            f"'Option' marker missing from QOC trace for {unit['type']}:\n{trace}"
        )

    @pytest.mark.parametrize("unit", _ALL_UNIT_TYPES, ids=lambda u: u["type"])
    def test_qoc_has_criteria_marker(self, unit):
        trace = generate_code_thinking(unit, style="qoc")
        assert "Criteria:" in trace, (
            f"'Criteria:' marker missing from QOC trace for {unit['type']}:\n{trace}"
        )


# ---------------------------------------------------------------------------
# QOC: determinism (same process)
# ---------------------------------------------------------------------------

class TestQOCDeterminism:
    @pytest.mark.parametrize("unit", _ALL_UNIT_TYPES, ids=lambda u: u["type"])
    def test_qoc_same_process_deterministic(self, unit):
        t1 = generate_code_thinking(unit, style="qoc")
        t2 = generate_code_thinking(unit, style="qoc")
        assert t1 == t2, f"QOC trace not deterministic for {unit['type']}"


# ---------------------------------------------------------------------------
# Linear: remains default and contains Entity/Use markers
# ---------------------------------------------------------------------------

class TestLinearDefault:
    @pytest.mark.parametrize("unit", _ALL_UNIT_TYPES, ids=lambda u: u["type"])
    def test_linear_is_default(self, unit):
        """generate_code_thinking with no style arg should produce linear (Entity: marker)."""
        trace = generate_code_thinking(unit)
        assert "Entity:" in trace, (
            f"'Entity:' marker missing from default (linear) trace for {unit['type']}:\n{trace}"
        )

    @pytest.mark.parametrize("unit", _ALL_UNIT_TYPES, ids=lambda u: u["type"])
    def test_linear_has_use_marker(self, unit):
        trace = generate_code_thinking(unit, style="linear")
        assert "Use:" in trace, (
            f"'Use:' marker missing from linear trace for {unit['type']}:\n{trace}"
        )

    @pytest.mark.parametrize("unit", _ALL_UNIT_TYPES, ids=lambda u: u["type"])
    def test_linear_has_entity_marker(self, unit):
        trace = generate_code_thinking(unit, style="linear")
        assert "Entity:" in trace, (
            f"'Entity:' marker missing from linear trace for {unit['type']}:\n{trace}"
        )


# ---------------------------------------------------------------------------
# Classless method: no "None" in either style
# ---------------------------------------------------------------------------

class TestClasslessMethod:
    def _classless_unit(self):
        return {"type": "method", "name": "orphan_method", "file": "a.py", "lineno": 5}

    def test_linear_classless_no_none(self):
        unit = self._classless_unit()
        trace = generate_code_thinking(unit, style="linear")
        assert "None" not in trace, f"Literal 'None' found in linear trace:\n{trace}"

    def test_qoc_classless_no_none(self):
        unit = self._classless_unit()
        trace = generate_code_thinking(unit, style="qoc")
        assert "None" not in trace, f"Literal 'None' found in QOC trace:\n{trace}"

    def test_make_thinking_classless_no_none(self):
        """_make_thinking wrapper must also be safe for classless methods."""
        unit = self._classless_unit()
        trace = _make_thinking(unit)
        assert "None" not in trace

    def test_qoc_classless_still_has_markers(self):
        unit = self._classless_unit()
        trace = generate_code_thinking(unit, style="qoc")
        assert "Question:" in trace
        assert "Option" in trace
        assert "Criteria:" in trace


# ---------------------------------------------------------------------------
# Unknown style falls back to linear
# ---------------------------------------------------------------------------

class TestUnknownStyleFallback:
    @pytest.mark.parametrize("unit", _ALL_UNIT_TYPES, ids=lambda u: u["type"])
    def test_bogus_style_falls_back_to_linear(self, unit):
        trace = generate_code_thinking(unit, style="bogus")
        assert "Entity:" in trace, (
            f"Bogus style did not fall back to linear for {unit['type']}:\n{trace}"
        )

    def test_generate_code_thinking_bogus_is_same_as_linear(self):
        unit = _make_unit("function", "check_me", file="x.py")
        linear_trace = generate_code_thinking(unit, style="linear")
        bogus_trace = generate_code_thinking(unit, style="bogus")
        assert linear_trace == bogus_trace, (
            "Bogus style did not produce identical output to linear"
        )


# ---------------------------------------------------------------------------
# generate_from_code: style parameter wires through
# ---------------------------------------------------------------------------

class TestGenerateFromCodeStyle:
    def test_explicit_qoc_style_yields_qoc_records(self):
        config = _make_config()
        units = [_make_unit("function", "alpha", file="a.py")]
        records = generate_from_code(config, units, style="qoc")
        assert len(records) == 1
        assert "Question:" in records[0]["thinking"], (
            f"Expected QOC thinking, got: {records[0]['thinking']}"
        )

    def test_explicit_linear_style_yields_linear_records(self):
        config = _make_config()
        units = [_make_unit("function", "beta", file="b.py")]
        records = generate_from_code(config, units, style="linear")
        assert len(records) == 1
        assert "Entity:" in records[0]["thinking"], (
            f"Expected linear thinking, got: {records[0]['thinking']}"
        )

    def test_default_style_is_linear(self):
        """Calling generate_from_code with no style arg must yield linear traces."""
        config = _make_config(thinking_style="deterministic")
        units = [_make_unit("class", "Gamma", file="g.py")]
        records = generate_from_code(config, units)
        assert "Entity:" in records[0]["thinking"]

    def test_config_hybrid_style_yields_qoc(self):
        """config.thinking_style='hybrid' → QOC when style=None."""
        config = _make_config(thinking_style="hybrid")
        units = [_make_unit("api_call", "requests.post", file="api.py")]
        records = generate_from_code(config, units)
        assert "Question:" in records[0]["thinking"], (
            f"Expected QOC from hybrid config, got: {records[0]['thinking']}"
        )

    def test_config_deterministic_style_yields_linear(self):
        """config.thinking_style='deterministic' → linear when style=None."""
        config = _make_config(thinking_style="deterministic")
        units = [_make_unit("function", "delta", file="d.py")]
        records = generate_from_code(config, units)
        assert "Entity:" in records[0]["thinking"]

    def test_explicit_style_overrides_config(self):
        """Explicit style='qoc' must override config.thinking_style='deterministic'."""
        config = _make_config(thinking_style="deterministic")
        units = [_make_unit("function", "epsilon", file="e.py")]
        records = generate_from_code(config, units, style="qoc")
        assert "Question:" in records[0]["thinking"]

    def test_all_records_qoc_styled(self):
        """When style='qoc', every record in a multi-unit call has QOC trace."""
        config = _make_config()
        units = [u.copy() for u in _ALL_UNIT_TYPES]
        records = generate_from_code(config, units, style="qoc")
        for rec, unit in zip(records, units):
            assert "Question:" in rec["thinking"], (
                f"Expected QOC for {unit['type']} but got: {rec['thinking'][:80]}"
            )


# ---------------------------------------------------------------------------
# make_record: style parameter respected
# ---------------------------------------------------------------------------

class TestMakeRecordStyle:
    def test_make_record_qoc_default(self):
        unit = _make_unit("function", "zeta", file="z.py")
        rec = _make_record(unit, style="qoc")
        assert rec["type"] == "code"
        assert "Question:" in rec["thinking"]

    def test_make_record_linear_default(self):
        unit = _make_unit("function", "eta", file="e.py")
        rec = _make_record(unit)  # no style → linear
        assert "Entity:" in rec["thinking"]

    def test_record_shape_unchanged(self):
        """Record must still have type, question, thinking, output."""
        unit = _make_unit("class", "Theta", file="t.py")
        for style in ("linear", "qoc"):
            rec = _make_record(unit, style=style)
            assert rec["type"] == "code"
            assert rec["question"]
            assert rec["thinking"]
            assert "unit" in rec["output"]
            assert "name" in rec["output"]
            assert "file" in rec["output"]


# ---------------------------------------------------------------------------
# Linear traces enriched with file:lineno
# ---------------------------------------------------------------------------

class TestLinearEnrichment:
    def test_linear_includes_lineno_function(self):
        unit = _make_unit("function", "lineno_func", file="src/x.py", lineno=42)
        trace = generate_code_thinking(unit, style="linear")
        assert "src/x.py:42" in trace, f"Expected file:lineno in trace:\n{trace}"

    def test_linear_includes_lineno_method(self):
        unit = _make_unit("method", "lineno_meth", file="src/y.py", lineno=99, **{"class": "C"})
        trace = generate_code_thinking(unit, style="linear")
        assert "src/y.py:99" in trace

    def test_linear_includes_lineno_class(self):
        unit = _make_unit("class", "LinenoCls", file="src/z.py", lineno=7)
        trace = generate_code_thinking(unit, style="linear")
        assert "src/z.py:7" in trace

    def test_linear_includes_lineno_api_call(self):
        unit = _make_unit("api_call", "requests.get", file="src/api.py", lineno=33)
        trace = generate_code_thinking(unit, style="linear")
        assert "src/api.py:33" in trace

    def test_linear_method_includes_class_name(self):
        unit = _make_unit("method", "render", file="v.py", **{"class": "View"})
        trace = generate_code_thinking(unit, style="linear")
        assert "View" in trace

    def test_qoc_includes_lineno(self):
        unit = _make_unit("function", "qoc_lineno", file="src/q.py", lineno=55)
        trace = generate_code_thinking(unit, style="qoc")
        assert "src/q.py:55" in trace


# ---------------------------------------------------------------------------
# Cross-process determinism for QOC (PYTHONHASHSEED-invariant)
# ---------------------------------------------------------------------------

_QOC_CROSS_PROCESS_SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {repo_root!r})
    from scripts.repo.generate_from_code import generate_code_thinking

    units = [
        {{"type": "function",  "name": "func_a",      "file": "pkg/a.py", "lineno": 1}},
        {{"type": "method",    "name": "meth_b",      "file": "pkg/b.py", "lineno": 2, "class": "B"}},
        {{"type": "class",     "name": "ClassC",      "file": "pkg/c.py", "lineno": 3}},
        {{"type": "api_call",  "name": "requests.get","file": "pkg/d.py", "lineno": 4}},
        # Classless method — must not emit 'None'
        {{"type": "method",    "name": "orphan",      "file": "pkg/e.py", "lineno": 5}},
    ]

    traces = [generate_code_thinking(u, style="qoc") for u in units]
    print(json.dumps(traces))
    """
)


def _run_qoc_cross_process(hashseed: int):
    """Run the QOC cross-process script with a specific PYTHONHASHSEED."""
    repo_root = str(_REPO_ROOT)
    script = _QOC_CROSS_PROCESS_SCRIPT.format(repo_root=repo_root)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"QOC cross-process subprocess failed (PYTHONHASHSEED={hashseed}):\n{result.stderr}"
    )
    return json.loads(result.stdout)


def test_qoc_cross_process_deterministic():
    """QOC traces must be identical under different PYTHONHASHSEED values."""
    traces_seed0 = _run_qoc_cross_process(0)
    traces_seed1 = _run_qoc_cross_process(1)
    assert traces_seed0 == traces_seed1, (
        "QOC traces differ between PYTHONHASHSEED=0 and PYTHONHASHSEED=1 — "
        "a non-stable hash is being used."
    )


def test_qoc_cross_process_no_none_in_classless():
    """QOC classless-method trace must not contain 'None' even across processes."""
    traces = _run_qoc_cross_process(42)
    # Last entry in the script is the classless 'orphan' method
    orphan_trace = traces[-1]
    assert "None" not in orphan_trace, (
        f"Literal 'None' found in cross-process QOC classless trace:\n{orphan_trace}"
    )


def test_qoc_cross_process_markers_present():
    """QOC cross-process traces contain the expected structural markers."""
    traces = _run_qoc_cross_process(7)
    for trace in traces:
        assert "Question:" in trace, f"Missing 'Question:' in cross-process trace:\n{trace}"
        assert "Option" in trace, f"Missing 'Option' in cross-process trace:\n{trace}"
        assert "Criteria:" in trace, f"Missing 'Criteria:' in cross-process trace:\n{trace}"


# ---------------------------------------------------------------------------
# Production-path wiring: style must reach records via _split_records,
# generate_from_repo, and the CLI — not only via generate_from_code.
# These tests FAIL against pre-fix code where _split_records() hardcoded linear.
# ---------------------------------------------------------------------------

_FIXTURE_REPO = _REPO_ROOT / "tests" / "fixtures" / "sample_repo"


def _fixture_config(thinking_style="deterministic"):
    """RepoConfig pointing at the sample_repo fixture."""
    return RepoConfig(
        name="prodpath-test",
        path=str(_FIXTURE_REPO),
        include=["**/*.py"],
        exclude=[],
        thinking_style=thinking_style,
    )


class TestStyleFromConfig:
    def test_hybrid_maps_to_qoc(self):
        assert _style_from_config(_make_config(thinking_style="hybrid")) == "qoc"

    def test_deterministic_maps_to_linear(self):
        assert _style_from_config(_make_config(thinking_style="deterministic")) == "linear"

    def test_unknown_maps_to_linear(self):
        assert _style_from_config(_make_config(thinking_style="something-else")) == "linear"


class TestSplitRecordsStyle:
    """_split_records must honor the style param (the production split path)."""

    def test_split_records_qoc(self):
        units = [_make_unit("function", "split_qoc", file="s.py")]
        train, hold = _split_records(units, 0.0, style="qoc")
        recs = train + hold
        assert recs, "expected at least one record"
        assert "Question:" in recs[0]["thinking"], (
            f"_split_records did not apply QOC style: {recs[0]['thinking']}"
        )

    def test_split_records_default_linear(self):
        """Default (no style) must remain linear — backward compatible."""
        units = [_make_unit("function", "split_lin", file="s.py")]
        train, hold = _split_records(units, 0.0)
        recs = train + hold
        assert "Entity:" in recs[0]["thinking"]


class TestGenerateFromRepoProductionPath:
    """generate_from_repo is the real disk-writing path. It must honor thinking_style."""

    def test_hybrid_config_yields_qoc_via_repo_path(self):
        """thinking_style='hybrid' → QOC traces through generate_from_repo.

        This is the regression test for the wiring blocker: pre-fix,
        _split_records hardcoded linear, so this asserted 'Question:' would be
        absent and the test would FAIL.
        """
        config = _fixture_config(thinking_style="hybrid")
        train, hold = generate_from_repo(config)
        recs = train + hold
        assert recs, "fixture produced no records"
        assert all("Question:" in r["thinking"] for r in recs), (
            "generate_from_repo did not produce QOC traces for hybrid config — "
            "style is not threaded through the production split path"
        )
        # And it must NOT be linear.
        assert not any("Entity:" in r["thinking"] for r in recs)

    def test_deterministic_config_yields_linear_via_repo_path(self):
        config = _fixture_config(thinking_style="deterministic")
        train, hold = generate_from_repo(config)
        recs = train + hold
        assert recs, "fixture produced no records"
        assert all("Entity:" in r["thinking"] for r in recs), (
            "generate_from_repo did not produce linear traces for deterministic config"
        )
        assert not any("Question:" in r["thinking"] for r in recs)


def test_cli_hybrid_writes_qoc_to_disk(tmp_path):
    """End-to-end: CLI with thinking_style=hybrid must write QOC traces to disk.

    Exercises main() → _split_records, the genuine production write path.
    Would FAIL pre-fix (main() called _split_records without style).
    """
    repo_path = tmp_path / "hybrepo"
    repo_path.mkdir()
    (repo_path / "mod.py").write_text(
        "def alpha(): pass\n"
        "class Gamma:\n"
        "    def delta(self): pass\n"
    )
    out_dir = tmp_path / "out"
    cfg_path = repo_path / "repo.yaml"
    cfg_path.write_text(
        f"name: hybrepo\n"
        f"path: {repo_path}\n"
        f"generation:\n"
        f"  thinking_style: hybrid\n"
    )

    script = _REPO_ROOT / "scripts" / "repo" / "generate_from_code.py"
    result = subprocess.run(
        [sys.executable, str(script), str(cfg_path), "--output-dir", str(out_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"

    train_path = out_dir / "hybrepo" / "training.jsonl"
    holdout_path = out_dir / "hybrepo" / "holdout.jsonl"
    all_lines = []
    for p in (train_path, holdout_path):
        if p.exists():
            all_lines += [l for l in p.read_text().splitlines() if l.strip()]
    assert all_lines, "no records written to disk"
    for line in all_lines:
        rec = json.loads(line)
        assert "Question:" in rec["thinking"], (
            f"CLI hybrid run wrote a non-QOC trace to disk: {rec['thinking']}"
        )
