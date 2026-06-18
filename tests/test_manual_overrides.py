"""Tests for Milestone 2.4 — manual thinking-trace overrides.

Covers:
- load_manual_overrides populates _MANUAL / _MANUAL_QOC from a YAML.
- Linear override is returned verbatim for matching unit + style="linear".
- QOC override is returned verbatim for matching unit + style="qoc".
- Style-specificity: no cross-fallback between linear and qoc dicts.
- No override present → generated trace unchanged (backward compat).
- Empty / missing path → clears dicts, no crash.
- Override flows through _make_record, _split_records, generate_from_code.
- generate_from_repo auto-loads from config.manual_overrides.
- generate_from_repo with no manual_overrides clears previous overrides.
- loader: load_repo_config parses manual_overrides, resolves relative path.
- Determinism: override returned identically across two calls.
- Cross-process determinism: override identical under PYTHONHASHSEED 0 vs 1.
"""

import json
import os
import subprocess
import sys
import textwrap
import tempfile
from pathlib import Path

import pytest
import yaml

from scripts.repo.generate_from_code import (
    _MANUAL,
    _MANUAL_QOC,
    _unit_key,
    clear_manual_overrides,
    load_manual_overrides,
    generate_code_thinking,
    generate_from_code,
    _make_record,
    _split_records,
    generate_from_repo,
)
from scripts.repo.loader import RepoConfig, load_repo_config

# ---------------------------------------------------------------------------
# Repo root for subprocess tests
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# autouse fixture: isolate module-level globals between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_overrides():
    """Clear override dicts before and after every test in this module."""
    clear_manual_overrides()
    yield
    clear_manual_overrides()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unit(unit_type="function", name="foo", file="pkg/x.py", **kwargs):
    u = {"type": unit_type, "name": name, "file": file, "lineno": 1}
    u.update(kwargs)
    return u


def _make_config(**kwargs):
    return RepoConfig(
        name="test-overrides",
        path="/tmp",
        include=["**/*.py"],
        exclude=[],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# load_manual_overrides: basic population
# ---------------------------------------------------------------------------


class TestLoadManualOverrides:
    def test_loads_manual_traces(self, tmp_path):
        overrides = tmp_path / "overrides.yaml"
        overrides.write_text(yaml.dump({
            "manual_traces": {"pkg/x.py:foo": "MY LINEAR TRACE"},
        }))
        load_manual_overrides(str(overrides))
        import scripts.repo.generate_from_code as mod
        assert mod._MANUAL == {"pkg/x.py:foo": "MY LINEAR TRACE"}
        assert mod._MANUAL_QOC == {}

    def test_loads_manual_traces_qoc(self, tmp_path):
        overrides = tmp_path / "overrides.yaml"
        overrides.write_text(yaml.dump({
            "manual_traces_qoc": {"pkg/x.py:foo": "MY QOC TRACE"},
        }))
        load_manual_overrides(str(overrides))
        import scripts.repo.generate_from_code as mod
        assert mod._MANUAL_QOC == {"pkg/x.py:foo": "MY QOC TRACE"}
        assert mod._MANUAL == {}

    def test_loads_both_sections(self, tmp_path):
        overrides = tmp_path / "overrides.yaml"
        overrides.write_text(yaml.dump({
            "manual_traces": {"pkg/x.py:foo": "LINEAR"},
            "manual_traces_qoc": {"pkg/x.py:foo": "QOC"},
        }))
        load_manual_overrides(str(overrides))
        import scripts.repo.generate_from_code as mod
        assert mod._MANUAL["pkg/x.py:foo"] == "LINEAR"
        assert mod._MANUAL_QOC["pkg/x.py:foo"] == "QOC"

    def test_falsy_path_clears_dicts(self, tmp_path):
        # Populate first, then clear via falsy path.
        overrides = tmp_path / "o.yaml"
        overrides.write_text(yaml.dump({"manual_traces": {"a:b": "X"}}))
        load_manual_overrides(str(overrides))
        import scripts.repo.generate_from_code as mod
        assert mod._MANUAL  # non-empty

        load_manual_overrides(None)
        assert mod._MANUAL == {}
        assert mod._MANUAL_QOC == {}

    def test_empty_string_path_clears_dicts(self):
        load_manual_overrides("")
        import scripts.repo.generate_from_code as mod
        assert mod._MANUAL == {}
        assert mod._MANUAL_QOC == {}

    def test_missing_file_clears_dicts(self, tmp_path):
        load_manual_overrides(str(tmp_path / "nonexistent.yaml"))
        import scripts.repo.generate_from_code as mod
        assert mod._MANUAL == {}
        assert mod._MANUAL_QOC == {}

    def test_replace_semantics(self, tmp_path):
        """Second call fully replaces — no accumulation."""
        f1 = tmp_path / "a.yaml"
        f1.write_text(yaml.dump({"manual_traces": {"a:b": "FIRST"}}))
        load_manual_overrides(str(f1))

        f2 = tmp_path / "b.yaml"
        f2.write_text(yaml.dump({"manual_traces": {"c:d": "SECOND"}}))
        load_manual_overrides(str(f2))

        import scripts.repo.generate_from_code as mod
        assert "a:b" not in mod._MANUAL
        assert mod._MANUAL == {"c:d": "SECOND"}

    def test_empty_yaml_no_crash(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        load_manual_overrides(str(f))
        import scripts.repo.generate_from_code as mod
        assert mod._MANUAL == {}
        assert mod._MANUAL_QOC == {}


# ---------------------------------------------------------------------------
# clear_manual_overrides
# ---------------------------------------------------------------------------


class TestClearManualOverrides:
    def test_clears_both_dicts(self, tmp_path):
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({
            "manual_traces": {"a:b": "X"},
            "manual_traces_qoc": {"a:b": "Y"},
        }))
        load_manual_overrides(str(f))
        import scripts.repo.generate_from_code as mod
        assert mod._MANUAL
        assert mod._MANUAL_QOC

        clear_manual_overrides()
        assert mod._MANUAL == {}
        assert mod._MANUAL_QOC == {}


# ---------------------------------------------------------------------------
# _unit_key
# ---------------------------------------------------------------------------


class TestUnitKey:
    def test_unit_key_format(self):
        unit = _make_unit(name="my_func", file="src/utils.py")
        assert _unit_key(unit) == "src/utils.py:my_func"

    def test_unit_key_matches_override_lookup(self):
        unit = _make_unit(name="foo", file="pkg/x.py")
        assert _unit_key(unit) == "pkg/x.py:foo"


# ---------------------------------------------------------------------------
# generate_code_thinking: linear override returned verbatim
# ---------------------------------------------------------------------------


class TestLinearOverride:
    def test_linear_override_returned(self, tmp_path):
        unit = _make_unit(name="foo", file="pkg/x.py")
        override = "MY HAND-AUTHORED LINEAR TRACE\nLine two"
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({"manual_traces": {"pkg/x.py:foo": override}}))
        load_manual_overrides(str(f))
        result = generate_code_thinking(unit, style="linear")
        assert result == override

    def test_linear_override_default_style(self, tmp_path):
        """Default style (no arg) must also return the linear override."""
        unit = _make_unit(name="bar", file="mod.py")
        override = "CUSTOM LINEAR"
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({"manual_traces": {"mod.py:bar": override}}))
        load_manual_overrides(str(f))
        result = generate_code_thinking(unit)
        assert result == override

    def test_linear_override_exact_string(self, tmp_path):
        """Override must be byte-identical, no stripping/mutating."""
        unit = _make_unit(name="process", file="a/b.py")
        override = "Entity: custom process\nScope: special\nUse: process(x)\nNOT: a placeholder"
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({"manual_traces": {"a/b.py:process": override}}))
        load_manual_overrides(str(f))
        assert generate_code_thinking(unit, style="linear") == override


# ---------------------------------------------------------------------------
# generate_code_thinking: qoc override returned verbatim
# ---------------------------------------------------------------------------


class TestQOCOverride:
    def test_qoc_override_returned(self, tmp_path):
        unit = _make_unit(name="foo", file="pkg/x.py")
        override = "Question: Is this a test?\nOption A: yes\nCriteria: yes wins."
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({"manual_traces_qoc": {"pkg/x.py:foo": override}}))
        load_manual_overrides(str(f))
        result = generate_code_thinking(unit, style="qoc")
        assert result == override


# ---------------------------------------------------------------------------
# Style-specificity: no cross-fallback
# ---------------------------------------------------------------------------


class TestStyleSpecificity:
    def test_linear_only_override_does_not_affect_qoc(self, tmp_path):
        """Unit with ONLY a linear override → qoc returns a GENERATED trace (has 'Question:')."""
        unit = _make_unit(name="foo", file="pkg/x.py")
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({"manual_traces": {"pkg/x.py:foo": "MY LINEAR ONLY"}}))
        load_manual_overrides(str(f))
        qoc_trace = generate_code_thinking(unit, style="qoc")
        # Must be generated (not the override)
        assert qoc_trace != "MY LINEAR ONLY"
        assert "Question:" in qoc_trace

    def test_qoc_only_override_does_not_affect_linear(self, tmp_path):
        """Unit with ONLY a qoc override → linear returns a GENERATED trace (has 'Entity:')."""
        unit = _make_unit(name="foo", file="pkg/x.py")
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({"manual_traces_qoc": {"pkg/x.py:foo": "MY QOC ONLY"}}))
        load_manual_overrides(str(f))
        linear_trace = generate_code_thinking(unit, style="linear")
        assert linear_trace != "MY QOC ONLY"
        assert "Entity:" in linear_trace

    def test_bogus_style_uses_linear_dict_not_qoc(self, tmp_path):
        """Unknown style falls back to _MANUAL (same as linear), NOT _MANUAL_QOC."""
        unit = _make_unit(name="foo", file="pkg/x.py")
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({
            "manual_traces": {"pkg/x.py:foo": "LINEAR OVERRIDE"},
            "manual_traces_qoc": {"pkg/x.py:foo": "QOC OVERRIDE"},
        }))
        load_manual_overrides(str(f))
        result = generate_code_thinking(unit, style="bogus")
        assert result == "LINEAR OVERRIDE"


# ---------------------------------------------------------------------------
# No override present → backward compat
# ---------------------------------------------------------------------------


class TestNoOverride:
    def test_no_override_linear_still_generates(self):
        unit = _make_unit(name="my_func", file="mod.py")
        trace = generate_code_thinking(unit, style="linear")
        assert "Entity:" in trace

    def test_no_override_qoc_still_generates(self):
        unit = _make_unit(name="my_func", file="mod.py")
        trace = generate_code_thinking(unit, style="qoc")
        assert "Question:" in trace

    def test_missing_key_falls_through_to_generation(self, tmp_path):
        """A loaded YAML with a different key must not affect an unrelated unit."""
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({"manual_traces": {"other/file.py:other_func": "OTHER"}}))
        load_manual_overrides(str(f))
        unit = _make_unit(name="unrelated", file="pkg/x.py")
        trace = generate_code_thinking(unit, style="linear")
        assert "Entity:" in trace
        assert trace != "OTHER"


# ---------------------------------------------------------------------------
# Override flows through _make_record, _split_records, generate_from_code
# ---------------------------------------------------------------------------


class TestOverridePipeline:
    def _load_override(self, tmp_path, key, value, section="manual_traces"):
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({section: {key: value}}))
        load_manual_overrides(str(f))

    def test_make_record_returns_override(self, tmp_path):
        unit = _make_unit(name="foo", file="pkg/x.py")
        override = "RECORD OVERRIDE"
        self._load_override(tmp_path, "pkg/x.py:foo", override)
        rec = _make_record(unit, style="linear")
        assert rec["thinking"] == override

    def test_make_record_qoc_override(self, tmp_path):
        unit = _make_unit(name="foo", file="pkg/x.py")
        override = "QOC RECORD OVERRIDE"
        self._load_override(tmp_path, "pkg/x.py:foo", override, section="manual_traces_qoc")
        rec = _make_record(unit, style="qoc")
        assert rec["thinking"] == override

    def test_split_records_contains_override(self, tmp_path):
        unit = _make_unit(name="foo", file="pkg/x.py")
        override = "SPLIT OVERRIDE"
        self._load_override(tmp_path, "pkg/x.py:foo", override)
        train, holdout = _split_records([unit], holdout_ratio=0.0)
        # holdout_ratio=0.0 → all go to train
        assert len(train) == 1
        assert train[0]["thinking"] == override

    def test_generate_from_code_override(self, tmp_path):
        unit = _make_unit(name="foo", file="pkg/x.py")
        override = "GEN OVERRIDE"
        self._load_override(tmp_path, "pkg/x.py:foo", override)
        config = _make_config()
        records = generate_from_code(config, [unit], style="linear")
        assert len(records) == 1
        assert records[0]["thinking"] == override

    def test_non_overridden_unit_in_batch_unaffected(self, tmp_path):
        """In a multi-unit batch, only the overridden unit has the custom trace."""
        unit_a = _make_unit(name="foo", file="pkg/x.py")
        unit_b = _make_unit(name="bar", file="pkg/y.py")
        override = "ONLY FOO OVERRIDE"
        self._load_override(tmp_path, "pkg/x.py:foo", override)
        config = _make_config()
        records = generate_from_code(config, [unit_a, unit_b], style="linear")
        assert records[0]["thinking"] == override
        assert "Entity:" in records[1]["thinking"]


# ---------------------------------------------------------------------------
# generate_from_repo: auto-loads from config.manual_overrides
# ---------------------------------------------------------------------------


class TestGenerateFromRepoAutoLoad:
    def _write_python_file(self, path: Path, func_name: str):
        path.write_text(f"def {func_name}():\n    pass\n")

    def _write_repo_yaml(self, cfg_path: Path, repo_path: Path, overrides_path=None):
        generation = {}
        if overrides_path:
            generation["manual_overrides"] = str(overrides_path)
        data = {
            "name": "test-auto-load",
            "path": str(repo_path),
            "extraction": {"units": ["functions"]},
            "generation": {"target_records": 500, "holdout_ratio": 0.15, **generation},
        }
        cfg_path.write_text(yaml.dump(data))

    def _write_overrides_yaml(self, path: Path, key: str, value: str):
        path.write_text(yaml.dump({"manual_traces": {key: value}}))

    def test_generate_from_repo_uses_override(self, tmp_path):
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        self._write_python_file(repo_dir / "helper.py", "my_helper")
        cfg_path = tmp_path / "repo.yaml"
        overrides_path = tmp_path / "overrides.yaml"
        override_value = "REPO OVERRIDE TEXT"
        # The key must match what the scanner produces: relative posix path + name
        self._write_overrides_yaml(overrides_path, "helper.py:my_helper", override_value)
        self._write_repo_yaml(cfg_path, repo_dir, overrides_path=overrides_path)
        config = load_repo_config(cfg_path)

        train, holdout = generate_from_repo(config)
        all_records = train + holdout
        matching = [r for r in all_records if r["output"]["name"] == "my_helper"]
        assert matching, "Expected a record for my_helper"
        assert matching[0]["thinking"] == override_value

    def test_generate_from_repo_clears_overrides_when_none(self, tmp_path):
        """Running with no manual_overrides must clear any previously set override."""
        # First run: set an override.
        repo_dir = tmp_path / "myrepo"
        repo_dir.mkdir()
        self._write_python_file(repo_dir / "helper.py", "my_helper")
        overrides_path = tmp_path / "overrides.yaml"
        self._write_overrides_yaml(overrides_path, "helper.py:my_helper", "SHOULD DISAPPEAR")

        cfg_with = tmp_path / "repo_with.yaml"
        self._write_repo_yaml(cfg_with, repo_dir, overrides_path=overrides_path)
        config_with = load_repo_config(cfg_with)
        generate_from_repo(config_with)  # loads override into module globals

        # Second run: no manual_overrides → should clear and return generated trace.
        cfg_without = tmp_path / "repo_without.yaml"
        self._write_repo_yaml(cfg_without, repo_dir, overrides_path=None)
        config_without = load_repo_config(cfg_without)
        train2, holdout2 = generate_from_repo(config_without)
        all2 = train2 + holdout2
        matching2 = [r for r in all2 if r["output"]["name"] == "my_helper"]
        assert matching2, "Expected a record for my_helper in second run"
        assert matching2[0]["thinking"] != "SHOULD DISAPPEAR"
        assert "Entity:" in matching2[0]["thinking"]


# ---------------------------------------------------------------------------
# loader: load_repo_config parses manual_overrides
# ---------------------------------------------------------------------------


class TestLoaderManualOverrides:
    def test_loader_parses_manual_overrides_absolute(self, tmp_path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        overrides_file = tmp_path / "overrides.yaml"
        overrides_file.write_text("")
        cfg = tmp_path / "repo.yaml"
        cfg.write_text(yaml.dump({
            "name": "my-repo",
            "path": str(repo_dir),
            "generation": {"manual_overrides": str(overrides_file)},
        }))
        config = load_repo_config(cfg)
        assert config.manual_overrides == str(overrides_file)

    def test_loader_resolves_relative_path(self, tmp_path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        # overrides.yaml sits next to repo.yaml
        cfg = tmp_path / "repo.yaml"
        cfg.write_text(yaml.dump({
            "name": "my-repo",
            "path": str(repo_dir),
            "generation": {"manual_overrides": "overrides.yaml"},
        }))
        config = load_repo_config(cfg)
        expected = str((tmp_path / "overrides.yaml").resolve())
        assert config.manual_overrides == expected

    def test_loader_absent_key_gives_none(self, tmp_path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        cfg = tmp_path / "repo.yaml"
        cfg.write_text(yaml.dump({
            "name": "my-repo",
            "path": str(repo_dir),
        }))
        config = load_repo_config(cfg)
        assert config.manual_overrides is None

    def test_loader_empty_string_gives_none(self, tmp_path):
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        cfg = tmp_path / "repo.yaml"
        cfg.write_text(yaml.dump({
            "name": "my-repo",
            "path": str(repo_dir),
            "generation": {"manual_overrides": ""},
        }))
        config = load_repo_config(cfg)
        assert config.manual_overrides is None

    def test_loader_does_not_require_file_to_exist(self, tmp_path):
        """load_repo_config must succeed even if the overrides file doesn't exist."""
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        cfg = tmp_path / "repo.yaml"
        cfg.write_text(yaml.dump({
            "name": "my-repo",
            "path": str(repo_dir),
            "generation": {"manual_overrides": "does_not_exist.yaml"},
        }))
        config = load_repo_config(cfg)  # must not raise
        assert config.manual_overrides is not None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_process_deterministic(self, tmp_path):
        unit = _make_unit(name="foo", file="pkg/x.py")
        override = "DETERMINISTIC OVERRIDE"
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({"manual_traces": {"pkg/x.py:foo": override}}))
        load_manual_overrides(str(f))
        r1 = generate_code_thinking(unit, style="linear")
        r2 = generate_code_thinking(unit, style="linear")
        assert r1 == r2 == override

    def test_qoc_same_process_deterministic(self, tmp_path):
        unit = _make_unit(name="foo", file="pkg/x.py")
        override = "QOC DETERMINISTIC OVERRIDE"
        f = tmp_path / "o.yaml"
        f.write_text(yaml.dump({"manual_traces_qoc": {"pkg/x.py:foo": override}}))
        load_manual_overrides(str(f))
        r1 = generate_code_thinking(unit, style="qoc")
        r2 = generate_code_thinking(unit, style="qoc")
        assert r1 == r2 == override


# ---------------------------------------------------------------------------
# Cross-process determinism (subprocess, PYTHONHASHSEED 0 vs 1)
# ---------------------------------------------------------------------------

_OVERRIDE_CROSS_PROCESS_SCRIPT = textwrap.dedent(
    """
    import sys, json, tempfile, pathlib, os
    sys.path.insert(0, {repo_root!r})
    import yaml

    overrides_path = {overrides_path!r}
    from scripts.repo.generate_from_code import load_manual_overrides, generate_code_thinking

    load_manual_overrides(overrides_path)

    units = [
        {{"type": "function", "name": "target_func", "file": "pkg/target.py", "lineno": 1}},
        {{"type": "class",    "name": "TargetClass", "file": "pkg/target.py", "lineno": 10}},
    ]
    results = []
    for style in ("linear", "qoc"):
        for unit in units:
            results.append(generate_code_thinking(unit, style=style))
    print(json.dumps(results))
    """
)


def _run_override_cross_process(hashseed: int, overrides_path: str):
    repo_root = str(_REPO_ROOT)
    script = _OVERRIDE_CROSS_PROCESS_SCRIPT.format(
        repo_root=repo_root,
        overrides_path=overrides_path,
    )
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"Override cross-process subprocess failed (PYTHONHASHSEED={hashseed}):\n{result.stderr}"
    )
    return json.loads(result.stdout)


def test_override_cross_process_deterministic(tmp_path):
    """Override traces must be identical under different PYTHONHASHSEED values."""
    overrides_path = tmp_path / "overrides.yaml"
    overrides_path.write_text(yaml.dump({
        "manual_traces": {"pkg/target.py:target_func": "CROSS PROC LINEAR"},
        "manual_traces_qoc": {"pkg/target.py:TargetClass": "CROSS PROC QOC"},
    }))

    results_seed0 = _run_override_cross_process(0, str(overrides_path))
    results_seed1 = _run_override_cross_process(1, str(overrides_path))
    assert results_seed0 == results_seed1, (
        "Override traces differ between PYTHONHASHSEED=0 and PYTHONHASHSEED=1"
    )


def test_override_cross_process_contains_override(tmp_path):
    """The override strings are present in cross-process output."""
    overrides_path = tmp_path / "overrides.yaml"
    linear_override = "CROSS PROC LINEAR OVERRIDE"
    qoc_override = "CROSS PROC QOC OVERRIDE"
    overrides_path.write_text(yaml.dump({
        "manual_traces": {"pkg/target.py:target_func": linear_override},
        "manual_traces_qoc": {"pkg/target.py:TargetClass": qoc_override},
    }))
    results = _run_override_cross_process(42, str(overrides_path))
    assert linear_override in results, "Linear override not found in cross-process results"
    assert qoc_override in results, "QOC override not found in cross-process results"
