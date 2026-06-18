"""Tests for scripts/repo_pipeline.py (Milestone 1.5).

Hermetic — all repo scanning uses the sample_repo fixture or in-memory
temp directories. No network access, no LLM calls, no writes to the real
data/ directory.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Repo root on sys.path so that scripts.* imports work in-process.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.repo_pipeline import run_pipeline

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def _make_temp_repo(tmp_root: Path, name: str = "myrepo") -> tuple[Path, Path]:
    """Create a minimal temp repo dir and repo.yaml; return (repo_dir, yaml_path)."""
    repo_dir = tmp_root / "src" / name
    repo_dir.mkdir(parents=True)

    # Write a small Python module with a handful of code units.
    (repo_dir / "mod.py").write_text(
        "def alpha(): pass\n"
        "def beta(x, y): return x + y\n"
        "class Gamma:\n"
        "    def delta(self): pass\n"
        "    def epsilon(self, val): return val\n"
    )

    yaml_path = tmp_root / "myconfig"
    yaml_path.mkdir(exist_ok=True)
    cfg = yaml_path / "repo.yaml"
    cfg.write_text(
        f"name: {name}\npath: {repo_dir}\n"
        "extraction:\n"
        "  units:\n"
        "    - functions\n"
        "    - classes\n"
        "    - methods\n"
        "    - api_calls\n"
    )
    return yaml_path, cfg


# ---------------------------------------------------------------------------
# e2e: both output files are created under data_dir/repos/<name>/
# ---------------------------------------------------------------------------

def test_e2e_writes_both_output_files():
    """Running the pipeline on a small repo writes training.jsonl and holdout.jsonl."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_dir, _ = _make_temp_repo(tmp_path)
        data_root = tmp_path / "data" / "repos"

        result = run_pipeline(repo_dir=repo_dir, data_dir=data_root)

        train_path = data_root / "myrepo" / "training.jsonl"
        holdout_path = data_root / "myrepo" / "holdout.jsonl"

        assert train_path.exists(), "training.jsonl not written"
        assert holdout_path.exists(), "holdout.jsonl not written"
        assert result["name"] == "myrepo"
        assert not result["skipped"]
        assert not result["dry_run"]


# ---------------------------------------------------------------------------
# Every emitted line is valid JSON with type == "code"
# ---------------------------------------------------------------------------

def test_output_lines_are_valid_json_with_type_code():
    """Every line in both output files is valid JSON with type='code'."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_dir, _ = _make_temp_repo(tmp_path)
        data_root = tmp_path / "out"

        run_pipeline(repo_dir=repo_dir, data_dir=data_root)

        for fname in ("training.jsonl", "holdout.jsonl"):
            fpath = data_root / "myrepo" / fname
            lines = [l.strip() for l in fpath.read_text().splitlines() if l.strip()]
            assert lines, f"{fname} is empty"
            for line in lines:
                obj = json.loads(line)  # must not raise
                assert isinstance(obj, dict), f"Expected dict, got {type(obj)}"
                assert obj.get("type") == "code", f"type != 'code' in {fname}: {obj!r}"


# ---------------------------------------------------------------------------
# train + holdout counts sum to scanned-unit count and keys are disjoint
# ---------------------------------------------------------------------------

def test_partition_complete_and_disjoint():
    """train + holdout covers all scanned units exactly once, with disjoint keys."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_dir, _ = _make_temp_repo(tmp_path)
        data_root = tmp_path / "out"

        result = run_pipeline(repo_dir=repo_dir, data_dir=data_root)

        train_path = data_root / "myrepo" / "training.jsonl"
        holdout_path = data_root / "myrepo" / "holdout.jsonl"

        def _read_keys(p: Path):
            keys = []
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                keys.append((obj["output"]["name"], obj["output"]["file"]))
            return keys

        train_keys = _read_keys(train_path)
        hold_keys = _read_keys(holdout_path)

        # Validate completeness independently against the SCANNER, not against
        # run_pipeline's own reported counts (which share the same source call).
        from scripts.repo.loader import load_repo_config
        from scripts.repo.scan_repo import scan_repo

        units = scan_repo(load_repo_config(repo_dir / "repo.yaml"))
        unit_keys = {(u["name"], u["file"]) for u in units}

        # Precondition for the set-equality below: this fixture has no two units
        # sharing the same (name, file) key. (name, file) is not globally unique
        # for the scanner — overloads/redefs could collide — so document that the
        # fixture avoids it; line below is the collision-proof completeness check.
        assert len(unit_keys) == len(units), "fixture has colliding (name, file) keys"

        # Complete: output files cover every scanned unit exactly once.
        assert len(train_keys) + len(hold_keys) == len(units)
        assert set(train_keys) | set(hold_keys) == unit_keys

        # Disjoint (name, file) pairs
        overlap = set(train_keys) & set(hold_keys)
        assert not overlap, f"Overlapping records in train/holdout: {overlap}"

        # Sanity: reported counts match file line counts.
        assert len(train_keys) == result["train_count"]
        assert len(hold_keys) == result["holdout_count"]


# ---------------------------------------------------------------------------
# Determinism: two runs produce byte-identical files
# ---------------------------------------------------------------------------

def test_determinism_byte_identical():
    """Two pipeline runs with fresh output dirs produce byte-identical files."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_dir, _ = _make_temp_repo(tmp_path)

        data_root_1 = tmp_path / "run1"
        data_root_2 = tmp_path / "run2"

        run_pipeline(repo_dir=repo_dir, data_dir=data_root_1)
        run_pipeline(repo_dir=repo_dir, data_dir=data_root_2)

        for fname in ("training.jsonl", "holdout.jsonl"):
            content1 = (data_root_1 / "myrepo" / fname).read_bytes()
            content2 = (data_root_2 / "myrepo" / fname).read_bytes()
            assert content1 == content2, f"{fname} differs between runs"


# ---------------------------------------------------------------------------
# dry_run=True writes no files
# ---------------------------------------------------------------------------

def test_dry_run_writes_no_files():
    """dry_run=True must not create any output files."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_dir, _ = _make_temp_repo(tmp_path)
        data_root = tmp_path / "out"

        result = run_pipeline(repo_dir=repo_dir, data_dir=data_root, dry_run=True)

        assert not (data_root / "myrepo" / "training.jsonl").exists(), (
            "training.jsonl must NOT be written in dry-run mode"
        )
        assert not (data_root / "myrepo" / "holdout.jsonl").exists(), (
            "holdout.jsonl must NOT be written in dry-run mode"
        )
        assert result["dry_run"] is True
        # Counts are still computed (scanning has no side effects)
        assert result["train_count"] + result["holdout_count"] > 0


# ---------------------------------------------------------------------------
# Skip-if-done: second run with non-empty outputs is skipped
# ---------------------------------------------------------------------------

def test_skip_if_done_preserves_sentinel_content():
    """A second run when outputs already exist is skipped (no regeneration)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_dir, _ = _make_temp_repo(tmp_path)
        data_root = tmp_path / "out"

        # First real run to create the output files.
        run_pipeline(repo_dir=repo_dir, data_dir=data_root)

        train_path = data_root / "myrepo" / "training.jsonl"
        holdout_path = data_root / "myrepo" / "holdout.jsonl"

        # Overwrite with sentinel content so we can detect if re-run touches it.
        sentinel_train = b'{"type": "code", "sentinel": true}\n'
        sentinel_hold = b'{"type": "code", "sentinel": true}\n'
        train_path.write_bytes(sentinel_train)
        holdout_path.write_bytes(sentinel_hold)

        # Second run — must be skipped.
        result = run_pipeline(repo_dir=repo_dir, data_dir=data_root)

        assert result["skipped"] is True, "Expected pipeline to be skipped"
        # Sentinel content must be unchanged.
        assert train_path.read_bytes() == sentinel_train, (
            "training.jsonl was overwritten despite skip-if-done"
        )
        assert holdout_path.read_bytes() == sentinel_hold, (
            "holdout.jsonl was overwritten despite skip-if-done"
        )


def test_skip_requires_both_files_nonempty():
    """Skip-if-done only fires when BOTH files are non-empty."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        repo_dir, _ = _make_temp_repo(tmp_path)
        data_root = tmp_path / "out"

        train_path = data_root / "myrepo" / "training.jsonl"
        holdout_path = data_root / "myrepo" / "holdout.jsonl"

        # Create only training.jsonl (non-empty); holdout.jsonl missing.
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_path.write_text('{"type": "code", "sentinel": true}\n')

        result = run_pipeline(repo_dir=repo_dir, data_dir=data_root)
        # Both files should exist and not be skipped
        assert not result["skipped"], (
            "Pipeline should run when holdout.jsonl is missing"
        )
        assert holdout_path.exists()


# ---------------------------------------------------------------------------
# Missing repo.yaml exits with a clear error
# ---------------------------------------------------------------------------

def test_missing_repo_yaml_exits_nonzero():
    """A --repo-dir without repo.yaml must cause SystemExit with non-zero."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        no_yaml_dir = tmp_path / "no_yaml"
        no_yaml_dir.mkdir()

        with pytest.raises(SystemExit) as exc_info:
            run_pipeline(repo_dir=no_yaml_dir, data_dir=tmp_path / "out")

        assert exc_info.value.code != 0, "Expected non-zero exit code"


def test_missing_repo_yaml_error_message():
    """The error message must mention the missing repo.yaml clearly."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        no_yaml_dir = tmp_path / "empty_dir"
        no_yaml_dir.mkdir()

        # sys.exit(message) stores the message in SystemExit.code (as a string).
        with pytest.raises(SystemExit) as exc_info:
            run_pipeline(repo_dir=no_yaml_dir, data_dir=tmp_path / "out")

        # The exit message string should reference repo.yaml.
        exit_msg = str(exc_info.value.code).lower()
        assert "repo.yaml" in exit_msg, (
            f"Error message doesn't mention repo.yaml: {exc_info.value.code!r}"
        )


# ---------------------------------------------------------------------------
# Path-traversal repo name is rejected and cannot escape the data root
# ---------------------------------------------------------------------------

def test_rejects_path_traversal_name():
    """A repo name with '..' must be rejected and must not escape the data root."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        repo_path = tmp_path / "evilrepo"
        repo_path.mkdir()
        (repo_path / "mod.py").write_text("def x(): pass\n")

        cfg_dir = tmp_path / "cfg"
        cfg_dir.mkdir()
        # name contains a traversal sequence; path is valid.
        (cfg_dir / "repo.yaml").write_text(
            f"name: ../../escaped\npath: {repo_path}\n"
        )

        data_root = tmp_path / "out"
        data_root.mkdir()

        with pytest.raises(SystemExit) as exc_info:
            run_pipeline(repo_dir=cfg_dir, data_dir=data_root)

        # Non-zero / "unsafe"-containing exit message.
        code = exc_info.value.code
        assert code != 0, "Traversal name should be rejected with non-zero exit"
        assert "unsafe" in str(code).lower(), (
            f"Expected 'unsafe' in error message, got: {code!r}"
        )

        # Nothing escaped outside the data root.
        escaped = tmp_path / "escaped"
        assert not escaped.exists(), "Output escaped the intended data root"


# ---------------------------------------------------------------------------
# Uses the sample_repo fixture (also tests the real fixture works end-to-end)
# ---------------------------------------------------------------------------

def test_e2e_with_sample_repo_fixture():
    """The sample_repo fixture can be processed end-to-end by the pipeline."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Build repo.yaml pointing at the existing sample_repo fixture.
        cfg_dir = tmp_path / "sample_cfg"
        cfg_dir.mkdir()
        (cfg_dir / "repo.yaml").write_text(
            f"name: sample-repo\npath: {SAMPLE_REPO}\n"
        )
        data_root = tmp_path / "out"

        result = run_pipeline(repo_dir=cfg_dir, data_dir=data_root)

        assert result["name"] == "sample-repo"
        assert result["train_count"] + result["holdout_count"] > 0
        assert not result["skipped"]

        train_path = data_root / "sample-repo" / "training.jsonl"
        holdout_path = data_root / "sample-repo" / "holdout.jsonl"
        assert train_path.exists()
        assert holdout_path.exists()


# ---------------------------------------------------------------------------
# Fix #2: extraction.units filter is honored
# ---------------------------------------------------------------------------

def test_extraction_units_filter_honored():
    """Pipeline must only emit records whose output.unit matches extraction.units."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        repo_dir = tmp_path / "filterepo"
        repo_dir.mkdir()
        # Write a module with functions, methods, and a class so all types exist.
        (repo_dir / "mod.py").write_text(
            "def alpha(): pass\n"
            "def beta(): pass\n"
            "class Gamma:\n"
            "    def delta(self): pass\n"
        )

        cfg_dir = tmp_path / "filtercfg"
        cfg_dir.mkdir()
        (cfg_dir / "repo.yaml").write_text(
            f"name: filterepo\npath: {repo_dir}\n"
            "extraction:\n"
            "  units:\n"
            "    - functions\n"
        )
        data_root = tmp_path / "out"

        run_pipeline(repo_dir=cfg_dir, data_dir=data_root)

        train_path = data_root / "filterepo" / "training.jsonl"
        holdout_path = data_root / "filterepo" / "holdout.jsonl"

        all_unit_types = set()
        for fpath in (train_path, holdout_path):
            for line in fpath.read_text().splitlines():
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    all_unit_types.add(obj["output"]["unit"])

        assert all_unit_types == {"function"}, (
            f"Expected only 'function' units, got: {all_unit_types}"
        )


# ---------------------------------------------------------------------------
# Fix #3: generation.target_records caps output
# ---------------------------------------------------------------------------

def test_target_records_caps_output():
    """Pipeline must cap total output to target_records when units exceed the limit."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        repo_dir = tmp_path / "caprepo"
        repo_dir.mkdir()
        # Write a module with 12 functions so there are more than target_records=5.
        funcs = "\n".join(f"def func_{i}(): pass" for i in range(12))
        (repo_dir / "mod.py").write_text(funcs + "\n")

        cfg_dir = tmp_path / "capcfg"
        cfg_dir.mkdir()
        (cfg_dir / "repo.yaml").write_text(
            f"name: caprepo\npath: {repo_dir}\n"
            "extraction:\n"
            "  units:\n"
            "    - functions\n"
            "generation:\n"
            "  target_records: 5\n"
        )
        data_root_1 = tmp_path / "run1"
        data_root_2 = tmp_path / "run2"

        run_pipeline(repo_dir=cfg_dir, data_dir=data_root_1)
        run_pipeline(repo_dir=cfg_dir, data_dir=data_root_2)

        def _read_keys(data_root):
            keys = []
            for fname in ("training.jsonl", "holdout.jsonl"):
                fpath = data_root / "caprepo" / fname
                for line in fpath.read_text().splitlines():
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        keys.append((obj["output"]["name"], obj["output"]["file"]))
            return keys

        keys1 = _read_keys(data_root_1)
        keys2 = _read_keys(data_root_2)

        assert len(keys1) == 5, f"Expected 5 records, got {len(keys1)}"
        # Determinism: both runs select the exact same 5 units.
        assert sorted(keys1) == sorted(keys2), "Capped selection is not deterministic"


# ---------------------------------------------------------------------------
# Milestone 3.2: syntax validation helpers and pipeline integration
# ---------------------------------------------------------------------------

from scripts.repo.generate_from_code import (
    _signature_well_formed,
    _unit_syntax_ok,
    _filter_invalid_syntax,
    generate_from_repo,
)
from scripts.repo.loader import RepoConfig


def _make_config(**kwargs) -> RepoConfig:
    """Build a minimal RepoConfig for unit testing (no path validation)."""
    defaults = dict(
        name="test",
        path=None,
        url="https://example.com/repo.git",  # skip local-path check
        target_records=500,
        holdout_ratio=0.15,
    )
    defaults.update(kwargs)
    return RepoConfig(**defaults)


# --- _signature_well_formed ---

class TestSignatureWellFormed:
    """Tests for the _signature_well_formed helper."""

    def test_plain_empty_params(self):
        assert _signature_well_formed("f()") is True

    def test_simple_params(self):
        assert _signature_well_formed("f(a, b)") is True

    def test_with_default(self):
        assert _signature_well_formed("f(a, b=1)") is True

    def test_type_annotation(self):
        assert _signature_well_formed("f(obj: Any)") is True

    def test_keyword_only(self):
        assert _signature_well_formed("f(*, k: str)") is True

    def test_positional_only(self):
        assert _signature_well_formed("f(a, /, b)") is True

    def test_class_form(self):
        # Class units emit ClassName(params) — same wrap works
        assert _signature_well_formed("Foo(x: int)") is True

    def test_complex_mixed(self):
        assert _signature_well_formed("f(a, b: int, *, c: str = 'x')") is True

    def test_malformed_double_arg(self):
        assert _signature_well_formed("f(1 2)") is False

    def test_malformed_keyword_in_params(self):
        assert _signature_well_formed("f(def)") is False

    def test_malformed_double_comma(self):
        assert _signature_well_formed("f(a, , b)") is False

    def test_malformed_unclosed_paren(self):
        assert _signature_well_formed("f(") is False

    def test_none_returns_true(self):
        assert _signature_well_formed(None) is True

    def test_empty_string_returns_true(self):
        assert _signature_well_formed("") is True

    def test_non_string_int_returns_true(self):
        assert _signature_well_formed(42) is True

    def test_non_string_list_returns_true(self):
        assert _signature_well_formed([]) is True


# --- _unit_syntax_ok ---

class TestUnitSyntaxOk:
    """Tests for the _unit_syntax_ok helper."""

    def test_unit_with_valid_signature(self):
        unit = {"type": "function", "name": "f", "file": "a.py", "signature": "f(x: int)"}
        assert _unit_syntax_ok(unit) is True

    def test_unit_with_malformed_signature_is_rejected(self):
        unit = {"type": "function", "name": "f", "file": "a.py", "signature": "f(1 2)"}
        assert _unit_syntax_ok(unit) is False

    def test_unit_with_malformed_call_signature_is_rejected(self):
        unit = {
            "type": "method", "name": "m", "file": "a.py",
            "signature": "m(self, x: int)",
            "call_signature": "m(1 2)",
        }
        assert _unit_syntax_ok(unit) is False

    def test_unit_with_no_signature_passes(self):
        unit = {"type": "function", "name": "f", "file": "a.py"}
        assert _unit_syntax_ok(unit) is True

    def test_unit_with_valid_both_signatures(self):
        unit = {
            "type": "method", "name": "m", "file": "a.py",
            "signature": "m(self, x: int)",
            "call_signature": "m(x: int)",
        }
        assert _unit_syntax_ok(unit) is True


# --- _filter_invalid_syntax ---

class TestFilterInvalidSyntax:
    """Tests for the _filter_invalid_syntax helper."""

    def test_drops_unit_with_malformed_signature(self):
        good = {"type": "function", "name": "good", "file": "a.py", "signature": "good(x)"}
        bad = {"type": "function", "name": "bad", "file": "a.py", "signature": "bad(1 2)"}
        result = _filter_invalid_syntax([good, bad])
        assert result == [good]

    def test_drops_unit_with_malformed_call_signature(self):
        good = {"type": "method", "name": "m", "file": "a.py", "signature": "m(self)"}
        bad = {
            "type": "method", "name": "bad", "file": "a.py",
            "signature": "bad(self)",
            "call_signature": "bad(def)",
        }
        result = _filter_invalid_syntax([good, bad])
        assert result == [good]

    def test_keeps_units_with_no_signature(self):
        unit = {"type": "function", "name": "f", "file": "a.py"}
        result = _filter_invalid_syntax([unit])
        assert result == [unit]

    def test_preserves_order(self):
        units = [
            {"type": "function", "name": f"f{i}", "file": "a.py", "signature": f"f{i}(x)"}
            for i in range(5)
        ]
        result = _filter_invalid_syntax(units)
        assert result == units

    def test_empty_input(self):
        assert _filter_invalid_syntax([]) == []

    def test_all_malformed_returns_empty(self):
        units = [
            {"type": "function", "name": "f", "file": "a.py", "signature": "f(1 2)"},
            {"type": "function", "name": "g", "file": "a.py", "signature": "g(def)"},
        ]
        assert _filter_invalid_syntax(units) == []


# --- Pipeline opt-in (validate_syntax flag) ---

class TestPipelineValidateSyntaxFlag:
    """Tests for the validate_syntax flag wired into generate_from_repo."""

    def test_malformed_unit_excluded_when_flag_is_true(self):
        """With validate_syntax=True, a unit with a malformed signature is excluded."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_dir = tmp_path / "repo"
            repo_dir.mkdir()
            (repo_dir / "mod.py").write_text("def good(x): pass\n")

            cfg_dir = tmp_path / "cfg"
            cfg_dir.mkdir()
            (cfg_dir / "repo.yaml").write_text(
                f"name: testrepo\npath: {repo_dir}\n"
                "generation:\n"
                "  validate_syntax: true\n"
            )

            from scripts.repo.loader import load_repo_config
            config = load_repo_config(cfg_dir / "repo.yaml")
            assert config.validate_syntax is True

            # Inject a malformed unit by scanning then calling helpers directly
            from scripts.repo.scan_repo import scan_repo
            from scripts.repo.generate_from_code import _filter_units_by_config, _filter_invalid_syntax

            units = scan_repo(config)
            units_filtered = _filter_units_by_config(units, config)
            # All real units should pass (no malformed sigs from scanner)
            valid_units = _filter_invalid_syntax(units_filtered)
            assert len(valid_units) == len(units_filtered), (
                "Real scanner units should all pass syntax validation"
            )

            # Now test with a hand-injected malformed unit
            malformed = {
                "type": "function", "name": "bad", "file": "mod.py",
                "signature": "bad(1 2)",
            }
            mixed = units_filtered + [malformed]
            result = _filter_invalid_syntax(mixed)
            assert malformed not in result, "Malformed unit must be excluded"
            assert len(result) == len(units_filtered), "Good units must be preserved"

    def test_flag_false_does_not_filter_end_to_end(self, monkeypatch):
        """With validate_syntax=False, generate_from_repo keeps a malformed unit.

        This exercises the backward-compat guard in generate_from_repo END TO END
        by monkeypatching scan_repo to return a hand-built malformed unit. If the
        `if getattr(config, "validate_syntax", False):` guard were removed (making
        the pipeline always filter), this test would FAIL because the malformed
        unit would be dropped.
        """
        good = {"type": "function", "name": "good_fn", "file": "mod.py", "signature": "good_fn(x)"}
        bad = {"type": "function", "name": "bad_fn", "file": "mod.py", "signature": "bad_fn(1 2)"}

        # generate_from_repo does `from scripts.repo.scan_repo import scan_repo`
        # as a LOCAL import, so patch the source-module attribute.
        monkeypatch.setattr(
            "scripts.repo.scan_repo.scan_repo", lambda config: [good, bad]
        )

        config = _make_config(validate_syntax=False, extraction_units=["functions"])
        train, holdout = generate_from_repo(config)

        names = {r["output"]["name"] for r in (train + holdout)}
        assert "good_fn" in names, "Clean unit must be present"
        assert "bad_fn" in names, (
            "Malformed unit must SURVIVE when validate_syntax=False (guard off)"
        )

    def test_flag_true_filters_end_to_end(self, monkeypatch):
        """With validate_syntax=True, generate_from_repo drops a malformed unit.

        Exercises the guard's enabled branch end to end via generate_from_repo.
        """
        good = {"type": "function", "name": "good_fn", "file": "mod.py", "signature": "good_fn(x)"}
        bad = {"type": "function", "name": "bad_fn", "file": "mod.py", "signature": "bad_fn(1 2)"}

        monkeypatch.setattr(
            "scripts.repo.scan_repo.scan_repo", lambda config: [good, bad]
        )

        config = _make_config(validate_syntax=True, extraction_units=["functions"])
        train, holdout = generate_from_repo(config)

        names = {r["output"]["name"] for r in (train + holdout)}
        assert "good_fn" in names, "Clean unit must be present"
        assert "bad_fn" not in names, (
            "Malformed unit must be DROPPED when validate_syntax=True (guard on)"
        )

    def test_default_config_has_validate_syntax_false(self):
        """Default RepoConfig has validate_syntax=False."""
        config = _make_config()
        assert config.validate_syntax is False


# --- Determinism ---

class TestSyntaxValidationDeterminism:
    """Syntax filter produces identical output across two calls."""

    def test_same_input_same_output(self):
        units = [
            {"type": "function", "name": f"f{i}", "file": "a.py", "signature": f"f{i}(x: int)"}
            for i in range(10)
        ]
        # Insert a malformed one in the middle
        units.insert(5, {"type": "function", "name": "bad", "file": "a.py", "signature": "bad(1 2)"})
        result1 = _filter_invalid_syntax(units)
        result2 = _filter_invalid_syntax(units)
        assert result1 == result2


# --- Loader round-trip ---

class TestLoaderValidateSyntaxRoundtrip:
    """validate_syntax round-trips through load_repo_config."""

    def test_validate_syntax_true_from_yaml(self):
        """repo.yaml with validate_syntax: true loads to config.validate_syntax is True."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_dir = tmp_path / "repo"
            repo_dir.mkdir()
            cfg_path = tmp_path / "repo.yaml"
            cfg_path.write_text(
                f"name: vs-test\npath: {repo_dir}\n"
                "generation:\n"
                "  validate_syntax: true\n"
            )
            from scripts.repo.loader import load_repo_config
            config = load_repo_config(cfg_path)
            assert config.validate_syntax is True

    def test_validate_syntax_omitted_defaults_to_false(self):
        """When validate_syntax is absent from YAML, config.validate_syntax is False."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_dir = tmp_path / "repo"
            repo_dir.mkdir()
            cfg_path = tmp_path / "repo.yaml"
            cfg_path.write_text(f"name: vs-default\npath: {repo_dir}\n")
            from scripts.repo.loader import load_repo_config
            config = load_repo_config(cfg_path)
            assert config.validate_syntax is False

    def test_validate_syntax_false_from_yaml(self):
        """repo.yaml with validate_syntax: false loads to config.validate_syntax is False."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_dir = tmp_path / "repo"
            repo_dir.mkdir()
            cfg_path = tmp_path / "repo.yaml"
            cfg_path.write_text(
                f"name: vs-false\npath: {repo_dir}\n"
                "generation:\n"
                "  validate_syntax: false\n"
            )
            from scripts.repo.loader import load_repo_config
            config = load_repo_config(cfg_path)
            assert config.validate_syntax is False


# --- Real-data no-op guard ---

class TestRealDataNoopGuard:
    """Annotated/kw-only/posonly signatures from a real scanner must NOT be dropped."""

    def test_annotated_signatures_not_dropped(self):
        """Units with annotation-style signatures all pass the syntax filter."""
        units = [
            {"type": "function", "name": "f1", "file": "a.py", "signature": "f1(obj: Any)"},
            {"type": "function", "name": "f2", "file": "a.py", "signature": "f2(*, k: str)"},
            {"type": "function", "name": "f3", "file": "a.py", "signature": "f3(a, /, b)"},
            {"type": "function", "name": "f4", "file": "a.py", "signature": "f4(a, b=1)"},
            {"type": "class", "name": "Foo", "file": "a.py", "signature": "Foo(x: int)"},
            {
                "type": "method", "name": "m", "file": "a.py",
                "signature": "m(self, x: int, *, flag: bool = False)",
                "call_signature": "m(x: int, *, flag: bool = False)",
            },
        ]
        result = _filter_invalid_syntax(units)
        assert result == units, (
            f"Some valid annotated units were wrongly dropped: "
            f"{[u['name'] for u in units if u not in result]}"
        )


# ---------------------------------------------------------------------------
# Milestone 3.3: trivial-unit rejection helpers and pipeline integration
# ---------------------------------------------------------------------------

from scripts.repo.generate_from_code import (
    _is_dunder_name,
    _is_trivial_unit,
    _filter_trivial_units,
)


class TestIsDunderName:
    """Unit tests for _is_dunder_name."""

    def test_dunder_enter_is_dunder(self):
        assert _is_dunder_name("__enter__") is True

    def test_dunder_exit_is_dunder(self):
        assert _is_dunder_name("__exit__") is True

    def test_dunder_repr_is_dunder(self):
        assert _is_dunder_name("__repr__") is True

    def test_dunder_len_is_dunder(self):
        assert _is_dunder_name("__len__") is True

    def test_dunder_call_is_dunder(self):
        assert _is_dunder_name("__call__") is True

    def test_dunder_init_is_dunder(self):
        # __init__ IS a dunder name — it's just in the keep-set.
        assert _is_dunder_name("__init__") is True

    def test_normal_name_not_dunder(self):
        assert _is_dunder_name("compute") is False

    def test_single_underscore_not_dunder(self):
        assert _is_dunder_name("_private") is False

    def test_non_string_not_dunder(self):
        assert _is_dunder_name(None) is False
        assert _is_dunder_name(42) is False

    def test_too_short_not_dunder(self):
        # "__" (len==2) and "____" (len==4) must not be dunders.
        assert _is_dunder_name("__") is False
        assert _is_dunder_name("____") is False


class TestIsTrivialUnit:
    """Unit tests for _is_trivial_unit."""

    def test_dunder_enter_method_is_trivial(self):
        unit = {"type": "method", "name": "__enter__", "file": "a.py", "class": "Ctx"}
        assert _is_trivial_unit(unit) is True

    def test_dunder_enter_function_is_trivial(self):
        # Top-level __enter__ is unusual but rules are the same.
        unit = {"type": "function", "name": "__enter__", "file": "a.py"}
        assert _is_trivial_unit(unit) is True

    def test_dunder_init_is_NOT_trivial(self):
        unit = {"type": "method", "name": "__init__", "file": "a.py", "class": "Foo"}
        assert _is_trivial_unit(unit) is False

    def test_stub_function_is_trivial(self):
        unit = {"type": "function", "name": "stub_fn", "file": "a.py", "is_stub": True}
        assert _is_trivial_unit(unit) is True

    def test_stub_method_is_trivial(self):
        unit = {"type": "method", "name": "stub_m", "file": "a.py", "class": "X", "is_stub": True}
        assert _is_trivial_unit(unit) is True

    def test_normal_function_not_trivial(self):
        unit = {"type": "function", "name": "compute", "file": "a.py"}
        assert _is_trivial_unit(unit) is False

    def test_normal_method_not_trivial(self):
        unit = {"type": "method", "name": "do_work", "file": "a.py", "class": "Worker"}
        assert _is_trivial_unit(unit) is False

    def test_class_unit_never_trivial_even_if_dunder_named(self):
        unit = {"type": "class", "name": "__enter__", "file": "a.py"}
        assert _is_trivial_unit(unit) is False

    def test_api_call_never_trivial(self):
        unit = {"type": "api_call", "name": "requests.get", "file": "a.py"}
        assert _is_trivial_unit(unit) is False


class TestFilterTrivialUnits:
    """Unit tests for _filter_trivial_units."""

    def test_drops_dunder_enter(self):
        good = {"type": "method", "name": "do_work", "file": "a.py", "class": "X"}
        bad = {"type": "method", "name": "__enter__", "file": "a.py", "class": "X"}
        result = _filter_trivial_units([good, bad])
        assert result == [good]

    def test_keeps_dunder_init(self):
        init = {"type": "method", "name": "__init__", "file": "a.py", "class": "Foo"}
        other = {"type": "method", "name": "__exit__", "file": "a.py", "class": "Foo"}
        result = _filter_trivial_units([init, other])
        assert result == [init]

    def test_drops_stub_unit(self):
        stub = {"type": "function", "name": "stub_fn", "file": "a.py", "is_stub": True}
        real = {"type": "function", "name": "real_fn", "file": "a.py"}
        result = _filter_trivial_units([stub, real])
        assert result == [real]

    def test_keeps_class_and_api_call(self):
        cls = {"type": "class", "name": "Foo", "file": "a.py"}
        api = {"type": "api_call", "name": "requests.get", "file": "a.py"}
        result = _filter_trivial_units([cls, api])
        assert result == [cls, api]

    def test_preserves_order(self):
        units = [
            {"type": "function", "name": f"fn_{i}", "file": "a.py"}
            for i in range(5)
        ]
        result = _filter_trivial_units(units)
        assert result == units

    def test_empty_input(self):
        assert _filter_trivial_units([]) == []

    def test_all_trivial_returns_empty(self):
        units = [
            {"type": "method", "name": "__enter__", "file": "a.py", "class": "X"},
            {"type": "function", "name": "stub", "file": "a.py", "is_stub": True},
        ]
        assert _filter_trivial_units(units) == []


class TestPipelineRejectTrivialFlag:
    """Tests for the reject_trivial flag wired into generate_from_repo."""

    def test_flag_false_keeps_stub_and_dunder_end_to_end(self, monkeypatch):
        """With reject_trivial=False (default), stubs and dunders survive.

        If the guard were always active, stub_fn and __enter__ would be dropped —
        causing this test to fail.
        """
        stub = {"type": "function", "name": "stub_fn", "file": "mod.py", "is_stub": True}
        dunder = {"type": "method", "name": "__enter__", "file": "mod.py", "class": "Ctx"}
        good = {"type": "function", "name": "good_fn", "file": "mod.py"}
        init = {"type": "method", "name": "__init__", "file": "mod.py", "class": "Ctx"}

        monkeypatch.setattr(
            "scripts.repo.scan_repo.scan_repo",
            lambda config: [stub, dunder, good, init],
        )

        config = _make_config(
            reject_trivial=False,
            extraction_units=["functions", "methods"],
        )
        train, holdout = generate_from_repo(config)
        names = {r["output"]["name"] for r in (train + holdout)}

        assert "stub_fn" in names, "stub_fn must survive when reject_trivial=False"
        assert "__enter__" in names, "__enter__ must survive when reject_trivial=False"
        assert "good_fn" in names
        assert "__init__" in names

    def test_flag_true_drops_stub_and_dunder_end_to_end(self, monkeypatch):
        """With reject_trivial=True, stubs and dunders (except __init__) are dropped.

        Exercises both rejection categories in a single end-to-end run through
        generate_from_repo.
        """
        stub = {"type": "function", "name": "stub_fn", "file": "mod.py", "is_stub": True}
        dunder = {"type": "method", "name": "__enter__", "file": "mod.py", "class": "Ctx"}
        good = {"type": "function", "name": "good_fn", "file": "mod.py"}
        init = {"type": "method", "name": "__init__", "file": "mod.py", "class": "Ctx"}

        monkeypatch.setattr(
            "scripts.repo.scan_repo.scan_repo",
            lambda config: [stub, dunder, good, init],
        )

        config = _make_config(
            reject_trivial=True,
            extraction_units=["functions", "methods"],
        )
        train, holdout = generate_from_repo(config)
        names = {r["output"]["name"] for r in (train + holdout)}

        assert "stub_fn" not in names, "stub_fn must be dropped when reject_trivial=True"
        assert "__enter__" not in names, "__enter__ must be dropped when reject_trivial=True"
        assert "good_fn" in names, "good_fn must survive"
        assert "__init__" in names, "__init__ must survive (kept dunder)"

    def test_default_config_has_reject_trivial_false(self):
        """Default RepoConfig has reject_trivial=False."""
        config = _make_config()
        assert config.reject_trivial is False


class TestLoaderRejectTrivialRoundtrip:
    """reject_trivial round-trips through load_repo_config."""

    def test_reject_trivial_true_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_dir = tmp_path / "repo"
            repo_dir.mkdir()
            cfg_path = tmp_path / "repo.yaml"
            cfg_path.write_text(
                f"name: rt-test\npath: {repo_dir}\n"
                "generation:\n"
                "  reject_trivial: true\n"
            )
            from scripts.repo.loader import load_repo_config
            config = load_repo_config(cfg_path)
            assert config.reject_trivial is True

    def test_reject_trivial_omitted_defaults_to_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_dir = tmp_path / "repo"
            repo_dir.mkdir()
            cfg_path = tmp_path / "repo.yaml"
            cfg_path.write_text(f"name: rt-default\npath: {repo_dir}\n")
            from scripts.repo.loader import load_repo_config
            config = load_repo_config(cfg_path)
            assert config.reject_trivial is False

    def test_reject_trivial_false_from_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo_dir = tmp_path / "repo"
            repo_dir.mkdir()
            cfg_path = tmp_path / "repo.yaml"
            cfg_path.write_text(
                f"name: rt-false\npath: {repo_dir}\n"
                "generation:\n"
                "  reject_trivial: false\n"
            )
            from scripts.repo.loader import load_repo_config
            config = load_repo_config(cfg_path)
            assert config.reject_trivial is False


class TestTrivialRejectionDeterminism:
    """Same input → identical filtered output across two calls."""

    def test_same_input_same_output(self):
        units = [
            {"type": "function", "name": f"fn_{i}", "file": "a.py"}
            for i in range(5)
        ]
        units.append({"type": "function", "name": "stub", "file": "a.py", "is_stub": True})
        units.append({"type": "method", "name": "__enter__", "file": "a.py", "class": "X"})
        result1 = _filter_trivial_units(units)
        result2 = _filter_trivial_units(units)
        assert result1 == result2


class TestBothFiltersInteraction:
    """validate_syntax + reject_trivial applied together."""

    def test_both_flags_drop_respective_units(self, monkeypatch):
        """When both flags are True, malformed-sig AND stub/dunder units are both dropped."""
        malformed = {
            "type": "function", "name": "malformed", "file": "mod.py",
            "signature": "malformed(1 2)",
        }
        stub = {"type": "function", "name": "stub_fn", "file": "mod.py", "is_stub": True}
        dunder = {"type": "method", "name": "__exit__", "file": "mod.py", "class": "Ctx"}
        good = {"type": "function", "name": "good_fn", "file": "mod.py"}
        init = {"type": "method", "name": "__init__", "file": "mod.py", "class": "Ctx"}

        monkeypatch.setattr(
            "scripts.repo.scan_repo.scan_repo",
            lambda config: [malformed, stub, dunder, good, init],
        )

        config = _make_config(
            validate_syntax=True,
            reject_trivial=True,
            extraction_units=["functions", "methods"],
        )
        train, holdout = generate_from_repo(config)
        names = {r["output"]["name"] for r in (train + holdout)}

        assert "malformed" not in names, "Malformed-sig unit must be dropped by validate_syntax"
        assert "stub_fn" not in names, "Stub unit must be dropped by reject_trivial"
        assert "__exit__" not in names, "__exit__ must be dropped by reject_trivial"
        assert "good_fn" in names
        assert "__init__" in names


# ---------------------------------------------------------------------------
# Milestone 3.4: property access-form correctness
# ---------------------------------------------------------------------------

from scripts.repo.generate_from_code import (
    _is_property_unit,
    generate_code_thinking,
)

# Concrete property unit matching the real-world amesh example.
_PROP_UNIT = {
    "type": "method",
    "name": "identity",
    "class": "ResourceAgent",
    "file": "x.py",
    "lineno": 1,
    "is_property": True,
}

# Same unit without a class key (classless edge case).
_PROP_UNIT_NOCLASS = {
    "type": "method",
    "name": "identity",
    "file": "x.py",
    "lineno": 1,
    "is_property": True,
}

# A static method unit — must NOT be hijacked by the property branch.
_STATIC_UNIT = {
    "type": "method",
    "name": "create",
    "class": "MyService",
    "file": "svc.py",
    "lineno": 5,
    "method_kind": "static",
    "signature": "create(url)",
    "call_signature": "create(url)",
}

# A classmethod unit — must NOT be hijacked by the property branch.
_CLASS_UNIT = {
    "type": "method",
    "name": "from_config",
    "class": "MyService",
    "file": "svc.py",
    "lineno": 10,
    "method_kind": "class",
    "signature": "from_config(cls, cfg)",
    "call_signature": "from_config(cfg)",
}


class TestIsPropertyUnit:
    """Unit tests for the _is_property_unit predicate."""

    def test_is_property_unit_true_when_flag_set(self):
        assert _is_property_unit({"is_property": True}) is True

    def test_is_property_unit_false_when_flag_missing(self):
        assert _is_property_unit({"type": "method", "name": "x"}) is False

    def test_is_property_unit_false_when_flag_false(self):
        assert _is_property_unit({"is_property": False}) is False

    def test_is_property_unit_false_for_plain_unit(self):
        assert _is_property_unit(_STATIC_UNIT) is False
        assert _is_property_unit(_CLASS_UNIT) is False


class TestLinearPropertyTrace:
    """_linear_method (via generate_code_thinking) uses attribute-access framing for properties."""

    def test_property_trace_use_line_is_attribute_access(self):
        trace = generate_code_thinking(_PROP_UNIT, style="linear")
        # The Use: line must be attribute access (no parentheses after name on that line)
        use_line = next(ln for ln in trace.splitlines() if ln.startswith("Use:"))
        assert "instance.identity" in use_line
        assert "instance.identity(" not in use_line

    def test_property_trace_mentions_property(self):
        trace = generate_code_thinking(_PROP_UNIT, style="linear")
        assert "property" in trace

    def test_property_trace_not_line_warns_against_calling(self):
        trace = generate_code_thinking(_PROP_UNIT, style="linear")
        # The NOT line must warn against calling with ()
        assert "NOT:" in trace
        # The NOT line must reference the call form to warn against it
        assert "identity()" in trace

    def test_property_trace_contains_class_name(self):
        trace = generate_code_thinking(_PROP_UNIT, style="linear")
        assert "ResourceAgent" in trace

    def test_property_trace_unknown_class_no_literal_none(self):
        trace = generate_code_thinking(_PROP_UNIT_NOCLASS, style="linear")
        assert "None" not in trace
        # The Use: line must be attribute access
        use_line = next(ln for ln in trace.splitlines() if ln.startswith("Use:"))
        assert "instance.identity" in use_line
        assert "instance.identity(" not in use_line

    def test_property_trace_unknown_class_still_uses_access_framing(self):
        trace = generate_code_thinking(_PROP_UNIT_NOCLASS, style="linear")
        assert "property" in trace
        assert "NOT:" in trace


class TestQocPropertyTrace:
    """_qoc_method (via generate_code_thinking, style='qoc') uses attribute-access framing."""

    def test_qoc_property_option_a_is_attribute_access(self):
        trace = generate_code_thinking(_PROP_UNIT, style="qoc")
        # Option A line must be attribute access without parentheses after the name
        option_a_line = next(ln for ln in trace.splitlines() if ln.startswith("Option A"))
        assert "instance.identity" in option_a_line
        assert "instance.identity(" not in option_a_line

    def test_qoc_property_option_b_flags_call_as_incorrect(self):
        trace = generate_code_thinking(_PROP_UNIT, style="qoc")
        # Option B must show the call form and flag it as incorrect
        assert "Option B" in trace
        assert "identity()" in trace
        assert "incorrect" in trace

    def test_qoc_property_not_line_warns_against_calling(self):
        trace = generate_code_thinking(_PROP_UNIT, style="qoc")
        assert "NOT:" in trace
        assert "identity()" in trace

    def test_qoc_property_unknown_class_no_literal_none(self):
        trace = generate_code_thinking(_PROP_UNIT_NOCLASS, style="qoc")
        assert "None" not in trace
        option_a_line = next(ln for ln in trace.splitlines() if ln.startswith("Option A"))
        assert "instance.identity" in option_a_line
        assert "instance.identity(" not in option_a_line

    def test_qoc_property_contains_class_name_in_criteria(self):
        trace = generate_code_thinking(_PROP_UNIT, style="qoc")
        assert "ResourceAgent" in trace
        assert "Criteria" in trace


class TestPropertyRegressionStaticClass:
    """Property branch must NOT hijack static or classmethod units."""

    def test_static_method_still_renders_class_call_form_linear(self):
        trace = generate_code_thinking(_STATIC_UNIT, style="linear")
        # Static method must use ClassName.method() form
        assert "MyService.create" in trace
        assert "static method" in trace
        # Must NOT use property framing
        assert "attribute access" not in trace
        assert "no parentheses" not in trace

    def test_classmethod_still_renders_class_call_form_linear(self):
        trace = generate_code_thinking(_CLASS_UNIT, style="linear")
        assert "MyService.from_config" in trace
        assert "classmethod" in trace
        assert "attribute access" not in trace
        assert "no parentheses" not in trace

    def test_static_method_still_renders_class_call_form_qoc(self):
        trace = generate_code_thinking(_STATIC_UNIT, style="qoc")
        assert "MyService.create" in trace
        assert "static method" in trace
        assert "attribute access" not in trace

    def test_classmethod_still_renders_class_call_form_qoc(self):
        trace = generate_code_thinking(_CLASS_UNIT, style="qoc")
        assert "MyService.from_config" in trace
        assert "classmethod" in trace
        assert "attribute access" not in trace


class TestPropertyEndToEnd:
    """End-to-end: scan a temp repo with a @property and check generated trace."""

    def test_property_unit_e2e_linear_trace(self, tmp_path):
        """Scanning a @property method produces a record whose thinking uses access framing."""
        (tmp_path / "agents.py").write_text(
            "class ResourceAgent:\n"
            "    @property\n"
            "    def identity(self):\n"
            "        return id(self)\n"
        )

        from scripts.repo.loader import RepoConfig
        from scripts.repo.scan_repo import scan_repo as _scan_repo
        from scripts.repo.generate_from_code import generate_code_thinking

        config = RepoConfig(name="e2e-prop", path=str(tmp_path))
        units = _scan_repo(config)
        prop_units = [u for u in units if u.get("is_property") and u["name"] == "identity"]
        assert prop_units, "Expected a property unit for 'identity'"
        unit = prop_units[0]
        assert unit.get("is_property") is True

        trace = generate_code_thinking(unit, style="linear")
        # The Use: line must be attribute access (no call-form on the use line)
        use_line = next(ln for ln in trace.splitlines() if ln.startswith("Use:"))
        assert "instance.identity" in use_line
        assert "instance.identity(" not in use_line
        assert "property" in trace

    def test_property_unit_e2e_qoc_trace(self, tmp_path):
        """QOC trace for a scanned @property uses attribute-access framing."""
        (tmp_path / "agents.py").write_text(
            "class ResourceAgent:\n"
            "    @property\n"
            "    def identity(self):\n"
            "        return id(self)\n"
        )

        from scripts.repo.loader import RepoConfig
        from scripts.repo.scan_repo import scan_repo as _scan_repo
        from scripts.repo.generate_from_code import generate_code_thinking

        config = RepoConfig(name="e2e-prop-qoc", path=str(tmp_path))
        units = _scan_repo(config)
        prop_units = [u for u in units if u.get("is_property") and u["name"] == "identity"]
        assert prop_units
        unit = prop_units[0]

        trace = generate_code_thinking(unit, style="qoc")
        # Option A line must be attribute access
        option_a_line = next(ln for ln in trace.splitlines() if ln.startswith("Option A"))
        assert "instance.identity" in option_a_line
        assert "instance.identity(" not in option_a_line
        assert "incorrect" in trace  # Option B flags the call as incorrect
