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
