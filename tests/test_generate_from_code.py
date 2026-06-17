"""Tests for generate_from_code.py (Milestone 1.4).

Hermetic tests — all repo scanning uses the sample_repo fixture or
in-memory temp directories. No network access, no LLM calls.
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from scripts.repo.loader import RepoConfig, load_repo_config
from scripts.repo.scan_repo import scan_repo
from scripts.repo.generate_from_code import (
    generate_from_code,
    generate_from_repo,
    _make_record,
    _pick_question,
    _make_thinking,
    _in_holdout,
    _split_records,
    _write_jsonl,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_repo"


def _make_config(path, holdout_ratio=0.15, include=None, exclude=None):
    """Build a minimal RepoConfig without touching disk."""
    return RepoConfig(
        name="test-gen",
        path=str(path),
        include=include or ["**/*.py"],
        exclude=exclude or [],
        holdout_ratio=holdout_ratio,
    )


def _make_unit(unit_type="function", name="do_thing", file="mod.py", **kwargs):
    """Construct a minimal scanner unit dict."""
    u = {"type": unit_type, "name": name, "file": file, "lineno": 1}
    u.update(kwargs)
    return u


@pytest.fixture()
def sample_config():
    """RepoConfig pointing at the sample_repo fixture."""
    return _make_config(FIXTURES_DIR)


@pytest.fixture()
def sample_units(sample_config):
    """All units from the sample_repo fixture."""
    return scan_repo(sample_config)


# ---------------------------------------------------------------------------
# One record per unit
# ---------------------------------------------------------------------------

def test_one_record_per_unit(sample_config, sample_units):
    """generate_from_code must emit exactly one record per input unit."""
    records = generate_from_code(sample_config, sample_units)
    assert len(records) == len(sample_units)


def test_record_count_with_explicit_units(sample_config):
    """Explicit list of units → same count of records."""
    units = [
        _make_unit("function", "foo"),
        _make_unit("class", "Bar"),
        _make_unit("method", "baz", cls_name="Bar"),
    ]
    # Add class key for method unit
    units[2]["class"] = "Bar"
    records = generate_from_code(sample_config, units)
    assert len(records) == 3


# ---------------------------------------------------------------------------
# type == "code" on every record
# ---------------------------------------------------------------------------

def test_type_code_on_all_records(sample_config, sample_units):
    """Every record must carry type='code'."""
    records = generate_from_code(sample_config, sample_units)
    for rec in records:
        assert rec.get("type") == "code", f"Missing type='code' in {rec}"


def test_type_code_on_single_record():
    unit = _make_unit("function", "my_func")
    rec = _make_record(unit)
    assert rec["type"] == "code"


# ---------------------------------------------------------------------------
# Question is non-empty and references the unit name
# ---------------------------------------------------------------------------

def test_question_nonempty(sample_config, sample_units):
    """Every record must have a non-empty question."""
    records = generate_from_code(sample_config, sample_units)
    for rec in records:
        assert rec.get("question"), f"Empty question in {rec}"


def test_question_references_name():
    """The question must contain the unit name."""
    for unit_type in ("function", "method", "class", "api_call"):
        unit = _make_unit(unit_type, name="my_special_name")
        if unit_type == "method":
            unit["class"] = "SomeClass"
        rec = _make_record(unit)
        assert "my_special_name" in rec["question"], (
            f"Unit name not found in question for {unit_type}: {rec['question']!r}"
        )


def test_question_references_name_for_all_sample_units(sample_config, sample_units):
    """For every fixture unit the question must contain the unit name."""
    records = generate_from_code(sample_config, sample_units)
    for unit, rec in zip(sample_units, records):
        assert unit["name"] in rec["question"], (
            f"Name {unit['name']!r} not in question {rec['question']!r}"
        )


# ---------------------------------------------------------------------------
# Thinking is non-empty and deterministic
# ---------------------------------------------------------------------------

def test_thinking_nonempty(sample_config, sample_units):
    """Every record must have a non-empty thinking field."""
    records = generate_from_code(sample_config, sample_units)
    for rec in records:
        assert rec.get("thinking"), f"Empty thinking in {rec}"


def test_thinking_deterministic():
    """Same unit always produces the same thinking trace."""
    unit = _make_unit("function", "stable_func", file="pkg/mod.py")
    rec1 = _make_record(unit)
    rec2 = _make_record(unit)
    assert rec1["thinking"] == rec2["thinking"]


def test_thinking_deterministic_for_all_types():
    """Determinism holds for all four scanner unit types."""
    units = [
        _make_unit("function", "f"),
        _make_unit("class", "C"),
        _make_unit("method", "m", **{"class": "C"}),
        _make_unit("api_call", "requests.get"),
    ]
    for unit in units:
        r1 = _make_record(unit)
        r2 = _make_record(unit)
        assert r1["thinking"] == r2["thinking"], (
            f"Non-deterministic thinking for {unit['type']} {unit['name']}"
        )


def test_full_generation_deterministic(sample_config, sample_units):
    """Running generate_from_code twice on the same input yields identical output."""
    records1 = generate_from_code(sample_config, sample_units)
    records2 = generate_from_code(sample_config, sample_units)
    assert records1 == records2


# Script run in fresh subprocesses to prove cross-PROCESS determinism. A bug
# using the builtin salted hash() would produce different output under
# different PYTHONHASHSEED values; SHA-256 must be invariant.
_CROSS_PROCESS_SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {repo_root!r})
    from scripts.repo.loader import RepoConfig
    from scripts.repo.generate_from_code import _split_records, _pick_question

    units = [
        {{"type": "function", "name": "func_%d" % i,
          "file": "pkg/mod%d.py" % i, "lineno": 1}}
        for i in range(120)
    ]
    # Add a few non-function types to exercise question selection too.
    units.append({{"type": "class", "name": "Widget", "file": "w.py", "lineno": 1}})
    units.append({{"type": "method", "name": "run", "file": "w.py",
                   "lineno": 2, "class": "Widget"}})
    units.append({{"type": "api_call", "name": "requests.get",
                   "file": "api.py", "lineno": 1}})

    train, hold = _split_records(units, 0.15)
    train_keys = [(r["output"]["name"], r["output"]["file"]) for r in train]
    hold_keys = [(r["output"]["name"], r["output"]["file"]) for r in hold]
    questions = [_pick_question(u) for u in units]

    print(json.dumps({{
        "train": train_keys,
        "hold": hold_keys,
        "questions": questions,
    }}))
    """
)


def _run_cross_process(hashseed):
    """Run the cross-process script with a specific PYTHONHASHSEED; return parsed output."""
    repo_root = str(Path(__file__).resolve().parent.parent)
    script = _CROSS_PROCESS_SCRIPT.format(repo_root=repo_root)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    return json.loads(result.stdout)


def test_holdout_split_deterministic_across_processes():
    """The split must be identical under different PYTHONHASHSEED values.

    This catches the latent bug where a non-stable (salted builtin hash())
    split would differ between processes — the same-process tests above cannot
    detect that because builtin hash() is stable within a single process.
    """
    out0 = _run_cross_process(0)
    out1 = _run_cross_process(1)
    assert out0["train"] == out1["train"]
    assert out0["hold"] == out1["hold"]
    # Sanity: the split actually produced both partitions.
    assert out0["train"] and out0["hold"]


def test_question_selection_deterministic_across_processes():
    """Question phrasing selection must also be invariant across PYTHONHASHSEED."""
    out0 = _run_cross_process(0)
    out1 = _run_cross_process(1)
    assert out0["questions"] == out1["questions"]


# ---------------------------------------------------------------------------
# Output object has required keys
# ---------------------------------------------------------------------------

def test_output_required_keys(sample_config, sample_units):
    """Every output object must have unit, name, file."""
    records = generate_from_code(sample_config, sample_units)
    for rec in records:
        out = rec.get("output", {})
        assert "unit" in out, f"Missing 'unit' in output: {out}"
        assert "name" in out, f"Missing 'name' in output: {out}"
        assert "file" in out, f"Missing 'file' in output: {out}"


def test_output_unit_mirrors_scanner_type():
    """output.unit must equal the scanner type field."""
    for t in ("function", "method", "class", "api_call"):
        unit = _make_unit(t, name="x")
        if t == "method":
            unit["class"] = "MyClass"
        rec = _make_record(unit)
        assert rec["output"]["unit"] == t


def test_output_file_matches_scanner():
    unit = _make_unit("function", "f", file="a/b/c.py")
    rec = _make_record(unit)
    assert rec["output"]["file"] == "a/b/c.py"


# ---------------------------------------------------------------------------
# Method records carry class in output
# ---------------------------------------------------------------------------

def test_method_output_has_class():
    """Method records must include 'class' in their output object."""
    unit = _make_unit("method", "my_method", **{"class": "MyClass"})
    rec = _make_record(unit)
    assert "class" in rec["output"], f"'class' missing from method output: {rec['output']}"
    assert rec["output"]["class"] == "MyClass"


def test_non_method_output_no_class():
    """Non-method records must NOT include 'class' in their output object."""
    for t in ("function", "class", "api_call"):
        unit = _make_unit(t, name="x")
        rec = _make_record(unit)
        assert "class" not in rec["output"], (
            f"Unexpected 'class' in output for {t}: {rec['output']}"
        )


def test_sample_fixture_method_units_carry_class(sample_config, sample_units):
    """All method units from the fixture must carry 'class' in their output."""
    records = generate_from_code(sample_config, sample_units)
    for unit, rec in zip(sample_units, records):
        if unit["type"] == "method":
            assert "class" in rec["output"], (
                f"Method {unit['name']} missing 'class' in output"
            )
            assert rec["output"]["class"] == unit["class"]


# ---------------------------------------------------------------------------
# Classless-method tolerance (hand-built units missing the 'class' key)
# ---------------------------------------------------------------------------

def test_pick_question_classless_method_no_keyerror():
    """_pick_question must not raise KeyError on a method unit without 'class'."""
    unit = {"type": "method", "name": "foo", "file": "a.py", "lineno": 1}
    q = _pick_question(unit)  # must not raise
    assert "foo" in q


def test_generate_from_code_classless_method(sample_config):
    """The pure entry point must accept hand-built method units missing 'class'."""
    units = [{"type": "method", "name": "foo", "file": "a.py", "lineno": 1}]
    records = generate_from_code(sample_config, units)  # must not raise
    assert len(records) == 1
    assert records[0]["type"] == "code"
    assert "foo" in records[0]["question"]


def test_thinking_classless_method_no_literal_none():
    """Classless method thinking must not emit the literal 'None'."""
    unit = {"type": "method", "name": "foo", "file": "a.py", "lineno": 1}
    thinking = _make_thinking(unit)
    assert "None" not in thinking


# ---------------------------------------------------------------------------
# Holdout split: honors holdout_ratio and is deterministic
# ---------------------------------------------------------------------------

def test_holdout_split_deterministic(sample_config, sample_units):
    """_split_records must be deterministic: same input → same partition."""
    train1, hold1 = _split_records(sample_units, sample_config.holdout_ratio)
    train2, hold2 = _split_records(sample_units, sample_config.holdout_ratio)
    assert train1 == train2
    assert hold1 == hold2


def test_holdout_ratio_approximate():
    """With many units, holdout size should be close to holdout_ratio."""
    # Build 200 distinct units
    units = [_make_unit("function", f"func_{i}", file=f"mod{i}.py") for i in range(200)]
    ratio = 0.15
    _, holdout = _split_records(units, ratio)
    # Allow ±10% tolerance around expected count
    expected = ratio * len(units)
    assert abs(len(holdout) - expected) <= 0.10 * len(units), (
        f"Holdout count {len(holdout)} too far from expected {expected:.0f}"
    )


def test_holdout_ratio_zero_gives_all_train():
    """holdout_ratio=0 → all units go to train."""
    units = [_make_unit("function", f"f{i}") for i in range(10)]
    train, hold = _split_records(units, 0.0)
    assert len(hold) == 0
    assert len(train) == 10


def test_holdout_ratio_one_gives_all_holdout():
    """holdout_ratio=1 → all units go to holdout."""
    units = [_make_unit("function", f"f{i}") for i in range(10)]
    train, hold = _split_records(units, 1.0)
    assert len(train) == 0
    assert len(hold) == 10


# ---------------------------------------------------------------------------
# Train + holdout partition is complete and disjoint
# ---------------------------------------------------------------------------

def test_partition_complete(sample_config, sample_units):
    """train + holdout must cover all input units (complete partition)."""
    records_all = generate_from_code(sample_config, sample_units)
    train, hold = _split_records(sample_units, sample_config.holdout_ratio)
    assert len(train) + len(hold) == len(records_all)


def test_partition_disjoint(sample_config, sample_units):
    """No record may appear in both train and holdout."""
    train, hold = _split_records(sample_units, sample_config.holdout_ratio)
    # Compare by (name, file) key to detect duplicates robustly
    train_keys = {(r["output"]["name"], r["output"]["file"]) for r in train}
    hold_keys = {(r["output"]["name"], r["output"]["file"]) for r in hold}
    overlap = train_keys & hold_keys
    assert not overlap, f"Overlapping records in train/holdout: {overlap}"


def test_partition_complete_and_disjoint_explicit():
    """Explicit unit list: every unit appears exactly once across train+holdout."""
    units = [_make_unit("function", f"fn_{i}", file=f"file{i}.py") for i in range(50)]
    train, hold = _split_records(units, 0.15)
    combined_names = [r["output"]["name"] for r in train + hold]
    unit_names = [u["name"] for u in units]
    assert sorted(combined_names) == sorted(unit_names)


# ---------------------------------------------------------------------------
# CLI writes both training.jsonl and holdout.jsonl
# ---------------------------------------------------------------------------

def test_cli_writes_both_files():
    """CLI invocation with a temp repo must create both output files."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp) / "myrepo"
        repo_path.mkdir()
        (repo_path / "mod.py").write_text(
            "def alpha(): pass\n"
            "def beta(): pass\n"
            "class Gamma:\n"
            "    def delta(self): pass\n"
        )

        out_dir = Path(tmp) / "out"

        cfg_path = repo_path / "repo.yaml"
        cfg_path.write_text(
            f"name: myrepo\npath: {repo_path}\n"
        )

        _SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "repo" / "generate_from_code.py"
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                str(cfg_path),
                "--output-dir",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

        train_path = out_dir / "myrepo" / "training.jsonl"
        holdout_path = out_dir / "myrepo" / "holdout.jsonl"
        assert train_path.exists(), "training.jsonl not written"
        assert holdout_path.exists(), "holdout.jsonl not written"


def test_cli_output_valid_jsonl():
    """CLI output files must contain valid JSON objects, one per line."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp) / "testrepo"
        repo_path.mkdir()
        (repo_path / "module.py").write_text(
            "def foo(): pass\n"
            "class Bar:\n"
            "    def baz(self): pass\n"
        )

        out_dir = Path(tmp) / "out"
        cfg_path = repo_path / "repo.yaml"
        cfg_path.write_text(f"name: testrepo\npath: {repo_path}\n")

        _SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "repo" / "generate_from_code.py"
        subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                str(cfg_path),
                "--output-dir",
                str(out_dir),
            ],
            check=True,
        )

        train_path = out_dir / "testrepo" / "training.jsonl"
        holdout_path = out_dir / "testrepo" / "holdout.jsonl"

        for path in (train_path, holdout_path):
            lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
            for line in lines:
                obj = json.loads(line)  # must not raise
                assert isinstance(obj, dict)
                assert obj.get("type") == "code"


def test_cli_dry_run_writes_nothing():
    """--dry-run must not create any output files."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp) / "dryrepo"
        repo_path.mkdir()
        (repo_path / "mod.py").write_text("def x(): pass\n")

        out_dir = Path(tmp) / "out"
        cfg_path = repo_path / "repo.yaml"
        cfg_path.write_text(f"name: dryrepo\npath: {repo_path}\n")

        _SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "repo" / "generate_from_code.py"
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                str(cfg_path),
                "--output-dir",
                str(out_dir),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"CLI dry-run failed: {result.stderr}"
        assert not (out_dir / "dryrepo" / "training.jsonl").exists()
        assert not (out_dir / "dryrepo" / "holdout.jsonl").exists()


def test_cli_rejects_path_traversal_name():
    """A repo name with '..' must be rejected and must not escape the output root."""
    with tempfile.TemporaryDirectory() as tmp:
        repo_path = Path(tmp) / "evilrepo"
        repo_path.mkdir()
        (repo_path / "mod.py").write_text("def x(): pass\n")

        out_dir = Path(tmp) / "out"
        out_dir.mkdir()
        cfg_path = repo_path / "repo.yaml"
        # name contains a traversal sequence
        cfg_path.write_text(f"name: ../../escaped\npath: {repo_path}\n")

        _SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "repo" / "generate_from_code.py"
        result = subprocess.run(
            [
                sys.executable,
                str(_SCRIPT),
                str(cfg_path),
                "--output-dir",
                str(out_dir),
            ],
            capture_output=True,
            text=True,
        )
        # Must exit non-zero with a clear error and write nothing outside out_dir.
        assert result.returncode != 0, "Traversal name should be rejected"
        assert "unsafe" in (result.stderr + result.stdout).lower()
        escaped = Path(tmp) / "escaped"
        assert not escaped.exists(), "Output escaped the intended root"


# ---------------------------------------------------------------------------
# generate_from_repo wrapper
# ---------------------------------------------------------------------------

def test_generate_from_repo_returns_tuple(sample_config):
    """generate_from_repo must return (train, holdout) tuple."""
    train, hold = generate_from_repo(sample_config)
    assert isinstance(train, list)
    assert isinstance(hold, list)
    assert len(train) + len(hold) > 0


def test_generate_from_repo_type_code(sample_config):
    """generate_from_repo results must all carry type='code'."""
    train, hold = generate_from_repo(sample_config)
    for rec in train + hold:
        assert rec["type"] == "code"


# ---------------------------------------------------------------------------
# _write_jsonl helper
# ---------------------------------------------------------------------------

def test_write_jsonl_creates_dirs_and_file():
    """_write_jsonl must create parent directories as needed."""
    with tempfile.TemporaryDirectory() as tmp:
        deep_path = Path(tmp) / "a" / "b" / "c" / "out.jsonl"
        records = [{"type": "code", "question": "q", "thinking": "t", "output": {}}]
        _write_jsonl(records, deep_path)
        assert deep_path.exists()
        lines = [l for l in deep_path.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0]) == records[0]
