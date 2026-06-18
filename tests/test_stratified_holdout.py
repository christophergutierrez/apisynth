"""Tests for the stratified holdout split (Milestone 3.1).

Covers: _stratified_holdout_keys, the "stratified" branch of _split_records,
loader parsing of holdout_strategy, and backward-compat of the "hash" default.
"""

import hashlib
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from scripts.repo.generate_from_code import (
    _in_holdout,
    _split_records,
    _stratified_holdout_keys,
)
from scripts.repo.loader import (
    DEFAULT_HOLDOUT_STRATEGY,
    VALID_HOLDOUT_STRATEGIES,
    RepoConfig,
    load_repo_config,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit(unit_type, name, file="mod.py", **kwargs):
    u = {"type": unit_type, "name": name, "file": file, "lineno": 1}
    u.update(kwargs)
    return u


def _build_mixed_units():
    """Return 280 synthetic units: 100 function, 50 method, 25 class, 105 api_call."""
    units = []
    for i in range(100):
        units.append(_make_unit("function", f"func_{i}", file=f"f/func_{i}.py"))
    for i in range(50):
        units.append(_make_unit("method", f"meth_{i}", file=f"m/meth_{i}.py"))
    for i in range(25):
        units.append(_make_unit("class", f"Cls_{i}", file=f"c/cls_{i}.py"))
    for i in range(105):
        units.append(_make_unit("api_call", f"api_{i}", file=f"a/api_{i}.py"))
    return units


# ---------------------------------------------------------------------------
# 1. Per-type ~20% holdout counts
# ---------------------------------------------------------------------------

def test_stratified_per_type_counts():
    """Each type's holdout count == round(0.20 * n) — exact counts checked."""
    units = _build_mixed_units()
    ratio = 0.20

    _, holdout = _split_records(units, ratio, strategy="stratified")

    # Count holdout by type (the 'unit' key in the output object mirrors type).
    by_type: dict = {}
    for rec in holdout:
        t = rec["output"]["unit"]
        by_type[t] = by_type.get(t, 0) + 1

    assert by_type.get("function", 0) == round(0.20 * 100)   # == 20
    assert by_type.get("method", 0)   == round(0.20 * 50)    # == 10
    assert by_type.get("class", 0)    == round(0.20 * 25)    # ==  5
    assert by_type.get("api_call", 0) == round(0.20 * 105)   # == 21


def test_stratified_count_uses_bankers_rounding():
    """A stratum of n=8 at ratio 0.20 → round(1.6)=2 (NOT int truncation to 1).

    Locks in round() semantics: a round→int mutation would yield 1 here.
    """
    units = [_make_unit("function", f"f{i}", file=f"r/f{i}.py") for i in range(8)]
    _, hold = _split_records(units, 0.20, strategy="stratified")
    assert len(hold) == 2, f"Expected 2 (round(0.2*8)), got {len(hold)}"


def test_stratified_collision_across_types_exact_counts():
    """Same file:name in DIFFERENT strata: per-type counts stay exact, no leakage.

    A top-level function and a method can share "<file>:<name>". If the two
    strata disagree on selecting that key, keying on the bare string would drag
    both into holdout (±1 count drift). Keying on (type,file,name) keeps each
    stratum's holdout exactly round(ratio*n) and the partition disjoint.
    """
    ratio = 0.20

    # Build 10 functions and 10 methods. Deliberately give ONE function and ONE
    # method the identical file:name so their strata may disagree on selection.
    funcs = [_make_unit("function", f"fn_{i}", file=f"p/fn_{i}.py") for i in range(9)]
    meths = [_make_unit("method", f"mt_{i}", file=f"p/mt_{i}.py", **{"class": "C"}) for i in range(9)]
    # The colliding pair: same file:name, different type.
    funcs.append(_make_unit("function", "shared", file="p/shared.py"))
    meths.append(_make_unit("method", "shared", file="p/shared.py", **{"class": "C"}))

    units = funcs + meths  # 10 function, 10 method

    train, hold = _split_records(units, ratio, strategy="stratified")

    # Per-type holdout counts must each be exactly round(0.20 * 10) == 2.
    by_type: dict = {}
    for rec in hold:
        t = rec["output"]["unit"]
        by_type[t] = by_type.get(t, 0) + 1
    assert by_type.get("function", 0) == 2, f"function holdout {by_type.get('function')} != 2"
    assert by_type.get("method", 0) == 2, f"method holdout {by_type.get('method')} != 2"

    # No (type,file,name) key may appear in both train and holdout.
    train_keys = {(r["output"]["unit"], r["output"]["file"], r["output"]["name"]) for r in train}
    hold_keys = {(r["output"]["unit"], r["output"]["file"], r["output"]["name"]) for r in hold}
    assert train_keys & hold_keys == set(), "Overlap between train and holdout"
    # Complete partition.
    assert len(train) + len(hold) == len(units)


def test_stratified_api_call_is_not_hash_skewed():
    """For 105 api_call units the stratified split yields 21, not ~43 from hash."""
    units = [_make_unit("api_call", f"api_{i}", file=f"a/api_{i}.py") for i in range(105)]
    ratio = 0.20

    _, hold_strat = _split_records(units, ratio, strategy="stratified")
    _, hold_hash  = _split_records(units, ratio, strategy="hash")

    assert len(hold_strat) == 21, f"Expected 21, got {len(hold_strat)}"
    # The hash split is KNOWN to deviate significantly on this shard; assert it differs.
    assert len(hold_hash) != 21, (
        "Hash split unexpectedly also returned 21 — test may need re-baselining "
        "if the sample happens to be perfectly uniform."
    )


# ---------------------------------------------------------------------------
# 2. Determinism and order-independence
# ---------------------------------------------------------------------------

def test_stratified_deterministic_same_input():
    """Two calls with identical input produce identical holdout key sets."""
    units = _build_mixed_units()
    keys1 = _stratified_holdout_keys(units, 0.20)
    keys2 = _stratified_holdout_keys(units, 0.20)
    assert keys1 == keys2


def test_stratified_order_independent():
    """Shuffling the input list does not change the holdout assignment."""
    units = _build_mixed_units()
    keys_original = _stratified_holdout_keys(units, 0.20)

    shuffled = units[:]
    random.seed(42)
    random.shuffle(shuffled)
    keys_shuffled = _stratified_holdout_keys(shuffled, 0.20)

    assert keys_original == keys_shuffled, (
        "Holdout keys changed after shuffling — split is not order-independent."
    )


def test_stratified_split_records_order_independent():
    """_split_records with strategy='stratified' is order-independent on partition sizes."""
    units = _build_mixed_units()
    train1, hold1 = _split_records(units, 0.20, strategy="stratified")

    shuffled = units[:]
    random.seed(99)
    random.shuffle(shuffled)
    train2, hold2 = _split_records(shuffled, 0.20, strategy="stratified")

    assert len(hold1) == len(hold2)
    assert len(train1) == len(train2)


# ---------------------------------------------------------------------------
# 3. Process-stability: sha256 not builtin hash()
# ---------------------------------------------------------------------------

_STABILITY_SCRIPT = """
import sys, json, os
sys.path.insert(0, {repo_root!r})
from scripts.repo.generate_from_code import _stratified_holdout_keys

units = [
    {{"type": "function", "name": "func_%d" % i, "file": "f/func_%d.py" % i, "lineno": 1}}
    for i in range(40)
] + [
    {{"type": "api_call", "name": "api_%d" % i, "file": "a/api_%d.py" % i, "lineno": 1}}
    for i in range(105)
]

keys = sorted(_stratified_holdout_keys(units, 0.20))
print(json.dumps(keys))
"""


def _run_stability(hashseed):
    repo_root = str(Path(__file__).resolve().parent.parent)
    script = _STABILITY_SCRIPT.format(repo_root=repo_root)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    import json
    return json.loads(result.stdout)


def test_stratified_process_stable():
    """Stratified keys are identical under different PYTHONHASHSEED values."""
    keys0 = _run_stability(0)
    keys1 = _run_stability(1)
    assert keys0 == keys1, "Stratified split differs between PYTHONHASHSEED=0 and 1"


# ---------------------------------------------------------------------------
# 4. Backward-compat: strategy="hash" == legacy _in_holdout path
# ---------------------------------------------------------------------------

def test_hash_strategy_identical_to_legacy():
    """strategy='hash' produces the exact same partition as the original _in_holdout path."""
    units = _build_mixed_units()
    ratio = 0.15

    # New explicit hash strategy.
    train_new, hold_new = _split_records(units, ratio, strategy="hash")

    # Rebuild expected sets using the legacy helper directly.
    expected_holdout_names = {
        (u["file"], u["name"]) for u in units if _in_holdout(u, ratio)
    }
    actual_holdout_names = {
        (r["output"]["file"], r["output"]["name"]) for r in hold_new
    }
    assert actual_holdout_names == expected_holdout_names


def test_default_strategy_is_hash():
    """Omitting strategy= (the default) behaves identically to strategy='hash'."""
    units = [_make_unit("function", f"f{i}", file=f"m{i}.py") for i in range(80)]
    ratio = 0.15

    train_default, hold_default = _split_records(units, ratio)
    train_hash,    hold_hash    = _split_records(units, ratio, strategy="hash")

    assert len(hold_default) == len(hold_hash)
    assert [r["output"]["name"] for r in hold_default] == [r["output"]["name"] for r in hold_hash]


# ---------------------------------------------------------------------------
# 5. Loader: holdout_strategy parsing and validation
# ---------------------------------------------------------------------------

def test_loader_holdout_strategy_defaults_to_hash():
    """When holdout_strategy is absent from YAML it defaults to 'hash'."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "repo.yaml"
        cfg_path.write_text(yaml.dump({"name": "t", "path": tmp}))
        config = load_repo_config(cfg_path)
    assert config.holdout_strategy == "hash"


def test_loader_holdout_strategy_stratified():
    """holdout_strategy: stratified is parsed from the generation section."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "repo.yaml"
        data = {
            "name": "t",
            "path": tmp,
            "generation": {"holdout_strategy": "stratified", "holdout_ratio": 0.20},
        }
        cfg_path.write_text(yaml.dump(data))
        config = load_repo_config(cfg_path)
    assert config.holdout_strategy == "stratified"


def test_loader_holdout_strategy_invalid_raises():
    """An unrecognised holdout_strategy raises ValueError."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "repo.yaml"
        data = {
            "name": "t",
            "path": tmp,
            "generation": {"holdout_strategy": "random_forest"},
        }
        cfg_path.write_text(yaml.dump(data))
        with pytest.raises(ValueError, match="holdout_strategy"):
            load_repo_config(cfg_path)


def test_loader_constants_exported():
    """Module constants are importable and well-formed."""
    assert DEFAULT_HOLDOUT_STRATEGY == "hash"
    assert "hash" in VALID_HOLDOUT_STRATEGIES
    assert "stratified" in VALID_HOLDOUT_STRATEGIES


def test_repoconfig_default_holdout_strategy():
    """RepoConfig default holdout_strategy is 'hash'."""
    cfg = RepoConfig(name="x", path="/tmp")
    assert cfg.holdout_strategy == "hash"


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

def test_stratified_single_unit_type():
    """A type with n=1 at ratio=0.20: round(0.20*1)=0 → unit goes to train."""
    units = [_make_unit("class", "Lonely", file="lone.py")]
    keys = _stratified_holdout_keys(units, 0.20)
    assert len(keys) == 0


def test_stratified_empty_units():
    """Empty unit list must not crash; both partitions are empty."""
    train, hold = _split_records([], 0.20, strategy="stratified")
    assert train == []
    assert hold == []


def test_stratified_single_type_absent():
    """A type group not present in units is simply absent — no KeyError."""
    units = [_make_unit("function", f"f{i}", file=f"m{i}.py") for i in range(10)]
    # Only "function" units; no "class" → must not crash.
    keys = _stratified_holdout_keys(units, 0.20)
    assert isinstance(keys, set)
    assert len(keys) == round(0.20 * 10)


def test_stratified_all_types_complete_partition():
    """train + holdout covers all units exactly once (no duplicates or gaps)."""
    units = _build_mixed_units()
    train, hold = _split_records(units, 0.20, strategy="stratified")
    assert len(train) + len(hold) == len(units)

    all_keys_train = [(r["output"]["file"], r["output"]["name"]) for r in train]
    all_keys_hold  = [(r["output"]["file"], r["output"]["name"]) for r in hold]
    assert len(set(all_keys_train) & set(all_keys_hold)) == 0, "Overlap between train and holdout"
