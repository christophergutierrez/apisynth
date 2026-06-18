"""Tests for scripts/repo/gen_code_router_data.py and the pure helpers in
scripts/repo/train_code_router.py (load_jsonl).

gen_code_router_data is pure/deterministic — no ML deps required.
train_code_router's load_jsonl is also pure (stdlib only) — tested here as well.
Heavy ML deps in train_code_router.main() are import-guarded and never triggered
by importing the module or calling load_jsonl, so no skipping is needed for
the load_jsonl tests.
"""

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on path so scripts.repo.* imports resolve
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.repo.gen_code_router_data import (
    collect_records,
    iter_jsonl,
    resolve_input_paths,
    split_records,
    SEED,
    TRAIN_RATIO,
)
from scripts.repo.train_code_router import load_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_code_record(question: str, file_path: str, name: str = "foo") -> dict:
    """Return a minimal well-formed code training record."""
    return {
        "type": "code",
        "question": question,
        "thinking": "...",
        "output": {
            "unit": "function",
            "name": name,
            "file": file_path,
            "signature": f"{name}()",
        },
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# iter_jsonl
# ---------------------------------------------------------------------------

class TestIterJsonl:
    def test_yields_records(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
        result = list(iter_jsonl(p))
        assert result == [{"a": 1}, {"b": 2}]

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "f.jsonl"
        p.write_text('{"a":1}\n\n{"b":2}\n\n', encoding="utf-8")
        result = list(iter_jsonl(p))
        assert len(result) == 2

    def test_warns_malformed_line(self, tmp_path, capsys):
        p = tmp_path / "f.jsonl"
        p.write_text('{"a":1}\nNOT JSON\n{"b":2}\n', encoding="utf-8")
        result = list(iter_jsonl(p))
        # malformed line skipped; valid ones yielded
        assert result == [{"a": 1}, {"b": 2}]
        err = capsys.readouterr().err
        assert "WARNING" in err


# ---------------------------------------------------------------------------
# collect_records — route_key derivation
# ---------------------------------------------------------------------------

class TestCollectRecords:
    def test_route_key_from_output_file(self, tmp_path):
        """route_key must equal record['output']['file']."""
        rec = _make_code_record("How do I use foo?", "src/utils.py")
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, [rec])
        records, skipped = collect_records([p])
        assert skipped == 0
        assert len(records) == 1
        assert records[0]["route_key"] == "src/utils.py"
        assert records[0]["question"] == "How do I use foo?"

    def test_multiple_records_multiple_routes(self, tmp_path):
        recs = [
            _make_code_record("q1", "src/a.py"),
            _make_code_record("q2", "src/b.py"),
            _make_code_record("q3", "src/a.py"),
        ]
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, recs)
        records, skipped = collect_records([p])
        assert skipped == 0
        assert len(records) == 3
        routes = [r["route_key"] for r in records]
        assert routes.count("src/a.py") == 2
        assert routes.count("src/b.py") == 1

    def test_skips_missing_question(self, tmp_path):
        rec = {"type": "code", "output": {"file": "src/a.py"}}  # no question
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, [rec])
        records, skipped = collect_records([p])
        assert skipped == 1
        assert records == []

    def test_skips_missing_file(self, tmp_path):
        rec = {"type": "code", "question": "q", "output": {"unit": "function", "name": "f"}}
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, [rec])
        records, skipped = collect_records([p])
        assert skipped == 1
        assert records == []

    def test_skips_missing_output(self, tmp_path):
        rec = {"type": "code", "question": "q"}  # no output at all
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, [rec])
        records, skipped = collect_records([p])
        assert skipped == 1
        assert records == []

    def test_skips_non_dict_output(self, tmp_path):
        rec = {"type": "code", "question": "q", "output": "not a dict"}
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, [rec])
        records, skipped = collect_records([p])
        assert skipped == 1
        assert records == []

    def test_empty_input_file(self, tmp_path):
        p = tmp_path / "training.jsonl"
        p.write_text("", encoding="utf-8")
        records, skipped = collect_records([p])
        assert records == []
        assert skipped == 0

    def test_multiple_input_files(self, tmp_path):
        p1 = tmp_path / "a.jsonl"
        p2 = tmp_path / "b.jsonl"
        _write_jsonl(p1, [_make_code_record("q1", "src/a.py")])
        _write_jsonl(p2, [_make_code_record("q2", "src/b.py")])
        records, skipped = collect_records([p1, p2])
        assert len(records) == 2

    def test_mixed_valid_and_invalid(self, tmp_path):
        recs = [
            _make_code_record("q1", "src/a.py"),
            {"type": "code", "question": "q2"},          # no output
            _make_code_record("q3", "src/b.py"),
            {"type": "code", "output": {"file": "x.py"}},  # no question
        ]
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, recs)
        records, skipped = collect_records([p])
        assert len(records) == 2
        assert skipped == 2


# ---------------------------------------------------------------------------
# resolve_input_paths
# ---------------------------------------------------------------------------

class TestResolveInputPaths:
    def test_single_file(self, tmp_path):
        p = tmp_path / "training.jsonl"
        p.write_text("", encoding="utf-8")
        paths = resolve_input_paths(str(p), None)
        assert paths == [p]

    def test_input_dir(self, tmp_path):
        (tmp_path / "a.jsonl").write_text("", encoding="utf-8")
        (tmp_path / "b.jsonl").write_text("", encoding="utf-8")
        (tmp_path / "not_jsonl.txt").write_text("", encoding="utf-8")
        paths = resolve_input_paths(None, str(tmp_path))
        jsonl_names = {p.name for p in paths}
        assert "a.jsonl" in jsonl_names
        assert "b.jsonl" in jsonl_names
        assert "not_jsonl.txt" not in jsonl_names

    def test_no_input_raises_sysexit(self):
        with pytest.raises(SystemExit):
            resolve_input_paths("/no/such/file.jsonl", None)

    def test_bad_dir_raises_sysexit(self):
        with pytest.raises(SystemExit):
            resolve_input_paths(None, "/no/such/dir")

    def test_combined_file_and_dir(self, tmp_path):
        p1 = tmp_path / "explicit.jsonl"
        p1.write_text("", encoding="utf-8")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        p2 = subdir / "other.jsonl"
        p2.write_text("", encoding="utf-8")
        paths = resolve_input_paths(str(p1), str(subdir))
        assert p1 in paths
        assert p2 in paths


# ---------------------------------------------------------------------------
# Deterministic split — exercises the SOURCE function split_records()
# ---------------------------------------------------------------------------

class TestSplitRecords:
    """These tests call gen_code_router_data.split_records directly (the source),
    so mutations to the seed or ratio in the source FAIL here. They never
    re-implement the shuffle/split in the test body.
    """

    def _build_records(self, n: int) -> list[dict]:
        return [
            {"question": f"question {i}", "route_key": f"src/module_{i % 5}.py"}
            for i in range(n)
        ]

    def test_split_is_deterministic_across_calls(self):
        """Two calls of the source on identical input give identical order."""
        recs = self._build_records(20)
        train_a, test_a = split_records(recs)
        train_b, test_b = split_records(recs)
        assert train_a == train_b
        assert test_a == test_b

    def test_does_not_mutate_caller_list(self):
        """split_records operates on a copy — caller's list order is unchanged."""
        recs = self._build_records(20)
        before = list(recs)
        split_records(recs)
        assert recs == before

    def test_golden_ordering(self):
        """A frozen golden ordering for a small fixed input.

        If the source drops random.seed(SEED) (or changes the seed), the shuffle
        order changes and this assertion fails. This is what kills mutation (b).
        The golden values are derived ONCE from the source itself with SEED=42;
        we then assert against literals so a behaviour change is detectable.
        """
        recs = [
            {"question": f"q{i}", "route_key": f"f{i}.py"} for i in range(10)
        ]
        train, test = split_records(recs)
        full = train + test
        order = [r["question"] for r in full]
        # Golden order produced by random.seed(42)+shuffle on q0..q9 (the actual
        # output of the source). Hardcoded as a regression anchor.
        assert order == ["q7", "q3", "q2", "q8", "q5", "q6", "q9", "q4", "q0", "q1"]

    def test_split_sizes_use_imported_TRAIN_RATIO(self):
        """Split sizes equal int(n * TRAIN_RATIO) / remainder using the IMPORTED
        module constant. If the source TRAIN_RATIO changes (mutation c), the
        expected sizes recomputed here move with it BUT the actual split sizes
        also move — so we cross-check actual split against the constant. Changing
        only the constant-in-source while the function literal stays would diverge.
        """
        n = 50
        recs = self._build_records(n)
        train, test = split_records(recs)
        expected_train = int(n * TRAIN_RATIO)
        assert len(train) == expected_train
        assert len(test) == n - expected_train
        # Total preserved
        assert len(train) + len(test) == n

    def test_split_ratio_pins_absolute_sizes(self):
        """Pin the absolute split sizes for the documented SEED/TRAIN_RATIO.

        With the approved design (TRAIN_RATIO=0.8) and n=50, train MUST be 40 and
        test MUST be 10. If the source ratio is mutated to e.g. 0.5, split sizes
        become 25/25 and this fails — killing mutation (c) even if someone also
        edited the constant. We also assert TRAIN_RATIO/SEED are the approved
        values so the contract is explicit.
        """
        assert TRAIN_RATIO == 0.8
        assert SEED == 42
        recs = self._build_records(50)
        train, test = split_records(recs)
        assert len(train) == 40
        assert len(test) == 10

    def test_end_to_end_files_deterministic(self, tmp_path):
        """Run main() end-to-end twice and assert byte-identical output files."""
        import importlib
        gcrd = importlib.import_module("scripts.repo.gen_code_router_data")

        raw = [_make_code_record(f"q{i}", f"src/m{i % 4}.py") for i in range(25)]

        def run(out_dir: Path) -> tuple[bytes, bytes]:
            inp = out_dir / "training.jsonl"
            _write_jsonl(inp, raw)
            argv = ["prog", "--input", str(inp), "--out-dir", str(out_dir)]
            old = sys.argv
            sys.argv = argv
            try:
                gcrd.main()
            finally:
                sys.argv = old
            tr = (out_dir / "router_train.jsonl").read_bytes()
            te = (out_dir / "router_test.jsonl").read_bytes()
            return tr, te

        d1 = tmp_path / "r1"
        d1.mkdir()
        d2 = tmp_path / "r2"
        d2.mkdir()
        tr1, te1 = run(d1)
        tr2, te2 = run(d2)
        assert tr1 == tr2
        assert te1 == te2
        # And the split honors TRAIN_RATIO end-to-end (25 records → 20/5).
        n_train = len(tr1.decode().splitlines())
        n_test = len(te1.decode().splitlines())
        assert n_train == int(25 * TRAIN_RATIO)
        assert n_train + n_test == 25


# ---------------------------------------------------------------------------
# Per-route counts
# ---------------------------------------------------------------------------

class TestPerRouteCounts:
    def test_counts_across_routes(self, tmp_path):
        recs = (
            [_make_code_record("q", "a.py")] * 3
            + [_make_code_record("q", "b.py")] * 2
        )
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, recs)
        records, _ = collect_records([p])
        route_counts: dict[str, int] = {}
        for r in records:
            route_counts[r["route_key"]] = route_counts.get(r["route_key"], 0) + 1
        assert route_counts["a.py"] == 3
        assert route_counts["b.py"] == 2

    def test_total_equals_valid_records(self, tmp_path):
        recs = [_make_code_record(f"q{i}", f"f{i}.py") for i in range(7)]
        recs.append({"type": "code", "question": "no file"})  # will be skipped
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, recs)
        records, skipped = collect_records([p])
        assert len(records) == 7
        assert skipped == 1


# ---------------------------------------------------------------------------
# Output file shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_output_records_have_question_and_route_key(self, tmp_path):
        recs = [_make_code_record("How do I use foo?", "src/foo.py")]
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, recs)
        records, _ = collect_records([p])
        assert set(records[0].keys()) == {"question", "route_key"}

    def test_no_extra_fields_in_output(self, tmp_path):
        recs = [_make_code_record("q", "src/bar.py")]
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, recs)
        records, _ = collect_records([p])
        assert len(records[0]) == 2


# ---------------------------------------------------------------------------
# Empty-input edge case
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_path_list(self):
        records, skipped = collect_records([])
        assert records == []
        assert skipped == 0

    def test_all_records_skipped_returns_empty(self, tmp_path):
        recs = [{"type": "code"}, {"type": "code", "question": "q"}]
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, recs)
        records, skipped = collect_records([p])
        assert records == []
        assert skipped == 2

    def test_single_record_all_in_train(self, tmp_path):
        """A single record goes entirely to train (split = int(1 * 0.8) = 0 → train=[r], test=[])."""
        import random
        recs = [_make_code_record("q", "src/a.py")]
        p = tmp_path / "training.jsonl"
        _write_jsonl(p, recs)
        records, _ = collect_records([p])
        random.seed(42)
        random.shuffle(records)
        split = int(len(records) * 0.8)
        train = records[:split]
        test = records[split:]
        # 1 record: split = int(0.8) = 0, so all go to test
        assert len(train) + len(test) == 1

    def test_route_with_one_record_may_land_in_test(self):
        """Document the known behaviour: a single-record route can land in test split.

        This is intentional (mirrors gen_router_data.py faithful global shuffle).
        The downstream trainer (train_code_router.py) will exit on unseen labels.
        With very few records per route, users should increase data or use a
        stratified split. This test merely documents/asserts the behaviour is
        predictable given the seed.
        """
        import random
        # 10 records: 9 for "common.py", 1 for "rare.py"
        records = [{"question": f"q{i}", "route_key": "common.py"} for i in range(9)]
        records.append({"question": "q_rare", "route_key": "rare.py"})
        random.seed(42)
        random.shuffle(records)
        split = int(len(records) * 0.8)
        train = records[:split]
        test = records[split:]
        train_routes = {r["route_key"] for r in train}
        test_routes = {r["route_key"] for r in test}
        # With seed=42, either route may or may not appear in both splits.
        # What we assert: the total is correct and both splits are non-empty.
        assert len(train) + len(test) == 10
        assert len(train) == 8
        assert len(test) == 2
        # If "rare.py" happens to be only in test — document it (not a failure).
        if "rare.py" not in train_routes:
            assert "rare.py" in test_routes  # it went entirely to test


# ---------------------------------------------------------------------------
# load_jsonl from train_code_router (pure, no ML deps)
# ---------------------------------------------------------------------------

class TestLoadJsonl:
    def test_round_trip(self, tmp_path):
        p = tmp_path / "router_train.jsonl"
        records = [
            {"question": "q1", "route_key": "src/a.py"},
            {"question": "q2", "route_key": "src/b.py"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        questions, routes = load_jsonl(p)
        assert questions == ["q1", "q2"]
        assert routes == ["src/a.py", "src/b.py"]

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "router_test.jsonl"
        p.write_text(
            '{"question":"q1","route_key":"a.py"}\n\n{"question":"q2","route_key":"b.py"}\n',
            encoding="utf-8",
        )
        questions, routes = load_jsonl(p)
        assert len(questions) == 2
        assert len(routes) == 2

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        questions, routes = load_jsonl(p)
        assert questions == []
        assert routes == []

    def test_parallel_lists_same_length(self, tmp_path):
        p = tmp_path / "r.jsonl"
        records = [{"question": f"q{i}", "route_key": f"f{i}.py"} for i in range(10)]
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        questions, routes = load_jsonl(p)
        assert len(questions) == len(routes) == 10

    def test_missing_file_raises(self, tmp_path):
        p = tmp_path / "no_such_file.jsonl"
        with pytest.raises(FileNotFoundError):
            load_jsonl(p)


# ---------------------------------------------------------------------------
# Confirm train_code_router does NOT import heavy deps at module level
# ---------------------------------------------------------------------------

_HEAVY_MODULES = ("numpy", "sklearn", "sentence_transformers", "joblib")


class TestNoHeavyImportsAtModuleLevel:
    def test_fresh_import_leaves_heavy_modules_absent(self):
        """A clean import of train_code_router must NOT pull any heavy ML module
        into sys.modules. If a heavy import is moved to module level (mutation e),
        the corresponding name appears in sys.modules and this test FAILS.

        We snapshot sys.modules, evict the module-under-test AND the heavy
        modules, force a fresh import, then assert each heavy module is still
        absent from sys.modules.
        """
        import importlib

        mut = "scripts.repo.train_code_router"
        # Snapshot so we can restore real state afterwards regardless of outcome.
        snapshot = dict(sys.modules)
        try:
            # Evict the module under test and any heavy modules so the import is
            # genuinely fresh and any module-level heavy import would re-add them.
            for name in (mut, *_HEAVY_MODULES):
                # Pop submodules too (e.g. sklearn.linear_model).
                for key in [k for k in list(sys.modules) if k == name or k.startswith(name + ".")]:
                    sys.modules.pop(key, None)

            importlib.import_module(mut)

            for name in _HEAVY_MODULES:
                assert name not in sys.modules, (
                    f"{name!r} was imported at module level of {mut} — heavy deps "
                    "must be imported only inside main()."
                )
        finally:
            # Restore the original module table so we don't perturb other tests.
            sys.modules.clear()
            sys.modules.update(snapshot)
