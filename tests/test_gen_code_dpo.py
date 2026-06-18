"""Unit tests for scripts/repo/gen_code_dpo.py — code-unit DPO pair generation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "repo"))
from gen_code_dpo import (
    _generate_rejected_code_outputs,
    _is_valid_negative,
    _make_code_dpo_record,
    _output_is_valid,
    gen_code_dpo,
)

# eval lives in scripts/ — reachable because gen_code_dpo already inserted it.
from eval import code_field_accuracy


def _is_valid_dpo_negative(pair: dict) -> bool:
    """True iff the written pair's rejected obeys the keep-rule against chosen."""
    return _is_valid_negative(pair["rejected"], pair["chosen"])


# ── Fixtures ────────────────────────────────────────────────────────────────

def _valid_output(unit="function", name="my_func", file="src/foo.py",
                  signature="my_func(x: int) -> str", cls=None) -> dict:
    """Return a well-formed code output dict."""
    out: dict = {"unit": unit, "name": name, "file": file, "signature": signature}
    if cls is not None:
        out["class"] = cls
    return out


def _valid_method_output() -> dict:
    return _valid_output(unit="method", name="do_thing",
                         file="src/mymodule.py",
                         signature="do_thing(self, x: int)",
                         cls="MyClass")


# ── Tests: _output_is_valid ─────────────────────────────────────────────────

class TestOutputIsValid:
    def test_valid_function_output(self):
        assert _output_is_valid(_valid_output()) is True

    def test_valid_method_with_class(self):
        assert _output_is_valid(_valid_method_output()) is True

    def test_missing_required_key_fails(self):
        out = _valid_output()
        del out["signature"]
        assert _output_is_valid(out) is False

    def test_wrong_unit_type_fails(self):
        out = _valid_output(unit="not_a_unit")
        assert _output_is_valid(out) is False

    def test_garbled_signature_fails(self):
        out = _valid_output(signature="((( unbalanced")
        assert _output_is_valid(out) is False

    def test_empty_name_fails(self):
        out = _valid_output(name="")
        assert _output_is_valid(out) is False

    def test_non_dict_fails(self):
        assert _output_is_valid("not a dict") is False
        assert _output_is_valid(None) is False


# ── Tests: _generate_rejected_code_outputs ──────────────────────────────────

class TestGenerateRejectedCodeOutputs:
    def test_always_generates_at_least_one_candidate(self):
        out = _valid_output()
        candidates = _generate_rejected_code_outputs(out)
        assert len(candidates) >= 1

    def test_strategy1_wrong_unit_type(self):
        out = _valid_output(unit="function")
        candidates = _generate_rejected_code_outputs(out)
        wrong_units = [c for c in candidates if c.get("unit") != "function"]
        assert len(wrong_units) >= 1

    def test_strategy2_perturbed_name(self):
        out = _valid_output(name="my_func")
        candidates = _generate_rejected_code_outputs(out)
        perturbed = [c for c in candidates if "_TYPO" in c.get("name", "")]
        assert len(perturbed) >= 1

    def test_strategy3_wrong_file(self):
        out = _valid_output(file="src/real.py")
        candidates = _generate_rejected_code_outputs(out)
        wrong_file = [c for c in candidates if c.get("file") != "src/real.py"]
        assert len(wrong_file) >= 1

    def test_strategy4_garbled_signature(self):
        out = _valid_output()
        candidates = _generate_rejected_code_outputs(out)
        garbled = [c for c in candidates if "(((" in c.get("signature", "")]
        assert len(garbled) >= 1

    def test_strategy5_dropped_signature_key(self):
        out = _valid_output()
        candidates = _generate_rejected_code_outputs(out)
        dropped = [c for c in candidates if "signature" not in c]
        assert len(dropped) >= 1

    def test_strategy6_wrong_class_when_class_present(self):
        out = _valid_method_output()
        candidates = _generate_rejected_code_outputs(out)
        wrong_class = [c for c in candidates if "TYPO" in c.get("class", "")]
        assert len(wrong_class) >= 1

    def test_strategy6_dropped_class_when_class_present(self):
        out = _valid_method_output()
        candidates = _generate_rejected_code_outputs(out)
        no_class = [c for c in candidates if "class" not in c]
        assert len(no_class) >= 1

    def test_no_class_strategies_when_no_class_key(self):
        out = _valid_output(unit="function")
        candidates = _generate_rejected_code_outputs(out)
        # Without a class key the class-swap strategies must not appear.
        # (candidates either lack class or never had it added)
        # Strategy 6 only runs when 'class' in output, so no candidates
        # should have 'class' key injected (they wouldn't have it naturally).
        for c in candidates:
            # All corruptions of a classless output should themselves be classless
            assert "class" not in c

    def test_all_candidates_differ_from_chosen(self):
        out = _valid_output()
        candidates = _generate_rejected_code_outputs(out)
        for c in candidates:
            assert c != out

    def test_at_least_one_candidate_fails_verifier(self):
        """At least one candidate must fail the verifier so gen_code_dpo can write a pair."""
        out = _valid_output()
        candidates = _generate_rejected_code_outputs(out)
        failing = [c for c in candidates if not _output_is_valid(c)]
        assert len(failing) >= 1, "Expected at least one candidate to fail the verifier"

    def test_method_at_least_one_candidate_fails_verifier(self):
        """Same check for a method (class-carrying) output."""
        out = _valid_method_output()
        candidates = _generate_rejected_code_outputs(out)
        failing = [c for c in candidates if not _output_is_valid(c)]
        assert len(failing) >= 1, "Expected at least one candidate to fail the verifier"

    def test_unit_type_cycles_correctly(self):
        # function → method, method → class, class → api_call, api_call → function
        for (original, expected_wrong) in [
            ("function", "method"),
            ("method", "class"),
            ("class", "api_call"),
            ("api_call", "function"),
        ]:
            out = _valid_output(unit=original)
            candidates = _generate_rejected_code_outputs(out)
            wrong_units = [c["unit"] for c in candidates if c.get("unit") != original]
            assert expected_wrong in wrong_units, (
                f"Expected {expected_wrong!r} in wrong-unit candidates for {original!r}"
            )

    def test_deterministic_same_input_same_output(self):
        out = _valid_output()
        first = _generate_rejected_code_outputs(out)
        second = _generate_rejected_code_outputs(out)
        assert first == second


# ── Tests: _make_code_dpo_record ────────────────────────────────────────────

class TestMakeCodeDpoRecord:
    def test_record_has_required_keys(self):
        chosen = _valid_output()
        rejected = _generate_rejected_code_outputs(chosen)[0]
        record = _make_code_dpo_record("How do I use `my_func`?", chosen, rejected)
        assert "type" in record
        assert "question" in record
        assert "chosen" in record
        assert "rejected" in record

    def test_type_is_code(self):
        chosen = _valid_output()
        rejected = _generate_rejected_code_outputs(chosen)[0]
        record = _make_code_dpo_record("q", chosen, rejected)
        assert record["type"] == "code"

    def test_question_preserved(self):
        chosen = _valid_output()
        rejected = _generate_rejected_code_outputs(chosen)[0]
        q = "How do I call `my_func`?"
        record = _make_code_dpo_record(q, chosen, rejected)
        assert record["question"] == q

    def test_chosen_and_rejected_differ(self):
        chosen = _valid_output()
        rejected = _generate_rejected_code_outputs(chosen)[0]
        record = _make_code_dpo_record("q", chosen, rejected)
        assert record["chosen"] != record["rejected"]

    def test_chosen_passes_verifier(self):
        chosen = _valid_output()
        rejected = _generate_rejected_code_outputs(chosen)[0]
        record = _make_code_dpo_record("q", chosen, rejected)
        assert _output_is_valid(record["chosen"]) is True

    def test_rejected_is_a_valid_negative(self):
        chosen = _valid_output()
        # Any surviving candidate under the keep-rule is a valid DPO negative:
        # it differs from gold AND is either malformed or field-mismatched.
        all_candidates = _generate_rejected_code_outputs(chosen)
        surviving = [c for c in all_candidates if _is_valid_negative(c, chosen)]
        assert surviving, "Need at least one surviving candidate for this test"
        rejected = surviving[0]
        record = _make_code_dpo_record("q", chosen, rejected)
        assert _is_valid_negative(record["rejected"], record["chosen"]) is True


# ── Tests: gen_code_dpo end-to-end ──────────────────────────────────────────

def _make_training_record(unit="function", name="foo", file="mod.py",
                           signature="foo(x)", cls=None) -> dict:
    """Create a minimal training.jsonl record."""
    output: dict = {"unit": unit, "name": name, "file": file, "signature": signature}
    if cls is not None:
        output["class"] = cls
    return {
        "type": "code",
        "question": f"How do I use `{name}`?",
        "thinking": "Entity: function foo",
        "output": output,
    }


class TestGenCodeDpo:
    def test_writes_pairs_to_output(self, tmp_path):
        training = tmp_path / "training.jsonl"
        output = tmp_path / "dpo.jsonl"

        record = _make_training_record()
        training.write_text(json.dumps(record) + "\n")

        count = gen_code_dpo(training, output)
        assert count >= 1
        assert output.exists()

        lines = [json.loads(l) for l in output.read_text().splitlines() if l.strip()]
        assert len(lines) == count

    def test_dry_run_writes_nothing(self, tmp_path):
        training = tmp_path / "training.jsonl"
        output = tmp_path / "dpo.jsonl"

        record = _make_training_record()
        training.write_text(json.dumps(record) + "\n")

        count = gen_code_dpo(training, output, dry_run=True)
        assert count >= 1
        assert not output.exists()

    def test_skips_record_with_invalid_chosen(self, tmp_path):
        training = tmp_path / "training.jsonl"
        output = tmp_path / "dpo.jsonl"

        # A malformed output (garbled signature) should be skipped.
        bad_output = {"unit": "function", "name": "foo", "file": "a.py",
                      "signature": "((( bad"}
        record = {
            "type": "code",
            "question": "How do I use `foo`?",
            "thinking": "...",
            "output": bad_output,
        }
        training.write_text(json.dumps(record) + "\n")

        count = gen_code_dpo(training, output)
        assert count == 0
        # Output file may not exist or may be empty
        if output.exists():
            assert output.read_text().strip() == ""

    def test_output_pair_format(self, tmp_path):
        training = tmp_path / "training.jsonl"
        output = tmp_path / "dpo.jsonl"

        record = _make_training_record(name="bar", signature="bar(a, b)")
        training.write_text(json.dumps(record) + "\n")

        gen_code_dpo(training, output)
        pair = json.loads(output.read_text().splitlines()[0])

        assert pair["type"] == "code"
        assert "question" in pair
        assert "chosen" in pair
        assert "rejected" in pair
        assert pair["chosen"] != pair["rejected"]
        assert _output_is_valid(pair["chosen"]) is True
        # Under the new keep-rule, the written rejected is a strictly-worse
        # negative: it differs from gold and is either malformed or field-mismatched.
        assert _is_valid_dpo_negative(pair) is True

    def test_multiple_records_each_get_one_pair(self, tmp_path):
        training = tmp_path / "training.jsonl"
        output = tmp_path / "dpo.jsonl"

        records = [
            _make_training_record(name="alpha", signature="alpha()"),
            _make_training_record(name="beta", signature="beta(x)"),
            _make_training_record(name="gamma", signature="gamma(x, y)"),
        ]
        training.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        count = gen_code_dpo(training, output)
        assert count == 3

    def test_non_code_records_skipped(self, tmp_path):
        training = tmp_path / "training.jsonl"
        output = tmp_path / "dpo.jsonl"

        records = [
            {"type": "api", "question": "q", "output": {"endpoint": "GET /x", "params": {}}},
            _make_training_record(name="foo", signature="foo()"),
        ]
        training.write_text("\n".join(json.dumps(r) for r in records) + "\n")

        count = gen_code_dpo(training, output)
        # Only the code record should produce a pair
        assert count == 1

    def test_question_from_input_record_preserved(self, tmp_path):
        training = tmp_path / "training.jsonl"
        output = tmp_path / "dpo.jsonl"

        q = "Show me an example of using `my_func`"
        record = _make_training_record(name="my_func", signature="my_func()")
        record["question"] = q
        training.write_text(json.dumps(record) + "\n")

        gen_code_dpo(training, output)
        pair = json.loads(output.read_text().splitlines()[0])
        assert pair["question"] == q

    def test_method_record_produces_valid_pair(self, tmp_path):
        training = tmp_path / "training.jsonl"
        output = tmp_path / "dpo.jsonl"

        record = _make_training_record(
            unit="method", name="process", file="mod.py",
            signature="process(self, data: list)", cls="Processor"
        )
        training.write_text(json.dumps(record) + "\n")

        count = gen_code_dpo(training, output)
        assert count == 1
        pair = json.loads(output.read_text().splitlines()[0])
        assert _output_is_valid(pair["chosen"]) is True
        assert _is_valid_dpo_negative(pair) is True

    def test_api_call_record_produces_valid_pair(self, tmp_path):
        training = tmp_path / "training.jsonl"
        output = tmp_path / "dpo.jsonl"

        record = _make_training_record(
            unit="api_call", name="requests.get", file="x.py",
            signature="requests.get(url, **kwargs)"
        )
        training.write_text(json.dumps(record) + "\n")

        count = gen_code_dpo(training, output)
        assert count == 1
        pair = json.loads(output.read_text().splitlines()[0])
        assert _output_is_valid(pair["chosen"]) is True
        assert _is_valid_dpo_negative(pair) is True

    def test_semantic_negative_survives(self, tmp_path):
        """A structurally-VALID but semantically-WRONG rejected must survive.

        This is the whole point of the keep-rule change: for an ordinary
        function record the first surviving candidate is the wrong-unit-type
        corruption, which is structurally valid (passes the verifier) yet
        differs from gold (field_accuracy < 1.0). It must be written as the
        negative rather than being discarded for being structurally valid.
        """
        training = tmp_path / "training.jsonl"
        output = tmp_path / "dpo.jsonl"

        record = _make_training_record(name="my_func", signature="my_func(x: int)")
        training.write_text(json.dumps(record) + "\n")

        count = gen_code_dpo(training, output)
        assert count == 1
        pair = json.loads(output.read_text().splitlines()[0])

        rejected = pair["rejected"]
        # Structurally valid (passes the verifier) ...
        assert _output_is_valid(rejected) is True
        # ... but semantically wrong (a field differs from gold).
        acc = code_field_accuracy(rejected, pair["chosen"])["field_accuracy"]
        assert acc < 1.0
        # And it satisfies the keep-rule contract.
        assert _is_valid_dpo_negative(pair) is True


# ── Tests: _is_valid_negative (keep-rule predicate) ─────────────────────────

class TestIsValidNegative:
    def test_identical_output_is_not_a_negative(self):
        chosen = _valid_output()
        assert _is_valid_negative(copy_dict(chosen), chosen) is False

    def test_malformed_candidate_is_a_negative(self):
        chosen = _valid_output()
        bad = _valid_output(signature="((( unbalanced")
        assert _output_is_valid(bad) is False
        assert _is_valid_negative(bad, chosen) is True

    def test_semantic_mismatch_is_a_negative(self):
        chosen = _valid_output(name="my_func")
        # Structurally valid but wrong name.
        bad = _valid_output(name="my_func_TYPO")
        assert _output_is_valid(bad) is True
        assert code_field_accuracy(bad, chosen)["field_accuracy"] < 1.0
        assert _is_valid_negative(bad, chosen) is True

    def test_structurally_valid_field_perfect_is_not_a_negative(self):
        """A candidate that is valid AND field-identical (but not == chosen, e.g.
        an extra ignored key) is NOT strictly worse → not a valid negative."""
        chosen = _valid_output()
        same = _valid_output()
        same["extra_ignored_key"] = "noise"  # not compared by field_accuracy
        assert same != chosen
        assert _output_is_valid(same) is True
        assert code_field_accuracy(same, chosen)["field_accuracy"] == 1.0
        assert _is_valid_negative(same, chosen) is False

    def test_every_strategy_candidate_survives_for_method(self):
        chosen = _valid_method_output()
        candidates = _generate_rejected_code_outputs(chosen)
        for c in candidates:
            assert _is_valid_negative(c, chosen) is True, (
                f"Candidate failed to survive keep-rule: {c}"
            )


def copy_dict(d: dict) -> dict:
    """Shallow copy helper for readability in tests."""
    return dict(d)
