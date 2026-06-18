"""Unit tests for scripts/eval.py — 3-tier evaluation rubric."""

import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from eval import (
    format_score, param_f1, score_record, _band,
    code_format_score, code_field_accuracy, code_signature_valid, score_code_record,
)


class TestFormatScore:
    def test_valid_single_step(self):
        assert format_score({"endpoint": "GET /v1/items", "params": {}}) == 1.0

    def test_valid_with_params(self):
        assert format_score({"endpoint": "GET /v1/items", "params": {"pageSize": 10}}) == 1.0

    def test_valid_chained(self):
        api_call = {"steps": [
            {"endpoint": "GET /v1/items", "params": {}},
            {"endpoint": "GET /v1/items/{id}", "params": {"id": "{{steps.0.id}}"}},
        ]}
        assert format_score(api_call) == 1.0

    def test_missing_endpoint(self):
        assert format_score({"params": {}}) == 0.0

    def test_missing_params(self):
        assert format_score({"endpoint": "GET /v1/items"}) == 0.0

    def test_not_a_dict(self):
        assert format_score("not a dict") == 0.0
        assert format_score(None) == 0.0
        assert format_score([]) == 0.0

    def test_params_not_dict(self):
        assert format_score({"endpoint": "GET /v1/items", "params": "oops"}) == 0.0

    def test_chained_empty_steps(self):
        assert format_score({"steps": []}) == 0.0

    def test_chained_step_missing_endpoint(self):
        assert format_score({"steps": [{"params": {}}]}) == 0.0


class TestParamF1:
    def test_exact_match(self):
        assert param_f1(
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10, "networkId": 5}},
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10, "networkId": 5}},
        ) == 1.0

    def test_no_params_both(self):
        assert param_f1(
            {"endpoint": "GET /v1/me", "params": {}},
            {"endpoint": "GET /v1/me", "params": {}},
        ) == 1.0

    def test_partial_overlap(self):
        # predicted: {pageSize}, expected: {pageSize, networkId}
        # precision = 1/1, recall = 1/2, F1 = 2/3
        score = param_f1(
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10}},
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10, "networkId": 5}},
        )
        assert abs(score - 2/3) < 0.001

    def test_no_overlap(self):
        score = param_f1(
            {"endpoint": "GET /v1/items", "params": {"x": 1}},
            {"endpoint": "GET /v1/items", "params": {"y": 2}},
        )
        assert score == 0.0

    def test_extra_params_predicted(self):
        # predicted has extra param: precision < 1
        score = param_f1(
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10, "extra": 1}},
            {"endpoint": "GET /v1/items", "params": {"pageSize": 10}},
        )
        assert 0 < score < 1.0

    def test_chained_union_of_steps(self):
        predicted = {"steps": [
            {"endpoint": "GET /v1/items", "params": {}},
            {"endpoint": "GET /v1/items/{id}", "params": {"id": 1}},
        ]}
        expected = {"steps": [
            {"endpoint": "GET /v1/items", "params": {}},
            {"endpoint": "GET /v1/items/{id}", "params": {"id": 1}},
        ]}
        assert param_f1(predicted, expected) == 1.0


class TestScoreRecord:
    def test_perfect_score(self):
        api_call = {"endpoint": "GET /v1/items", "params": {"pageSize": 10}}
        result = score_record(api_call, api_call)
        assert result["format_score"] == 1.0
        assert result["param_f1"] == 1.0
        assert result["composite_score"] == 1.0
        assert result["band"] == "GOLD"

    def test_format_fail_short_circuits(self):
        result = score_record(None, {"endpoint": "GET /v1/items", "params": {}})
        assert result["format_score"] == 0.0
        assert result["param_f1"] == 0.0
        assert result["band"] == "FAIL"

    def test_no_executable_by_default(self):
        api_call = {"endpoint": "GET /v1/items", "params": {}}
        result = score_record(api_call, api_call)
        assert result["executable"] is None


class TestBand:
    def test_gold(self):
        assert _band(0.95) == "GOLD"
        assert _band(1.0) == "GOLD"

    def test_silver(self):
        assert _band(0.75) == "SILVER"

    def test_bronze(self):
        assert _band(0.5) == "BRONZE"

    def test_fail(self):
        assert _band(0.3) == "FAIL"
        assert _band(0.0) == "FAIL"


# ── Phase-3 code-unit evaluation tests ────────────────────────────────────

_VALID_OUTPUT = {
    "unit": "function",
    "name": "_canon",
    "file": "core/utils.py",
    "signature": "_canon(obj: Any)",
}

_VALID_METHOD_OUTPUT = {
    "unit": "method",
    "name": "sign",
    "file": "core/auth.py",
    "signature": "sign(self, payload: dict) -> str",
    "class": "AuthClient",
}


class TestCodeFormatScore:
    def test_valid_function(self):
        assert code_format_score(_VALID_OUTPUT) == 1.0

    def test_valid_method_with_class(self):
        assert code_format_score(_VALID_METHOD_OUTPUT) == 1.0

    def test_valid_unit_types(self):
        base = dict(_VALID_OUTPUT)
        for unit_type in ("function", "method", "class", "api_call"):
            base["unit"] = unit_type
            assert code_format_score(base) == 1.0

    def test_missing_unit_key(self):
        output = {k: v for k, v in _VALID_OUTPUT.items() if k != "unit"}
        assert code_format_score(output) == 0.0

    def test_missing_name_key(self):
        output = {k: v for k, v in _VALID_OUTPUT.items() if k != "name"}
        assert code_format_score(output) == 0.0

    def test_missing_file_key(self):
        output = {k: v for k, v in _VALID_OUTPUT.items() if k != "file"}
        assert code_format_score(output) == 0.0

    def test_missing_signature_key(self):
        output = {k: v for k, v in _VALID_OUTPUT.items() if k != "signature"}
        assert code_format_score(output) == 0.0

    def test_bad_unit_value(self):
        output = {**_VALID_OUTPUT, "unit": "variable"}
        assert code_format_score(output) == 0.0

    def test_empty_name_string(self):
        output = {**_VALID_OUTPUT, "name": ""}
        assert code_format_score(output) == 0.0

    def test_empty_file_string(self):
        output = {**_VALID_OUTPUT, "file": ""}
        assert code_format_score(output) == 0.0

    def test_empty_signature_string(self):
        output = {**_VALID_OUTPUT, "signature": ""}
        assert code_format_score(output) == 0.0

    def test_name_not_string(self):
        output = {**_VALID_OUTPUT, "name": 42}
        assert code_format_score(output) == 0.0

    def test_file_not_string(self):
        output = {**_VALID_OUTPUT, "file": None}
        assert code_format_score(output) == 0.0

    def test_signature_not_string(self):
        output = {**_VALID_OUTPUT, "signature": ["bad"]}
        assert code_format_score(output) == 0.0

    def test_class_present_non_string(self):
        output = {**_VALID_OUTPUT, "class": 123}
        assert code_format_score(output) == 0.0

    def test_class_present_string(self):
        output = {**_VALID_OUTPUT, "class": "MyClass"}
        assert code_format_score(output) == 1.0

    def test_non_dict_string(self):
        assert code_format_score("not a dict") == 0.0

    def test_non_dict_none(self):
        assert code_format_score(None) == 0.0

    def test_non_dict_list(self):
        assert code_format_score([]) == 0.0


class TestCodeFieldAccuracy:
    def test_exact_full_match_returns_1(self):
        result = code_field_accuracy(_VALID_OUTPUT, _VALID_OUTPUT)
        assert result["field_accuracy"] == 1.0
        assert result["unit_match"] is True
        assert result["name_match"] is True
        assert result["file_match"] is True
        assert result["signature_match"] is True

    def test_exact_full_match_method_with_class(self):
        result = code_field_accuracy(_VALID_METHOD_OUTPUT, _VALID_METHOD_OUTPUT)
        assert result["field_accuracy"] == 1.0
        assert result["class_match"] is True

    def test_single_field_mismatch_4_fields(self):
        predicted = {**_VALID_OUTPUT, "name": "other_func"}
        result = code_field_accuracy(predicted, _VALID_OUTPUT)
        # 3 match out of 4
        assert result["field_accuracy"] == round(3 / 4, 4)
        assert result["name_match"] is False
        assert result["unit_match"] is True
        assert result["file_match"] is True
        assert result["signature_match"] is True

    def test_all_fields_mismatch(self):
        predicted = {
            "unit": "class",
            "name": "Other",
            "file": "other.py",
            "signature": "Other()",
        }
        result = code_field_accuracy(predicted, _VALID_OUTPUT)
        assert result["field_accuracy"] == 0.0
        assert all(not result[f"{k}_match"] for k in ("unit", "name", "file", "signature"))

    def test_class_not_compared_when_expected_lacks_it(self):
        # _VALID_OUTPUT has no 'class' key; class_match should NOT be in result.
        result = code_field_accuracy(_VALID_OUTPUT, _VALID_OUTPUT)
        assert "class_match" not in result

    def test_class_compared_when_expected_has_it(self):
        result = code_field_accuracy(_VALID_METHOD_OUTPUT, _VALID_METHOD_OUTPUT)
        assert "class_match" in result

    def test_class_mismatch_reduces_accuracy(self):
        predicted = {**_VALID_METHOD_OUTPUT, "class": "WrongClass"}
        result = code_field_accuracy(predicted, _VALID_METHOD_OUTPUT)
        # 4 match out of 5
        assert result["field_accuracy"] == round(4 / 5, 4)
        assert result["class_match"] is False

    def test_predicted_not_dict_returns_zero(self):
        result = code_field_accuracy(None, _VALID_OUTPUT)
        assert result["field_accuracy"] == 0.0
        assert result["unit_match"] is False
        assert result["name_match"] is False
        assert result["file_match"] is False
        assert result["signature_match"] is False

    def test_predicted_not_dict_with_class_expected(self):
        result = code_field_accuracy("bad", _VALID_METHOD_OUTPUT)
        assert result["field_accuracy"] == 0.0
        assert result["class_match"] is False

    def test_field_accuracy_rounded_to_4(self):
        # 3/4 = 0.75, already clean, but use 2/3 via method-output with class mismatch
        # to confirm rounding: 4/5 = 0.8 — just verify rounding is 4 places
        predicted = {**_VALID_METHOD_OUTPUT, "class": "WrongClass"}
        result = code_field_accuracy(predicted, _VALID_METHOD_OUTPUT)
        assert isinstance(result["field_accuracy"], float)
        assert result["field_accuracy"] == round(result["field_accuracy"], 4)


class TestCodeSignatureValid:
    def test_well_formed_simple(self):
        output = {**_VALID_OUTPUT, "signature": "_canon(obj)"}
        assert code_signature_valid(output) is True

    def test_well_formed_with_annotation(self):
        output = {**_VALID_OUTPUT, "signature": "_canon(obj: Any)"}
        assert code_signature_valid(output) is True

    def test_well_formed_annotated_signature(self):
        # Annotated signature with a typed parameter
        output = {**_VALID_OUTPUT, "signature": "_canon(obj: Any)"}
        assert code_signature_valid(output) is True

    def test_well_formed_method_signature(self):
        output = {**_VALID_OUTPUT, "signature": "sign(self, payload: dict) -> str"}
        assert code_signature_valid(output) is True

    def test_malformed_signature(self):
        output = {**_VALID_OUTPUT, "signature": "f(:::)"}
        assert code_signature_valid(output) is False

    def test_malformed_gibberish(self):
        output = {**_VALID_OUTPUT, "signature": "not(valid(syntax"}
        assert code_signature_valid(output) is False

    def test_non_dict_input(self):
        assert code_signature_valid("not a dict") is None

    def test_missing_signature_key(self):
        output = {k: v for k, v in _VALID_OUTPUT.items() if k != "signature"}
        assert code_signature_valid(output) is None

    def test_empty_signature(self):
        output = {**_VALID_OUTPUT, "signature": ""}
        assert code_signature_valid(output) is None

    def test_api_call_dotted_call_form_is_valid(self):
        # api_call signatures are call-site forms (dotted receiver + call args),
        # not def signatures. Validated as an expression, not wrap-as-def.
        output = {
            "unit": "api_call",
            "name": "requests.get",
            "file": "x.py",
            "signature": "requests.get(url, **kwargs)",
        }
        assert code_signature_valid(output) is True

    def test_api_call_plain_call_form_is_valid(self):
        output = {
            "unit": "api_call",
            "name": "post",
            "file": "x.py",
            "signature": "post(url, data=None)",
        }
        assert code_signature_valid(output) is True

    def test_api_call_malformed_is_invalid(self):
        output = {
            "unit": "api_call",
            "name": "x",
            "file": "x.py",
            "signature": "requests.get(",
        }
        assert code_signature_valid(output) is False

    def test_non_api_call_dotted_signature_is_invalid(self):
        # A dotted call form on a non-api_call unit is NOT a valid def fragment.
        output = {**_VALID_OUTPUT, "signature": "a.b(x)"}
        assert code_signature_valid(output) is False


class TestScoreCodeRecord:
    def test_perfect_self_match_no_sig_check(self):
        result = score_code_record(_VALID_OUTPUT, _VALID_OUTPUT)
        assert result["format_score"] == 1.0
        assert result["field_accuracy"] == 1.0
        assert result["composite_score"] == 1.0
        assert result["band"] == "GOLD"
        assert result["signature_valid"] is None

    def test_perfect_self_match_with_sig_check(self):
        result = score_code_record(_VALID_OUTPUT, _VALID_OUTPUT, check_signature=True)
        assert result["format_score"] == 1.0
        assert result["field_accuracy"] == 1.0
        assert result["signature_valid"] is True
        # composite = (1 + 1 + 1) / 3 = 1.0
        assert result["composite_score"] == 1.0
        assert result["band"] == "GOLD"

    def test_method_self_match_with_class(self):
        result = score_code_record(_VALID_METHOD_OUTPUT, _VALID_METHOD_OUTPUT)
        assert result["format_score"] == 1.0
        assert result["field_accuracy"] == 1.0
        assert result["class_match"] is True

    def test_format_fail_short_circuits(self):
        # Passing None as predicted — format fails
        result = score_code_record(None, _VALID_OUTPUT)
        assert result["format_score"] == 0.0
        assert result["field_accuracy"] == 0.0
        assert result["band"] == "FAIL"
        assert result["signature_valid"] is None

    def test_format_fail_with_sig_check(self):
        result = score_code_record(None, _VALID_OUTPUT, check_signature=True)
        assert result["format_score"] == 0.0
        assert result["signature_valid"] is False

    def test_partial_field_match(self):
        predicted = {**_VALID_OUTPUT, "name": "wrong_name"}
        result = score_code_record(predicted, _VALID_OUTPUT)
        assert result["format_score"] == 1.0
        assert result["name_match"] is False
        # 3/4 field accuracy
        assert result["field_accuracy"] == round(3 / 4, 4)
        # composite = (1.0 + 0.75) / 2 = 0.875
        assert result["composite_score"] == round((1.0 + 0.75) / 2, 4)
        assert result["band"] == "SILVER"

    def test_composite_with_three_tiers(self):
        # Perfect match + check_signature → (1 + 1 + 1) / 3 = 1.0
        result = score_code_record(_VALID_OUTPUT, _VALID_OUTPUT, check_signature=True)
        assert result["composite_score"] == 1.0

    def test_composite_with_failing_signature(self):
        predicted = {**_VALID_OUTPUT, "signature": "f(:::)"}
        result = score_code_record(predicted, _VALID_OUTPUT, check_signature=True)
        # format = 1.0, field_accuracy = 3/4 = 0.75 (signature mismatch), sig = False
        assert result["signature_valid"] is False
        expected_comp = round((1.0 + round(3 / 4, 4) + 0.0) / 3, 4)
        assert result["composite_score"] == expected_comp

    def test_api_call_self_match_with_sig_check_is_gold(self):
        # Regression: api_call call-site signatures must validate under
        # --check-signature, not drop the record to BRONZE.
        api = {
            "unit": "api_call",
            "name": "requests.get",
            "file": "x.py",
            "signature": "requests.get(url, **kwargs)",
        }
        result = score_code_record(api, api, check_signature=True)
        assert result["signature_valid"] is True
        assert result["composite_score"] == 1.0
        assert result["band"] == "GOLD"

    def test_return_dict_keys(self):
        result = score_code_record(_VALID_OUTPUT, _VALID_OUTPUT)
        assert "format_score" in result
        assert "field_accuracy" in result
        assert "composite_score" in result
        assert "band" in result
        assert "signature_valid" in result


# ── Self-consistency: scoring a record against itself yields perfect scores ──

# Representative code-unit outputs (one per unit type), generic and hermetic —
# no external generated data required. Mirrors the shape scan_repo/generate emit.
_SELF_CONSISTENCY_OUTPUTS = [
    _VALID_OUTPUT,
    _VALID_METHOD_OUTPUT,
    {"unit": "class", "name": "SessionManager", "file": "core/auth.py",
     "signature": "SessionManager()"},
    {"unit": "api_call", "name": "requests.get", "file": "api/client.py",
     "signature": "requests.get(url, timeout=30)"},
]


class TestSelfConsistency:
    """Scoring each representative code record against itself yields perfect scores."""

    def test_self_consistency_format_and_field_accuracy(self):
        for output in _SELF_CONSISTENCY_OUTPUTS:
            result = score_code_record(output, output)
            assert result["format_score"] == 1.0, f"format fail for: {output}"
            assert result["field_accuracy"] == 1.0, f"field_accuracy fail for: {output}"

    def test_self_consistency_with_signature_check(self):
        for output in _SELF_CONSISTENCY_OUTPUTS:
            result = score_code_record(output, output, check_signature=True)
            assert result["format_score"] == 1.0
            assert result["field_accuracy"] == 1.0
            # Generated signatures are well-formed (AST-parseable).
            assert result["signature_valid"] is not False, (
                f"unexpected malformed signature: {output.get('signature')}"
            )
