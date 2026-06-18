"""Tests for Milestone 2.3 — docstring + surrounding-context enrichment.

Covers:
- Scanner produces real signatures, docstrings, and call_signatures
- Generator surfaces doc in linear and QOC traces
- Backward compatibility: hand-built units with no new keys unchanged
- Determinism and cross-process stability
- No-None invariant on enriched method traces
- API call units carry NO new keys
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from scripts.repo.loader import RepoConfig
from scripts.repo.scan_repo import scan_repo
from scripts.repo.generate_from_code import (
    _make_thinking,
    _make_record,
    _signature_for,
    generate_code_thinking,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "sample_repo"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _fixture_config():
    """RepoConfig pointing at the sample_repo fixture."""
    return RepoConfig(
        name="ctx",
        path=str(FIXTURES_DIR),
        include=["**/*.py"],
        exclude=[],
    )


def _unit_by_name(units, name):
    """Return the first unit dict with the given name (raises if not found)."""
    for u in units:
        if u["name"] == name:
            return u
    raise KeyError(f"No unit with name={name!r} found. Available: {[u['name'] for u in units]}")


@pytest.fixture(scope="module")
def fixture_units():
    """Scan the sample_repo fixture once for the whole module."""
    return scan_repo(_fixture_config())


# ===========================================================================
# Part A — Scanner-side assertions
# ===========================================================================

class TestScannerSignatures:
    """Real signatures, docstrings, call_signatures from fixture units."""

    def test_create_user_signature(self, fixture_units):
        """create_user function has real signature and NO doc key."""
        u = _unit_by_name(fixture_units, "create_user")
        assert u["signature"] == "create_user(name: str, email: str)"
        assert "doc" not in u, f"create_user should have no doc key, got: {u.get('doc')!r}"

    def test_slugify_doc(self, fixture_units):
        """slugify function has correct docstring summary."""
        u = _unit_by_name(fixture_units, "slugify")
        assert u.get("doc") == "Convert text to URL-safe slug."

    def test_greet_method_signature_and_call_signature(self, fixture_units):
        """greet method has full def signature (includes self) and call_signature (no self)."""
        u = _unit_by_name(fixture_units, "greet")
        assert u["type"] == "method"
        assert u["signature"] == "greet(self)"
        assert u["call_signature"] == "greet()"

    def test_from_dict_call_signature_drops_cls(self, fixture_units):
        """from_dict classmethod: call_signature drops leading cls."""
        u = _unit_by_name(fixture_units, "from_dict")
        assert u["type"] == "method"
        assert u["call_signature"] == "from_dict(data: dict)"

    def test_validate_email_static_method_call_signature(self, fixture_units):
        """validate_email staticmethod: first param (email) is NOT dropped — not self/cls."""
        u = _unit_by_name(fixture_units, "validate_email")
        assert u["type"] == "method"
        assert u["call_signature"] == "validate_email(email: str)"

    def test_user_class_signature_and_doc(self, fixture_units):
        """User class has instantiation-form signature (self dropped) and docstring."""
        u = _unit_by_name(fixture_units, "User")
        assert u["type"] == "class"
        assert u["signature"] == "User(name: str, email: str)"
        assert u["doc"] == "A user model."

    def test_class_without_init_has_no_signature(self):
        """A class with no __init__ must NOT have a 'signature' key."""
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            (repo_path / "noinit.py").write_text(
                "class NoInit:\n"
                "    \"\"\"A class with no __init__.\"\"\"\n"
                "    def method(self): pass\n"
            )
            cfg = RepoConfig(name="noinit", path=str(repo_path), include=["**/*.py"], exclude=[])
            units = scan_repo(cfg)
            cls_units = [u for u in units if u["type"] == "class" and u["name"] == "NoInit"]
            assert cls_units, "NoInit class not found in scan results"
            cls_unit = cls_units[0]
            assert "signature" not in cls_unit, (
                f"NoInit (no __init__) should NOT have a 'signature' key, got: {cls_unit.get('signature')!r}"
            )

    def test_api_call_units_carry_no_new_keys(self, fixture_units):
        """API call units must NOT carry signature, doc, or call_signature."""
        api_calls = [u for u in fixture_units if u["type"] == "api_call"]
        assert api_calls, "expected at least one api_call unit in fixture"
        for u in api_calls:
            assert "signature" not in u, (
                f"api_call {u['name']!r} unexpectedly has 'signature': {u.get('signature')!r}"
            )
            assert "doc" not in u, (
                f"api_call {u['name']!r} unexpectedly has 'doc': {u.get('doc')!r}"
            )
            assert "call_signature" not in u, (
                f"api_call {u['name']!r} unexpectedly has 'call_signature'"
            )

    def test_backward_compat_required_keys_present(self, fixture_units):
        """Every unit still carries the required base keys: type, name, file, lineno."""
        for u in fixture_units:
            for key in ("type", "name", "file", "lineno"):
                assert key in u, f"Unit {u.get('name')!r} missing required key {key!r}"

    def test_method_units_still_carry_class_key(self, fixture_units):
        """Method units still carry the 'class' key (backward compat)."""
        methods = [u for u in fixture_units if u["type"] == "method"]
        assert methods, "expected at least one method unit"
        for u in methods:
            assert "class" in u, f"Method {u['name']!r} is missing the 'class' key"

    # ------------------------------------------------------------------
    # method_kind key: static / class / absent for plain instance methods
    # ------------------------------------------------------------------

    def test_from_dict_has_method_kind_class(self, fixture_units):
        """from_dict is a @classmethod — scanner must set method_kind='class'."""
        u = _unit_by_name(fixture_units, "from_dict")
        assert u.get("method_kind") == "class", (
            f"Expected method_kind='class' for from_dict, got: {u.get('method_kind')!r}"
        )

    def test_validate_email_has_method_kind_static(self, fixture_units):
        """validate_email is a @staticmethod — scanner must set method_kind='static'."""
        u = _unit_by_name(fixture_units, "validate_email")
        assert u.get("method_kind") == "static", (
            f"Expected method_kind='static' for validate_email, got: {u.get('method_kind')!r}"
        )

    def test_greet_has_no_method_kind(self, fixture_units):
        """greet is a plain instance method — 'method_kind' must NOT be present."""
        u = _unit_by_name(fixture_units, "greet")
        assert "method_kind" not in u, (
            f"Plain instance method greet should not have 'method_kind', got: {u.get('method_kind')!r}"
        )


# ===========================================================================
# Part B — Generator-side assertions
# ===========================================================================

class TestGeneratorDocInTraces:
    """When unit.doc is set, both linear and QOC traces contain Doc: <docstring>."""

    _UNIT_WITH_DOC = {
        "type": "function",
        "name": "slugify",
        "file": "utils/helpers.py",
        "lineno": 6,
        "doc": "Convert text to URL-safe slug.",
        "signature": "slugify(text: str)",
    }
    _UNIT_WITHOUT_DOC = {
        "type": "function",
        "name": "create_user",
        "file": "core/models.py",
        "lineno": 43,
        "signature": "create_user(name: str, email: str)",
    }

    def test_linear_trace_contains_doc_when_present(self):
        trace = _make_thinking(self._UNIT_WITH_DOC, style="linear")
        assert "Doc: Convert text to URL-safe slug." in trace, (
            f"Expected 'Doc:' line in linear trace:\n{trace}"
        )

    def test_linear_trace_no_doc_line_when_absent(self):
        trace = _make_thinking(self._UNIT_WITHOUT_DOC, style="linear")
        assert "Doc:" not in trace, (
            f"Unexpected 'Doc:' line in linear trace for unit with no doc:\n{trace}"
        )

    def test_qoc_trace_contains_doc_when_present(self):
        trace = _make_thinking(self._UNIT_WITH_DOC, style="qoc")
        assert "Doc: Convert text to URL-safe slug." in trace, (
            f"Expected 'Doc:' line in QOC trace:\n{trace}"
        )

    def test_qoc_trace_no_doc_line_when_absent(self):
        trace = _make_thinking(self._UNIT_WITHOUT_DOC, style="qoc")
        assert "Doc:" not in trace, (
            f"Unexpected 'Doc:' line in QOC trace for unit with no doc:\n{trace}"
        )

    def test_doc_line_for_method_unit(self):
        """Method unit with doc emits Doc: in traces."""
        unit = {
            "type": "method",
            "name": "some_method",
            "file": "pkg/m.py",
            "lineno": 5,
            "class": "SomeClass",
            "signature": "some_method(self)",
            "call_signature": "some_method()",
            "doc": "Does something useful.",
        }
        for style in ("linear", "qoc"):
            trace = _make_thinking(unit, style=style)
            assert "Doc: Does something useful." in trace, (
                f"Expected 'Doc:' in {style} trace for method with doc:\n{trace}"
            )

    def test_doc_line_for_class_unit(self):
        """Class unit with doc emits Doc: in traces."""
        unit = {
            "type": "class",
            "name": "MyClass",
            "file": "pkg/c.py",
            "lineno": 1,
            "signature": "MyClass(x: int)",
            "doc": "A helpful class.",
        }
        for style in ("linear", "qoc"):
            trace = _make_thinking(unit, style=style)
            assert "Doc: A helpful class." in trace, (
                f"Expected 'Doc:' in {style} trace for class with doc:\n{trace}"
            )


class TestSignatureForReflectsRealSig:
    """_signature_for returns real signature when unit carries one."""

    def test_function_with_real_signature(self):
        unit = {
            "type": "function",
            "name": "create_user",
            "file": "core/models.py",
            "lineno": 43,
            "signature": "create_user(name: str, email: str)",
        }
        assert _signature_for(unit) == "create_user(name: str, email: str)"

    def test_method_with_real_signature(self):
        unit = {
            "type": "method",
            "name": "greet",
            "file": "core/models.py",
            "lineno": 13,
            "class": "User",
            "signature": "greet(self)",
            "call_signature": "greet()",
        }
        assert _signature_for(unit) == "greet(self)"

    def test_class_with_real_signature(self):
        unit = {
            "type": "class",
            "name": "User",
            "file": "core/models.py",
            "lineno": 6,
            "signature": "User(name: str, email: str)",
            "doc": "A user model.",
        }
        assert _signature_for(unit) == "User(name: str, email: str)"

    def test_output_signature_reflects_real_sig(self):
        """_make_record output.signature uses the real signature from scanner."""
        unit = {
            "type": "function",
            "name": "slugify",
            "file": "utils/helpers.py",
            "lineno": 6,
            "signature": "slugify(text: str)",
            "doc": "Convert text to URL-safe slug.",
        }
        rec = _make_record(unit)
        assert rec["output"]["signature"] == "slugify(text: str)", (
            f"Expected real signature in output, got: {rec['output']['signature']!r}"
        )


class TestMethodCallFormUsesRealCallSignature:
    """Method trace uses real call_signature (no self/cls) when present."""

    _GREET = {
        "type": "method",
        "name": "greet",
        "file": "core/models.py",
        "lineno": 13,
        "class": "User",
        "signature": "greet(self)",
        "call_signature": "greet()",
    }

    @pytest.mark.parametrize("style", ["linear", "qoc"])
    def test_method_trace_no_self_in_call_form(self, style):
        """Enriched method traces must not contain '(self' (call form uses call_signature)."""
        trace = _make_thinking(self._GREET, style=style)
        assert "(self" not in trace, (
            f"{style} enriched method trace contains '(self' — call form is wrong:\n{trace}"
        )

    @pytest.mark.parametrize("style", ["linear", "qoc"])
    def test_method_trace_no_cls_in_call_form(self, style):
        """Enriched method traces must not contain '(cls'."""
        trace = _make_thinking(self._GREET, style=style)
        assert "(cls" not in trace, (
            f"{style} enriched method trace contains '(cls':\n{trace}"
        )


class TestBackwardCompat:
    """Hand-built units with no new keys produce byte-identical traces to before."""

    _HAND_METHOD = {
        "type": "method",
        "name": "do_thing",
        "file": "pkg/m.py",
        "lineno": 10,
        "class": "Widget",
    }
    _HAND_ORPHAN = {
        "type": "method",
        "name": "do_thing",
        "file": "pkg/m.py",
        "lineno": 1,
    }

    def test_hand_method_signature_stub_form(self):
        """Hand-built method (no signature key) uses stub 'do_thing(self, ...)'."""
        assert _signature_for(self._HAND_METHOD) == "do_thing(self, ...)"

    def test_hand_method_output_signature(self):
        """Hand-built method output.signature is the def stub form with self."""
        rec = _make_record(self._HAND_METHOD)
        assert rec["output"]["signature"] == "do_thing(self, ...)", (
            f"Expected stub def form in output.signature, got: {rec['output']['signature']!r}"
        )

    @pytest.mark.parametrize("style", ["linear", "qoc"])
    def test_hand_method_call_form_no_signature_key(self, style):
        """Hand-built method (no call_signature) still yields 'do_thing(...)' call form."""
        trace = _make_thinking(self._HAND_METHOD, style=style)
        assert "do_thing(...)" in trace, (
            f"Expected stub call form 'do_thing(...)' in {style} trace:\n{trace}"
        )

    @pytest.mark.parametrize("style", ["linear", "qoc"])
    def test_hand_method_no_self_in_call_use(self, style):
        """Hand-built method traces must still NOT embed '(self' in the call form."""
        trace = _make_thinking(self._HAND_METHOD, style=style)
        assert "(self" not in trace, (
            f"Hand-built {style} trace contains '(self' in call form:\n{trace}"
        )


class TestDeterminism:
    """Enriched units produce identical traces across two calls in same process."""

    _ENRICHED_UNIT = {
        "type": "function",
        "name": "slugify",
        "file": "utils/helpers.py",
        "lineno": 6,
        "signature": "slugify(text: str)",
        "doc": "Convert text to URL-safe slug.",
    }

    @pytest.mark.parametrize("style", ["linear", "qoc"])
    def test_enriched_trace_same_process_deterministic(self, style):
        t1 = _make_thinking(self._ENRICHED_UNIT, style=style)
        t2 = _make_thinking(self._ENRICHED_UNIT, style=style)
        assert t1 == t2, f"Enriched {style} trace not deterministic"

    def test_enriched_make_record_deterministic(self):
        r1 = _make_record(self._ENRICHED_UNIT)
        r2 = _make_record(self._ENRICHED_UNIT)
        assert r1 == r2


class TestNoNoneInEnrichedTrace:
    """Enriched method traces must not contain the literal 'None'."""

    _ENRICHED_METHOD = {
        "type": "method",
        "name": "greet",
        "file": "core/models.py",
        "lineno": 13,
        "class": "User",
        "signature": "greet(self)",
        "call_signature": "greet()",
    }

    @pytest.mark.parametrize("style", ["linear", "qoc"])
    def test_enriched_method_no_none(self, style):
        trace = _make_thinking(self._ENRICHED_METHOD, style=style)
        assert "None" not in trace, (
            f"Literal 'None' found in enriched {style} method trace:\n{trace}"
        )


# ===========================================================================
# Cross-process determinism for enriched traces
# ===========================================================================

_ENRICHED_CROSS_PROCESS_SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {repo_root!r})
    from scripts.repo.generate_from_code import generate_code_thinking

    unit = {{
        "type": "function",
        "name": "slugify",
        "file": "utils/helpers.py",
        "lineno": 6,
        "signature": "slugify(text: str)",
        "doc": "Convert text to URL-safe slug.",
    }}

    traces = [
        generate_code_thinking(unit, style="linear"),
        generate_code_thinking(unit, style="qoc"),
    ]
    print(json.dumps(traces))
    """
)


def _run_enriched_cross_process(hashseed: int):
    """Run the enriched cross-process script with a specific PYTHONHASHSEED."""
    repo_root = str(_REPO_ROOT)
    script = _ENRICHED_CROSS_PROCESS_SCRIPT.format(repo_root=repo_root)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = str(hashseed)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"Enriched cross-process subprocess failed (PYTHONHASHSEED={hashseed}):\n{result.stderr}"
    )
    return json.loads(result.stdout)


def test_enriched_cross_process_deterministic():
    """Enriched traces must be identical under different PYTHONHASHSEED values."""
    traces0 = _run_enriched_cross_process(0)
    traces1 = _run_enriched_cross_process(1)
    assert traces0 == traces1, (
        "Enriched traces differ between PYTHONHASHSEED=0 and PYTHONHASHSEED=1"
    )


def test_enriched_cross_process_doc_line_present():
    """Cross-process enriched linear trace must contain 'Doc:' line."""
    traces = _run_enriched_cross_process(42)
    linear_trace = traces[0]
    assert "Doc: Convert text to URL-safe slug." in linear_trace, (
        f"'Doc:' line missing from cross-process linear trace:\n{linear_trace}"
    )


def test_enriched_cross_process_qoc_doc_line_present():
    """Cross-process enriched QOC trace must contain 'Doc:' line."""
    traces = _run_enriched_cross_process(7)
    qoc_trace = traces[1]
    assert "Doc: Convert text to URL-safe slug." in qoc_trace, (
        f"'Doc:' line missing from cross-process QOC trace:\n{qoc_trace}"
    )


# ===========================================================================
# Part C — Static/classmethod rendering tests (Milestone: trace-accuracy fix)
# ===========================================================================

# Canonical unit dicts mirroring the fixture scanner output for the two target
# methods.  The 'method_kind' key is the additive key added by the scanner.
_VALIDATE_EMAIL_UNIT = {
    "type": "method",
    "name": "validate_email",
    "file": "core/models.py",
    "lineno": 24,
    "class": "User",
    "signature": "validate_email(email: str)",
    "call_signature": "validate_email(email: str)",
    "method_kind": "static",
}

_FROM_DICT_UNIT = {
    "type": "method",
    "name": "from_dict",
    "file": "core/models.py",
    "lineno": 20,
    "class": "User",
    "signature": "from_dict(cls, data: dict)",
    "call_signature": "from_dict(data: dict)",
    "method_kind": "class",
}

_GREET_UNIT = {
    "type": "method",
    "name": "greet",
    "file": "core/models.py",
    "lineno": 13,
    "class": "User",
    "signature": "greet(self)",
    "call_signature": "greet()",
    # No 'method_kind' key — plain instance method.
}


class TestStaticmethodLinearTrace:
    """Linear trace for validate_email (@staticmethod) renders correctly."""

    def test_entity_line_says_staticmethod(self):
        trace = _make_thinking(_VALIDATE_EMAIL_UNIT, style="linear")
        assert "Entity: staticmethod validate_email on class User" in trace, (
            f"Expected 'Entity: staticmethod' in linear trace:\n{trace}"
        )

    def test_class_dot_call_form_present(self):
        trace = _make_thinking(_VALIDATE_EMAIL_UNIT, style="linear")
        assert "User.validate_email(" in trace, (
            f"Expected 'User.validate_email(' in linear trace:\n{trace}"
        )

    def test_no_instance_dot_in_use_line(self):
        trace = _make_thinking(_VALIDATE_EMAIL_UNIT, style="linear")
        assert "instance." not in trace, (
            f"Linear static trace must not contain 'instance.':\n{trace}"
        )

    def test_no_misleading_standalone_not_line(self):
        """The old 'NOT: calling ... standalone function' must not appear."""
        trace = _make_thinking(_VALIDATE_EMAIL_UNIT, style="linear")
        assert "NOT: calling validate_email() as a standalone function" not in trace, (
            f"Misleading 'standalone function' NOT-line present in static trace:\n{trace}"
        )

    def test_deterministic(self):
        t1 = _make_thinking(_VALIDATE_EMAIL_UNIT, style="linear")
        t2 = _make_thinking(_VALIDATE_EMAIL_UNIT, style="linear")
        assert t1 == t2, "Linear staticmethod trace is not deterministic"

    def test_exact_trace(self):
        """Full linear trace for validate_email must match the expected format exactly."""
        expected = (
            "Entity: staticmethod validate_email on class User\n"
            "File: core/models.py:24\n"
            "Scope: single unit — static method\n"
            "Use: User.validate_email(email: str)  # static method — no instance needed\n"
            "NOT: passing self or cls — a static method receives neither (calling via an instance is allowed but unnecessary)"
        )
        actual = _make_thinking(_VALIDATE_EMAIL_UNIT, style="linear")
        assert actual == expected, (
            f"Linear static trace does not match expected format.\n"
            f"Expected:\n{expected}\n\nActual:\n{actual}"
        )


class TestClassmethodLinearTrace:
    """Linear trace for from_dict (@classmethod) renders correctly."""

    def test_entity_line_says_classmethod(self):
        trace = _make_thinking(_FROM_DICT_UNIT, style="linear")
        assert "Entity: classmethod from_dict on class User" in trace, (
            f"Expected 'Entity: classmethod' in linear trace:\n{trace}"
        )

    def test_class_dot_call_form_present(self):
        trace = _make_thinking(_FROM_DICT_UNIT, style="linear")
        assert "User.from_dict(data: dict)" in trace, (
            f"Expected 'User.from_dict(data: dict)' in linear trace:\n{trace}"
        )

    def test_no_instance_dot_in_use_line(self):
        trace = _make_thinking(_FROM_DICT_UNIT, style="linear")
        assert "instance." not in trace, (
            f"Linear classmethod trace must not contain 'instance.':\n{trace}"
        )

    def test_cls_bound_note_present(self):
        trace = _make_thinking(_FROM_DICT_UNIT, style="linear")
        assert "cls is bound automatically" in trace or "cls" in trace, (
            f"cls-bound note missing from classmethod trace:\n{trace}"
        )

    def test_deterministic(self):
        t1 = _make_thinking(_FROM_DICT_UNIT, style="linear")
        t2 = _make_thinking(_FROM_DICT_UNIT, style="linear")
        assert t1 == t2, "Linear classmethod trace is not deterministic"

    def test_exact_trace(self):
        """Full linear trace for from_dict must match the expected format exactly."""
        expected = (
            "Entity: classmethod from_dict on class User\n"
            "File: core/models.py:20\n"
            "Scope: single unit — class method\n"
            "Use: User.from_dict(data: dict)  # classmethod — cls is bound automatically\n"
            "NOT: passing cls explicitly — Python binds it to User"
        )
        actual = _make_thinking(_FROM_DICT_UNIT, style="linear")
        assert actual == expected, (
            f"Linear classmethod trace does not match expected format.\n"
            f"Expected:\n{expected}\n\nActual:\n{actual}"
        )


class TestStaticmethodQOCTrace:
    """QOC trace for validate_email (@staticmethod) renders correctly."""

    def test_question_mentions_class_not_instance(self):
        trace = _make_thinking(_VALIDATE_EMAIL_UNIT, style="qoc")
        assert "called on the `User` class" in trace, (
            f"QOC static trace should ask about class call:\n{trace}"
        )

    def test_class_dot_call_form_present(self):
        trace = _make_thinking(_VALIDATE_EMAIL_UNIT, style="qoc")
        assert "User.validate_email(" in trace, (
            f"Expected 'User.validate_email(' in QOC trace:\n{trace}"
        )

    def test_no_instance_dot_in_trace(self):
        trace = _make_thinking(_VALIDATE_EMAIL_UNIT, style="qoc")
        assert "instance." not in trace, (
            f"QOC static trace must not contain 'instance.':\n{trace}"
        )

    def test_deterministic(self):
        t1 = _make_thinking(_VALIDATE_EMAIL_UNIT, style="qoc")
        t2 = _make_thinking(_VALIDATE_EMAIL_UNIT, style="qoc")
        assert t1 == t2, "QOC staticmethod trace is not deterministic"

    def test_exact_trace(self):
        """Full QOC trace for validate_email must match the expected format exactly."""
        expected = (
            "Question: Should `validate_email` be called on the `User` class or as a standalone function?\n"
            "Option A: call on the class — User.validate_email(email: str)  (static method — no instance needed)\n"
            "Option B: call as a bare standalone function — validate_email(...)  (incorrect — it is namespaced under User)\n"
            "Criteria: `validate_email` is a static method of `User` at core/models.py:24. It takes no self and needs no instance; call it via the class. Option A wins.\n"
            "NOT: passing self or cls to validate_email — it receives neither (an instance may call it, but need not)"
        )
        actual = _make_thinking(_VALIDATE_EMAIL_UNIT, style="qoc")
        assert actual == expected, (
            f"QOC static trace does not match expected format.\n"
            f"Expected:\n{expected}\n\nActual:\n{actual}"
        )


class TestClassmethodQOCTrace:
    """QOC trace for from_dict (@classmethod) renders correctly."""

    def test_question_mentions_class_not_instance(self):
        trace = _make_thinking(_FROM_DICT_UNIT, style="qoc")
        assert "called on the `User` class" in trace, (
            f"QOC classmethod trace should ask about class call:\n{trace}"
        )

    def test_class_dot_call_form_present(self):
        trace = _make_thinking(_FROM_DICT_UNIT, style="qoc")
        assert "User.from_dict(data: dict)" in trace, (
            f"Expected 'User.from_dict(data: dict)' in QOC trace:\n{trace}"
        )

    def test_no_instance_as_primary_option(self):
        """Option A must be the class form, not instance form."""
        trace = _make_thinking(_FROM_DICT_UNIT, style="qoc")
        assert "Option A: call on the class" in trace, (
            f"Option A should be 'call on the class' for classmethod:\n{trace}"
        )

    def test_deterministic(self):
        t1 = _make_thinking(_FROM_DICT_UNIT, style="qoc")
        t2 = _make_thinking(_FROM_DICT_UNIT, style="qoc")
        assert t1 == t2, "QOC classmethod trace is not deterministic"

    def test_exact_trace(self):
        """Full QOC trace for from_dict must match the expected format exactly."""
        expected = (
            "Question: Should `from_dict` be called on the `User` class or on an instance?\n"
            "Option A: call on the class — User.from_dict(data: dict)  (classmethod — cls is bound automatically)\n"
            "Option B: call on an instance — instance.from_dict(data: dict)  (also valid, but the class form is idiomatic)\n"
            "Criteria: `from_dict` is a classmethod of `User` at core/models.py:20. cls is bound automatically; no instance is required. Option A wins.\n"
            "NOT: passing cls explicitly — Python binds it to User"
        )
        actual = _make_thinking(_FROM_DICT_UNIT, style="qoc")
        assert actual == expected, (
            f"QOC classmethod trace does not match expected format.\n"
            f"Expected:\n{expected}\n\nActual:\n{actual}"
        )


class TestInstanceMethodRegressionAfterFix:
    """Plain instance method greet must still render with instance-call framing."""

    def test_linear_greet_still_uses_instance_framing(self):
        trace = _make_thinking(_GREET_UNIT, style="linear")
        assert "instance.greet()" in trace, (
            f"greet instance-method trace must still use 'instance.greet()':\n{trace}"
        )

    def test_linear_greet_entity_says_method(self):
        trace = _make_thinking(_GREET_UNIT, style="linear")
        assert "Entity: method greet on class User" in trace, (
            f"greet entity line must say 'method' (not staticmethod/classmethod):\n{trace}"
        )

    def test_qoc_greet_still_uses_instance_framing(self):
        trace = _make_thinking(_GREET_UNIT, style="qoc")
        assert "instance.greet()" in trace, (
            f"greet QOC trace must still use 'instance.greet()':\n{trace}"
        )

    def test_greet_linear_no_method_kind_key(self):
        """greet unit has no method_kind key — must not leak 'static' or 'class' text."""
        assert "method_kind" not in _GREET_UNIT
        trace = _make_thinking(_GREET_UNIT, style="linear")
        assert "staticmethod" not in trace
        assert "classmethod" not in trace
