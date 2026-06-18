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
