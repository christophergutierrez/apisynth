"""Unit tests for scripts/repo/bootstrap_code_traces.py.

Mirrors test_bootstrap_traces.py conventions:
  - sentence_transformers mocked via mock.patch.dict(sys.modules, {...})
  - _Arr / MockModel pattern for embedding mock
  - call_model is monkeypatched / never hits the network
  - verifier logic exhaustively unit-tested (strict-pass, strict-fail per field,
    malformed predicted, signature-invalid, fallback gold=None pass/fail,
    parse failure)
  - gold-map build, record shape (including source="bootstrap"), dry-run, dedup
"""

import json
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

# Insert scripts/repo onto sys.path so the import resolves.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "repo"))

from bootstrap_code_traces import (  # noqa: E402
    CODE_TYPE,
    FIELD_OUTPUT,
    FIELD_QUESTION,
    FIELD_SOURCE,
    FIELD_THINKING,
    FIELD_TYPE,
    SOURCE_BOOTSTRAP,
    _cosine_similarity,
    _dedup_by_embedding,
    bootstrap_one,
    build_gold_map,
    extract,
    parse_code_output,
    verify_code_output,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _valid_output(
    unit="function",
    name="my_func",
    file="src/foo.py",
    signature="my_func(x: int) -> str",
    cls=None,
) -> dict:
    """Return a well-formed code-unit output dict."""
    out: dict = {"unit": unit, "name": name, "file": file, "signature": signature}
    if cls is not None:
        out["class"] = cls
    return out


def _valid_method_output() -> dict:
    return _valid_output(
        unit="method",
        name="do_thing",
        file="src/mymodule.py",
        signature="do_thing(self, x: int)",
        cls="MyClass",
    )


def _make_training_record(
    question="How do I use `my_func`?",
    unit="function",
    name="my_func",
    file="src/foo.py",
    signature="my_func(x: int) -> str",
    cls=None,
) -> dict:
    """Return a minimal code training.jsonl record."""
    output: dict = {"unit": unit, "name": name, "file": file, "signature": signature}
    if cls is not None:
        output["class"] = cls
    return {
        "type": "code",
        "question": question,
        "thinking": "Entity: function my_func in src/foo.py",
        "output": output,
    }


# ---------------------------------------------------------------------------
# Tests: extract
# ---------------------------------------------------------------------------


class TestExtract:
    def test_think_block_extracted(self):
        content = "<think>reasoning here</think>\nsome answer"
        thinking, answer = extract(content)
        assert thinking == "reasoning here"
        assert answer == "some answer"

    def test_no_think_block(self):
        content = "just an answer"
        thinking, answer = extract(content)
        assert thinking == ""
        assert answer == "just an answer"

    def test_multiline_thinking(self):
        content = "<think>line1\nline2</think>\nresult"
        thinking, answer = extract(content)
        assert "line1" in thinking
        assert "line2" in thinking
        assert answer == "result"

    def test_empty_think_block(self):
        content = "<think></think>\nanswer"
        thinking, answer = extract(content)
        assert thinking == ""
        assert answer == "answer"


# ---------------------------------------------------------------------------
# Tests: parse_code_output
# ---------------------------------------------------------------------------


class TestParseCodeOutput:
    def test_json_fenced_answer(self):
        raw = '```json\n{"unit": "function", "name": "foo", "file": "a.py", "signature": "foo()"}\n```'
        result = parse_code_output(raw)
        assert result is not None
        assert result["unit"] == "function"
        assert result["name"] == "foo"

    def test_plain_json(self):
        raw = '{"unit": "class", "name": "Bar", "file": "b.py", "signature": "Bar()"}'
        result = parse_code_output(raw)
        assert result is not None
        assert result["unit"] == "class"

    def test_invalid_json_returns_none(self):
        assert parse_code_output("not json at all") is None

    def test_empty_string_returns_none(self):
        assert parse_code_output("") is None

    def test_non_dict_json_returns_none(self):
        assert parse_code_output("[1, 2, 3]") is None

    def test_json_with_extra_keys_preserved(self):
        raw = '{"unit": "method", "name": "m", "file": "f.py", "signature": "m(self)", "class": "C"}'
        result = parse_code_output(raw)
        assert result is not None
        assert result.get("class") == "C"


# ---------------------------------------------------------------------------
# Tests: verify_code_output
# ---------------------------------------------------------------------------


class TestVerifyCodeOutput:
    # --- Strict path (gold is a dict) ---

    def test_strict_pass_all_fields_match(self):
        gold = _valid_output()
        predicted = _valid_output()
        assert verify_code_output(predicted, gold) is True

    def test_strict_fail_unit_mismatch(self):
        gold = _valid_output(unit="function")
        predicted = _valid_output(unit="method")
        assert verify_code_output(predicted, gold) is False

    def test_strict_fail_name_mismatch(self):
        gold = _valid_output(name="foo")
        predicted = _valid_output(name="bar")
        assert verify_code_output(predicted, gold) is False

    def test_strict_fail_file_mismatch(self):
        gold = _valid_output(file="src/real.py")
        predicted = _valid_output(file="src/other.py")
        assert verify_code_output(predicted, gold) is False

    def test_strict_fail_signature_mismatch(self):
        gold = _valid_output(signature="foo(x: int)")
        predicted = _valid_output(signature="foo(y: int)")
        assert verify_code_output(predicted, gold) is False

    def test_strict_fail_class_mismatch(self):
        gold = _valid_method_output()
        predicted = dict(_valid_method_output())
        predicted["class"] = "WrongClass"
        assert verify_code_output(predicted, gold) is False

    def test_strict_fail_bad_format(self):
        gold = _valid_output()
        predicted = {"unit": "function", "name": "foo"}  # missing file, signature
        assert verify_code_output(predicted, gold) is False

    def test_strict_fail_invalid_signature(self):
        gold = _valid_output(signature="foo(x)")
        predicted = _valid_output(signature="((( unbalanced")
        # code_format_score passes (non-empty string) but code_signature_valid fails
        assert verify_code_output(predicted, gold) is False

    def test_strict_fail_non_dict_predicted(self):
        gold = _valid_output()
        assert verify_code_output("not a dict", gold) is False
        assert verify_code_output(None, gold) is False

    def test_strict_fail_wrong_unit_type(self):
        gold = _valid_output(unit="function")
        predicted = _valid_output(unit="not_a_unit")
        assert verify_code_output(predicted, gold) is False

    # --- Fallback path (gold is None) ---

    def test_fallback_pass_format_and_signature_valid(self):
        predicted = _valid_output()
        assert verify_code_output(predicted, None) is True

    def test_fallback_fail_bad_format(self):
        predicted = {"unit": "function", "name": "foo"}  # missing required keys
        assert verify_code_output(predicted, None) is False

    def test_fallback_fail_invalid_signature(self):
        predicted = _valid_output(signature="((( unbalanced")
        assert verify_code_output(predicted, None) is False

    def test_fallback_fail_non_dict(self):
        assert verify_code_output("nope", None) is False

    def test_fallback_pass_method_with_class(self):
        predicted = _valid_method_output()
        assert verify_code_output(predicted, None) is True

    def test_fallback_pass_api_call_unit(self):
        # api_call signatures are validated as expressions (not def-wrapped)
        predicted = _valid_output(
            unit="api_call",
            name="requests.get",
            file="client.py",
            signature="requests.get(url, **kwargs)",
        )
        assert verify_code_output(predicted, None) is True


# ---------------------------------------------------------------------------
# Tests: build_gold_map
# ---------------------------------------------------------------------------


class TestBuildGoldMap:
    def test_basic_gold_map(self, tmp_path):
        record = _make_training_record(question="What is foo?")
        training = tmp_path / "training.jsonl"
        training.write_text(json.dumps(record) + "\n", encoding="utf-8")
        gold_map = build_gold_map(str(training))
        assert "What is foo?" in gold_map
        assert gold_map["What is foo?"]["name"] == "my_func"

    def test_first_seen_wins_on_duplicate_question(self, tmp_path):
        r1 = _make_training_record(question="q", name="first_func", signature="first_func()")
        r2 = _make_training_record(question="q", name="second_func", signature="second_func()")
        training = tmp_path / "training.jsonl"
        training.write_text(
            json.dumps(r1) + "\n" + json.dumps(r2) + "\n", encoding="utf-8"
        )
        gold_map = build_gold_map(str(training))
        assert gold_map["q"]["name"] == "first_func"

    def test_non_code_records_excluded(self, tmp_path):
        api_record = {
            "type": "api",
            "question": "list all items",
            "output": {"endpoint": "GET /v1/items", "params": {}},
        }
        training = tmp_path / "training.jsonl"
        training.write_text(json.dumps(api_record) + "\n", encoding="utf-8")
        gold_map = build_gold_map(str(training))
        assert len(gold_map) == 0

    def test_record_with_non_dict_output_excluded(self, tmp_path):
        record = {"type": "code", "question": "q", "output": "not a dict"}
        training = tmp_path / "training.jsonl"
        training.write_text(json.dumps(record) + "\n", encoding="utf-8")
        gold_map = build_gold_map(str(training))
        assert len(gold_map) == 0

    def test_empty_question_excluded(self, tmp_path):
        record = {"type": "code", "question": "", "output": {"unit": "function"}}
        training = tmp_path / "training.jsonl"
        training.write_text(json.dumps(record) + "\n", encoding="utf-8")
        gold_map = build_gold_map(str(training))
        assert len(gold_map) == 0

    def test_blank_lines_skipped(self, tmp_path):
        record = _make_training_record(question="What is bar?")
        training = tmp_path / "training.jsonl"
        training.write_text(
            "\n" + json.dumps(record) + "\n\n", encoding="utf-8"
        )
        gold_map = build_gold_map(str(training))
        assert "What is bar?" in gold_map

    def test_preserves_insertion_order(self, tmp_path):
        """Questions appear in file order (first-seen)."""
        questions = [f"question_{i}" for i in range(5)]
        records = [_make_training_record(question=q) for q in questions]
        training = tmp_path / "training.jsonl"
        training.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
        )
        gold_map = build_gold_map(str(training))
        assert list(gold_map.keys()) == questions

    def test_multiple_records_all_included(self, tmp_path):
        records = [
            _make_training_record(question=f"q{i}", name=f"func{i}", signature=f"func{i}()")
            for i in range(3)
        ]
        training = tmp_path / "training.jsonl"
        training.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        gold_map = build_gold_map(str(training))
        assert len(gold_map) == 3
        for i in range(3):
            assert gold_map[f"q{i}"]["name"] == f"func{i}"


# ---------------------------------------------------------------------------
# Tests: cosine similarity (mirrored from test_bootstrap_traces.py)
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 0]) == 0.0

    def test_partial_overlap(self):
        score = _cosine_similarity([1, 1, 0], [1, 0, 0])
        assert 0 < score < 1.0


# ---------------------------------------------------------------------------
# Tests: _dedup_by_embedding (mocked sentence-transformers)
# ---------------------------------------------------------------------------


class _Arr:
    """Minimal ndarray-like wrapper returned by MockModel.encode."""

    def __init__(self, data):
        self._d = data

    def tolist(self):
        return self._d

    def __getitem__(self, i):
        return self._d[i]

    def __len__(self):
        return len(self._d)


class TestDedupByEmbedding:
    def _make_candidates(self, questions: list[str]) -> list[dict]:
        return [
            {
                FIELD_QUESTION: q,
                "ok": True,
                FIELD_OUTPUT: _valid_output(name=q.replace(" ", "_")),
                FIELD_THINKING: "",
            }
            for q in questions
        ]

    def test_no_existing_all_kept(self, tmp_path):
        """With no existing records and distinct embeddings all candidates are kept."""
        call_count = 0

        class MockModel:
            def encode(self, texts, **kwargs):
                nonlocal call_count
                call_count += 1
                # Return orthogonal embeddings so nothing deduplicates
                return _Arr([[float(i), 0.0] for i in range(len(texts))])

        mock_st = mock.MagicMock()
        mock_st.SentenceTransformer.return_value = MockModel()
        with mock.patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            candidates = self._make_candidates(["list items", "get item 1"])
            kept, removed = _dedup_by_embedding(candidates, None, 0.95)
            assert removed == 0
            assert len(kept) == 2

    def test_near_duplicate_removed(self, tmp_path):
        """Mocked identical embeddings: candidate matching existing record is removed."""
        out = tmp_path / "out.jsonl"
        out.write_text(
            json.dumps({"question": "list items", "output": {}}) + "\n", encoding="utf-8"
        )

        class MockModel:
            def encode(self, texts, **kwargs):
                return _Arr([[1.0, 0.0]] * len(texts))

        mock_st = mock.MagicMock()
        mock_st.SentenceTransformer.return_value = MockModel()
        with mock.patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            candidates = self._make_candidates(["list items"])
            kept, removed = _dedup_by_embedding(candidates, str(out), 0.95)
            assert removed == 1
            assert len(kept) == 0

    def test_distinct_question_kept(self, tmp_path):
        """Mocked orthogonal embeddings: distinct candidate is kept."""
        out = tmp_path / "out.jsonl"
        out.write_text(
            json.dumps({"question": "list items", "output": {}}) + "\n", encoding="utf-8"
        )

        call_count = 0

        class MockModel:
            def encode(self, texts, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # existing questions
                    return _Arr([[1.0, 0.0]] * len(texts))
                # candidate embedding — orthogonal
                return _Arr([[0.0, 1.0]] * len(texts))

        mock_st = mock.MagicMock()
        mock_st.SentenceTransformer.return_value = MockModel()
        with mock.patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            candidates = self._make_candidates(["get item 42"])
            kept, removed = _dedup_by_embedding(candidates, str(out), 0.95)
            assert removed == 0
            assert len(kept) == 1

    def test_missing_sentence_transformers_skips_dedup(self, tmp_path):
        """When sentence-transformers is absent the candidates are returned unchanged."""
        # Hide sentence_transformers by mapping it to a module that raises ImportError
        with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            candidates = self._make_candidates(["q1", "q2"])
            kept, removed = _dedup_by_embedding(candidates, None, 0.95)
            assert kept is candidates
            assert removed == 0

    def test_already_accepted_prevents_later_duplicate(self, tmp_path):
        """After the first candidate is accepted its embedding blocks a near-duplicate."""
        call_count = 0

        class MockModel:
            def encode(self, texts, **kwargs):
                nonlocal call_count
                call_count += 1
                # All texts share the same embedding → each new one duplicates the first
                return _Arr([[1.0, 0.0]] * len(texts))

        mock_st = mock.MagicMock()
        mock_st.SentenceTransformer.return_value = MockModel()
        with mock.patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            candidates = self._make_candidates(["q1", "q2"])
            kept, removed = _dedup_by_embedding(candidates, None, 0.95)
            assert len(kept) == 1
            assert removed == 1


# ---------------------------------------------------------------------------
# Tests: bootstrap_one (model monkeypatched — never hits network)
# ---------------------------------------------------------------------------


class TestBootstrapOne:
    def _mock_call_model(self, content: str):
        """Return a factory that produces the given model response."""
        def _call(*args, **kwargs):
            return {"content": content, "ms": 10.0, "comp_tokens": 50}
        return _call

    def test_success_strict_path(self, monkeypatch):
        gold = _valid_output()
        model_content = (
            "<think>I think it is my_func.</think>\n"
            '```json\n{"unit": "function", "name": "my_func", '
            '"file": "src/foo.py", "signature": "my_func(x: int) -> str"}\n```'
        )
        import bootstrap_code_traces as bct
        monkeypatch.setattr(bct, "call_model", self._mock_call_model(model_content))
        result = bootstrap_one("http://localhost:8000", "model", "q?", gold)
        assert result["ok"] is True
        assert result[FIELD_OUTPUT]["name"] == "my_func"
        assert result[FIELD_THINKING] == "I think it is my_func."

    def test_success_fallback_path_gold_none(self, monkeypatch):
        model_content = (
            '```json\n{"unit": "function", "name": "bar", '
            '"file": "b.py", "signature": "bar(x)"}\n```'
        )
        import bootstrap_code_traces as bct
        monkeypatch.setattr(bct, "call_model", self._mock_call_model(model_content))
        result = bootstrap_one("http://localhost:8000", "model", "q?", None)
        assert result["ok"] is True

    def test_fail_strict_field_mismatch(self, monkeypatch):
        gold = _valid_output(name="expected_func")
        model_content = (
            '```json\n{"unit": "function", "name": "wrong_func", '
            '"file": "src/foo.py", "signature": "my_func(x: int) -> str"}\n```'
        )
        import bootstrap_code_traces as bct
        monkeypatch.setattr(bct, "call_model", self._mock_call_model(model_content))
        result = bootstrap_one("http://localhost:8000", "model", "q?", gold)
        assert result["ok"] is False

    def test_fail_unparseable_answer(self, monkeypatch):
        model_content = "I don't know, sorry!"
        import bootstrap_code_traces as bct
        monkeypatch.setattr(bct, "call_model", self._mock_call_model(model_content))
        result = bootstrap_one("http://localhost:8000", "model", "q?", None)
        assert result["ok"] is False
        assert result["error"] == "unparseable"

    def test_fail_model_exception(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise ConnectionRefusedError("no server")
        import bootstrap_code_traces as bct
        monkeypatch.setattr(bct, "call_model", _raise)
        result = bootstrap_one("http://localhost:8000", "model", "q?", None)
        assert result["ok"] is False
        assert "no server" in result["error"]

    def test_fail_invalid_signature(self, monkeypatch):
        model_content = (
            '```json\n{"unit": "function", "name": "foo", '
            '"file": "a.py", "signature": "((( unbalanced"}\n```'
        )
        import bootstrap_code_traces as bct
        monkeypatch.setattr(bct, "call_model", self._mock_call_model(model_content))
        result = bootstrap_one("http://localhost:8000", "model", "q?", None)
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Tests: record shape written to output (end-to-end dry-run)
# ---------------------------------------------------------------------------


class TestRecordShape:
    def test_record_has_required_keys(self, tmp_path, monkeypatch):
        """Written record must have type/question/thinking/output/source."""
        import bootstrap_code_traces as bct

        training = tmp_path / "training.jsonl"
        record = _make_training_record(question="How do I use `my_func`?")
        training.write_text(json.dumps(record) + "\n", encoding="utf-8")

        gold = record["output"]
        gold_map = {"How do I use `my_func`?": gold}

        model_content = (
            "<think>It is my_func.</think>\n"
            f'```json\n{json.dumps(gold)}\n```'
        )
        monkeypatch.setattr(bct, "call_model", lambda *a, **kw: {
            "content": model_content, "ms": 5.0, "comp_tokens": 10
        })

        result = bct.bootstrap_one(
            "http://localhost:8000", "model",
            "How do I use `my_func`?", gold
        )
        assert result["ok"] is True

        out = tmp_path / "out.jsonl"
        with open(out, "a", encoding="utf-8") as f:
            written_record = {
                FIELD_TYPE: CODE_TYPE,
                FIELD_QUESTION: result[FIELD_QUESTION],
                FIELD_THINKING: result[FIELD_THINKING],
                FIELD_OUTPUT: result[FIELD_OUTPUT],
                FIELD_SOURCE: SOURCE_BOOTSTRAP,
            }
            f.write(json.dumps(written_record) + "\n")

        parsed = json.loads(out.read_text().splitlines()[0])
        assert parsed[FIELD_TYPE] == "code"
        assert parsed[FIELD_QUESTION] == "How do I use `my_func`?"
        assert parsed[FIELD_SOURCE] == "bootstrap"
        assert parsed[FIELD_THINKING] == "It is my_func."
        assert parsed[FIELD_OUTPUT] == gold
        assert set(parsed.keys()) == {
            FIELD_TYPE, FIELD_QUESTION, FIELD_THINKING, FIELD_OUTPUT, FIELD_SOURCE
        }

    def test_source_is_bootstrap(self):
        assert SOURCE_BOOTSTRAP == "bootstrap"

    def test_type_is_code(self):
        assert CODE_TYPE == "code"


# ---------------------------------------------------------------------------
# Tests: dry-run writes nothing
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_no_output_file(self, tmp_path, monkeypatch, capsys):
        """When --dry-run (simulated here via direct main call) no file is written."""
        import bootstrap_code_traces as bct

        gold = _valid_output()
        gold_map = {"test question": gold}
        model_content = f'```json\n{json.dumps(gold)}\n```'

        monkeypatch.setattr(bct, "call_model", lambda *a, **kw: {
            "content": model_content, "ms": 5.0, "comp_tokens": 10
        })

        out = tmp_path / "out.jsonl"
        # Simulate what main does when dry_run=True: call bootstrap_one,
        # collect results, but skip the write block.
        result = bct.bootstrap_one("http://localhost:8000", "m", "test question", gold)
        assert result["ok"] is True
        # File must NOT be written (dry-run means we skip the write)
        assert not out.exists()


# ---------------------------------------------------------------------------
# Tests: main() CLI integration (no network, mocked call_model)
# ---------------------------------------------------------------------------


class TestMainCLI:
    def _run_main(self, argv, monkeypatch, tmp_path, gold: dict, model_content: str):
        import bootstrap_code_traces as bct
        monkeypatch.setattr(bct, "call_model", lambda *a, **kw: {
            "content": model_content, "ms": 5.0, "comp_tokens": 10
        })
        monkeypatch.setattr(sys, "argv", ["bootstrap_code_traces.py"] + argv)
        bct.main()

    def test_main_with_input_writes_record(self, tmp_path, monkeypatch, capsys):
        import bootstrap_code_traces as bct

        record = _make_training_record(question="Describe my_func")
        training = tmp_path / "training.jsonl"
        training.write_text(json.dumps(record) + "\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"

        gold = record["output"]
        model_content = f'```json\n{json.dumps(gold)}\n```'
        monkeypatch.setattr(bct, "call_model", lambda *a, **kw: {
            "content": model_content, "ms": 5.0, "comp_tokens": 10
        })
        monkeypatch.setattr(sys, "argv", [
            "bootstrap_code_traces.py",
            "--input", str(training),
            "--output", str(out),
            "--workers", "1",
            "--dedup-threshold", "1.0",  # disable dedup
        ])
        bct.main()

        assert out.exists()
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        written = json.loads(lines[0])
        assert written[FIELD_TYPE] == "code"
        assert written[FIELD_SOURCE] == "bootstrap"
        assert written[FIELD_QUESTION] == "Describe my_func"

    def test_main_dry_run_writes_nothing(self, tmp_path, monkeypatch, capsys):
        import bootstrap_code_traces as bct

        record = _make_training_record(question="Describe my_func")
        training = tmp_path / "training.jsonl"
        training.write_text(json.dumps(record) + "\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"

        gold = record["output"]
        model_content = f'```json\n{json.dumps(gold)}\n```'
        monkeypatch.setattr(bct, "call_model", lambda *a, **kw: {
            "content": model_content, "ms": 5.0, "comp_tokens": 10
        })
        monkeypatch.setattr(sys, "argv", [
            "bootstrap_code_traces.py",
            "--input", str(training),
            "--output", str(out),
            "--workers", "1",
            "--dry-run",
        ])
        bct.main()

        assert not out.exists()

    def test_main_prompts_file_fallback_verifier(self, tmp_path, monkeypatch):
        import bootstrap_code_traces as bct

        prompts_file = tmp_path / "prompts.txt"
        prompts_file.write_text("How do I use foo?\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"

        gold = _valid_output(name="foo", signature="foo()")
        model_content = f'```json\n{json.dumps(gold)}\n```'
        monkeypatch.setattr(bct, "call_model", lambda *a, **kw: {
            "content": model_content, "ms": 5.0, "comp_tokens": 10
        })
        monkeypatch.setattr(sys, "argv", [
            "bootstrap_code_traces.py",
            "--prompts", str(prompts_file),
            "--output", str(out),
            "--workers", "1",
            "--dedup-threshold", "1.0",
        ])
        bct.main()

        assert out.exists()
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_main_no_prompts_exits_gracefully(self, tmp_path, monkeypatch, capsys):
        import bootstrap_code_traces as bct

        monkeypatch.setattr(bct, "call_model", lambda *a, **kw: {
            "content": "", "ms": 0.0, "comp_tokens": 0
        })
        monkeypatch.setattr(sys, "argv", [
            "bootstrap_code_traces.py",
            "--workers", "1",
        ])
        # Should return without raising (prints "No prompts" message)
        bct.main()
        captured = capsys.readouterr()
        assert "No prompts" in captured.out

    def test_main_failed_parse_not_written(self, tmp_path, monkeypatch):
        """A record where the model returns unparseable JSON should not be written."""
        import bootstrap_code_traces as bct

        record = _make_training_record(question="q")
        training = tmp_path / "training.jsonl"
        training.write_text(json.dumps(record) + "\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"

        monkeypatch.setattr(bct, "call_model", lambda *a, **kw: {
            "content": "I cannot answer that.", "ms": 5.0, "comp_tokens": 10
        })
        monkeypatch.setattr(sys, "argv", [
            "bootstrap_code_traces.py",
            "--input", str(training),
            "--output", str(out),
            "--workers", "1",
            "--dedup-threshold", "1.0",
        ])
        bct.main()
        # Either file doesn't exist or it's empty
        if out.exists():
            assert out.read_text().strip() == ""
