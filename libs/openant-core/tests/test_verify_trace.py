"""Stage-2 verify trace (OPENANT_TRACE_DIR): full agent conversation as JSONL.

Offline stub adapters; no real LLM calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_CORE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_CORE_ROOT))

from utilities.agentic_enhancer.repository_index import RepositoryIndex
from utilities.finding_verifier import FindingVerifier
from utilities.llm import PhaseBinding, TextBlock, ToolUseBlock
from utilities.llm.adapter import CompletionResult
from utilities.llm_client import reset_warning_state
from types import SimpleNamespace

from utilities.verify_trace import TRACE_DIR_ENV, extract_reasoning


@pytest.fixture(autouse=True)
def _reset():
    reset_warning_state()
    yield
    reset_warning_state()


class _ScriptedAdapter:
    name = "anthropic"
    supports_tools = True
    pricing = {"claude-x": {"input": 1.0, "output": 1.0}}

    def __init__(self, turns):
        self._turns = list(turns)

    def complete(self, *, model, system, messages, max_tokens, tools=None):
        return self._turns.pop(0)


class _RaisingAdapter:
    name = "anthropic"
    supports_tools = True
    pricing = {"claude-x": {"input": 1.0, "output": 1.0}}

    def complete(self, **kwargs):
        raise RuntimeError("boom")


def _verifier(adapter):
    binding = PhaseBinding(phase="verify", adapter=adapter, model="claude-x", provider_name="anthropic")
    return FindingVerifier(index=RepositoryIndex({}, repo_path=None), binding=binding)


def _two_turn_adapter():
    return _ScriptedAdapter([
        CompletionResult(
            content=[
                TextBlock("checking callers"),
                ToolUseBlock(id="t1", name="search_usages", input={"function_name": "foo"}),
            ],
            input_tokens=10, output_tokens=5, stop_reason="tool_use",
        ),
        CompletionResult(
            content=[ToolUseBlock(id="t2", name="finish",
                                  input={"agree": True, "correct_finding": "vulnerable"})],
            input_tokens=12, output_tokens=3, stop_reason="tool_use",
        ),
    ])


def _events(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run(tmp_path, monkeypatch, adapter=None, trace_id="a/b.java:Cls.m"):
    monkeypatch.setenv(TRACE_DIR_ENV, str(tmp_path))
    v = _verifier(adapter or _two_turn_adapter())
    result = v.verify_result(code="x = 1", finding="vulnerable", attack_vector="a",
                             reasoning="r", trace_id=trace_id)
    files = list(tmp_path.glob("*.jsonl"))
    return result, files


def test_trace_records_full_conversation(tmp_path, monkeypatch):
    result, files = _run(tmp_path, monkeypatch)
    assert len(files) == 1
    ev = _events(files[0])
    assert [e["event"] for e in ev] == [
        "start", "turn", "tool_result", "turn", "tool_result", "end"]
    assert [e["seq"] for e in ev] == [1, 2, 3, 4, 5, 6]

    start = ev[0]
    assert start["system_prompt"] and start["user_prompt"]
    assert "search_usages" in start["tools"] and "finish" in start["tools"]

    turn1 = ev[1]
    assert turn1["iteration"] == 1
    assert [b["type"] for b in turn1["blocks"]] == ["text", "tool_use"]
    assert turn1["blocks"][0]["text"] == "checking callers"
    assert turn1["blocks"][1]["input"] == {"function_name": "foo"}
    assert turn1["input_tokens"] == 10 and turn1["output_tokens"] == 5


def test_every_tool_use_has_exactly_one_matching_result(tmp_path, monkeypatch):
    _, files = _run(tmp_path, monkeypatch)
    ev = _events(files[0])
    uses = [b["id"] for e in ev if e["event"] == "turn" for b in e["blocks"] if b["type"] == "tool_use"]
    results = [e["tool_use_id"] for e in ev if e["event"] == "tool_result"]
    assert sorted(uses) == sorted(results) == ["t1", "t2"]


def test_end_event_matches_returned_result(tmp_path, monkeypatch):
    result, files = _run(tmp_path, monkeypatch)
    ev = _events(files[0])
    end = ev[-1]
    assert end["result"] == result.to_dict()
    assert end["result"]["iterations"] == sum(1 for e in ev if e["event"] == "turn") == 2


def test_disabled_without_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv(TRACE_DIR_ENV, raising=False)
    _verifier(_two_turn_adapter()).verify_result(
        code="x", finding="vulnerable", attack_vector="a", reasoning="r", trace_id="u1")
    assert list(tmp_path.iterdir()) == []


def test_disabled_without_trace_id(tmp_path, monkeypatch):
    monkeypatch.setenv(TRACE_DIR_ENV, str(tmp_path))
    _verifier(_two_turn_adapter()).verify_result(
        code="x", finding="vulnerable", attack_vector="a", reasoning="r")
    assert list(tmp_path.iterdir()) == []


def test_error_is_traced_and_still_raised(tmp_path, monkeypatch):
    monkeypatch.setenv(TRACE_DIR_ENV, str(tmp_path))
    with pytest.raises(RuntimeError, match="boom"):
        _verifier(_RaisingAdapter()).verify_result(
            code="x", finding="vulnerable", attack_vector="a", reasoning="r", trace_id="u2")
    ev = _events(next(tmp_path.glob("*.jsonl")))
    assert [e["event"] for e in ev] == ["start", "error"]
    assert ev[-1]["error_type"] == "RuntimeError" and ev[-1]["error"] == "boom"


def test_rerun_replaces_previous_trace(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, trace_id="same")
    _, files = _run(tmp_path, monkeypatch, trace_id="same")
    assert len(files) == 1
    assert [e["seq"] for e in _events(files[0])][0] == 1
    assert len(_events(files[0])) == 6


def test_unwritable_trace_dir_does_not_break_verification(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    monkeypatch.setenv(TRACE_DIR_ENV, str(blocker))
    r = _verifier(_two_turn_adapter()).verify_result(
        code="x", finding="vulnerable", attack_vector="a", reasoning="r", trace_id="u3")
    assert r.agree is True and r.incomplete is False


def test_distinct_ids_get_distinct_files(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, trace_id="a/b:c")
    _run(tmp_path, monkeypatch, trace_id="a_b:c")
    assert len(list(tmp_path.glob("*.jsonl"))) == 2


# ---------------- reasoning_content (chain of thought) ----------------

def _raw_with(reasoning_field="reasoning_content", text="step 1: think"):
    msg = SimpleNamespace(**{reasoning_field: text}) if reasoning_field else SimpleNamespace()
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_extract_reasoning_from_sdk_object_and_dict_and_alt_name():
    assert extract_reasoning(_raw_with()) == "step 1: think"
    assert extract_reasoning(_raw_with("reasoning", "alt")) == "alt"
    assert extract_reasoning({"choices": [{"message": {"reasoning_content": "d"}}]}) == "d"


def test_extract_reasoning_from_pydantic_style_model_extra():
    msg = SimpleNamespace(model_extra={"reasoning_content": "extra"})
    assert extract_reasoning(SimpleNamespace(choices=[SimpleNamespace(message=msg)])) == "extra"


def test_extract_reasoning_returns_none_when_absent_or_malformed():
    for raw in (None, {}, SimpleNamespace(), _raw_with(None), _raw_with("reasoning_content", ""),
                SimpleNamespace(choices=[])):
        assert extract_reasoning(raw) is None


def test_turn_event_records_reasoning(tmp_path, monkeypatch):
    adapter = _ScriptedAdapter([
        CompletionResult(
            content=[ToolUseBlock(id="t", name="finish", input={"agree": True, "correct_finding": "vulnerable"})],
            input_tokens=1, output_tokens=1, stop_reason="tool_use",
            raw=_raw_with(text="I should check the guard first"),
        ),
    ])
    _, files = _run(tmp_path, monkeypatch, adapter=adapter)
    turn = next(e for e in _events(files[0]) if e["event"] == "turn")
    assert turn["reasoning"] == "I should check the guard first"


def test_turn_event_reasoning_is_none_without_raw(tmp_path, monkeypatch):
    _, files = _run(tmp_path, monkeypatch)
    assert all(e["reasoning"] is None for e in _events(files[0]) if e["event"] == "turn")
