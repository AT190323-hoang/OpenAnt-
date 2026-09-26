"""Per-unit JSONL trace of the Stage 2 (verify) agent loop.

Enabled only when OPENANT_TRACE_DIR is set. One file per unit, written
incrementally so a crashed or truncated run still leaves its partial trace.
Tracing must never break verification: every write swallows its own errors.
"""
import hashlib
import json
import os
import re
import time
from typing import Any, Optional

TRACE_DIR_ENV = "OPENANT_TRACE_DIR"


def _trace_path(trace_dir: str, trace_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", trace_id)[:120]
    digest = hashlib.sha1(trace_id.encode("utf-8")).hexdigest()[:8]
    return os.path.join(trace_dir, f"{safe}-{digest}.jsonl")


def blocks_to_dicts(blocks) -> list[dict[str, Any]]:
    out = []
    for b in blocks:
        name = type(b).__name__
        if name == "TextBlock":
            out.append({"type": "text", "text": b.text})
        elif name == "ToolUseBlock":
            out.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
        else:
            out.append({"type": name})
    return out


def extract_reasoning(raw: Any) -> Optional[str]:
    """The model's chain of thought, when the provider returns it.

    OpenAI-compatible reasoning hosts put it in ``message.reasoning_content``
    (or ``reasoning``); the adapter drops it, but the provider-native response
    stays in ``CompletionResult.raw``. Diagnostic only, never raises.
    """
    try:
        choices = raw["choices"] if isinstance(raw, dict) else raw.choices
        msg = choices[0]["message"] if isinstance(choices[0], dict) else choices[0].message
        for name in ("reasoning_content", "reasoning"):
            val = msg.get(name) if isinstance(msg, dict) else getattr(msg, name, None)
            if val is None and not isinstance(msg, dict):
                val = (getattr(msg, "model_extra", None) or {}).get(name)
            if isinstance(val, str) and val:
                return val
    except Exception:
        pass
    return None


class VerifyTrace:
    def __init__(self, trace_id: Optional[str]):
        trace_dir = os.environ.get(TRACE_DIR_ENV)
        self.enabled = bool(trace_dir and trace_id)
        self.path = None
        self._started = False
        if self.enabled:
            try:
                os.makedirs(trace_dir, exist_ok=True)
                self.path = _trace_path(trace_dir, trace_id)
            except OSError:
                self.enabled = False
        self.trace_id = trace_id
        self._seq = 0

    def emit(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        try:
            self._seq += 1
            record = {
                "seq": self._seq,
                "ts": time.time(),
                "event": event,
                "trace_id": self.trace_id,
                **fields,
            }
            # first event of a run truncates, so a re-run replaces the old trace
            mode = "a" if self._started else "w"
            with open(self.path, mode, encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._started = True
        except Exception:
            pass
