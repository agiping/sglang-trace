# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Fault-driven Request Shadow Trace (v0.1).

Per-request shadow state that is silently dropped on success and dumped to a
local JSONL file on fault. Intentionally decoupled from the OTLP/RequestStage
path: hot-path writes are O(1) tuple appends into a bounded ring; serialization
and IO happen on a single background thread.

Env vars:
  SGLANG_FAULT_TRACE_DIR          enable feature when non-empty; output dir
  SGLANG_FAULT_TRACE_SAMPLE_RATE  sample rate for normal requests (default 0.01)
  SGLANG_FAULT_TRACE_RING_SIZE    decode step ring slots per req (default 64)
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ----- Stage codes (small ints for cheap hot-path writes) ---------------------

# Tokenizer-manager side
STAGE_HTTP_ARRIVE = 1
STAGE_TOKENIZE = 2
STAGE_DISPATCH_TO_SCHED = 3
STAGE_HTTP_ERROR = 4
STAGE_HTTP_RESPOND = 5
STAGE_ABORT_FROM_SCHED = 6

# Scheduler side
STAGE_REQ_RECEIVED = 20
STAGE_VALIDATE_OK = 21
STAGE_VALIDATE_FAIL = 22
STAGE_PREFILL_FORWARD = 23
STAGE_PREFILL_CHUNK = 24
STAGE_PREFILL_FINISH = 25
STAGE_DECODE_FIRST = 26
STAGE_RETRACT = 27
STAGE_ABORT_SET = 28
STAGE_ABORT_COMMIT = 29
STAGE_FINISH_NORMAL = 30
STAGE_FINISH_LENGTH = 31

_STAGE_NAMES = {
    1: "http_arrive",
    2: "tokenize",
    3: "dispatch_to_sched",
    4: "http_error",
    5: "http_respond",
    6: "abort_from_sched",
    20: "req_received",
    21: "validate_ok",
    22: "validate_fail",
    23: "prefill_forward",
    24: "prefill_chunk",
    25: "prefill_finish",
    26: "decode_first",
    27: "retract",
    28: "abort_set",
    29: "abort_commit",
    30: "finish_normal",
    31: "finish_length",
}


def stage_name(code: int) -> str:
    return _STAGE_NAMES.get(code, f"stage_{code}")


# ----- Config (read once at process init) ------------------------------------

_ENABLED: bool = False
_OUTPUT_DIR: str = ""
_SAMPLE_RATE: float = 0.0
_RING_SIZE: int = 64
_ROLE: str = "unknown"
_PID: int = 0
_INIT_LOCK = threading.Lock()
_INIT_DONE = False


def _read_env_config() -> None:
    global _ENABLED, _OUTPUT_DIR, _SAMPLE_RATE, _RING_SIZE
    out_dir = os.environ.get("SGLANG_FAULT_TRACE_DIR", "").strip()
    if not out_dir:
        _ENABLED = False
        return
    _ENABLED = True
    _OUTPUT_DIR = out_dir
    try:
        _SAMPLE_RATE = float(os.environ.get("SGLANG_FAULT_TRACE_SAMPLE_RATE", "0.01"))
    except ValueError:
        _SAMPLE_RATE = 0.01
    try:
        _RING_SIZE = int(os.environ.get("SGLANG_FAULT_TRACE_RING_SIZE", "64"))
    except ValueError:
        _RING_SIZE = 64
    if _RING_SIZE <= 0:
        _RING_SIZE = 64


def init_shadow_trace(role: str) -> None:
    """Call once per process. Safe to call multiple times (idempotent)."""
    global _INIT_DONE, _ROLE, _PID
    with _INIT_LOCK:
        if _INIT_DONE:
            return
        _read_env_config()
        _ROLE = role or "unknown"
        _PID = os.getpid()
        _INIT_DONE = True
        if _ENABLED:
            try:
                os.makedirs(_OUTPUT_DIR, exist_ok=True)
            except OSError as e:
                logger.error(
                    "shadow_trace: failed to create dir %s: %s; disabling",
                    _OUTPUT_DIR,
                    e,
                )
                globals()["_ENABLED"] = False
                return
            _Dumper.instance().start()
            logger.info(
                "shadow_trace: enabled role=%s pid=%d dir=%s sample_rate=%.4f ring=%d",
                _ROLE,
                _PID,
                _OUTPUT_DIR,
                _SAMPLE_RATE,
                _RING_SIZE,
            )


def is_enabled() -> bool:
    return _ENABLED


def should_sample(rid: str) -> bool:
    """Stable per-rid sampling decision (no float rng on hot path)."""
    if _SAMPLE_RATE <= 0.0:
        return False
    if _SAMPLE_RATE >= 1.0:
        return True
    # hash() of str is randomized per process via PYTHONHASHSEED; this is fine
    # for sampling purposes (we just need deterministic-within-process).
    return (hash(rid) & 0xFFFFFF) < int(_SAMPLE_RATE * 0xFFFFFF)


# ----- ShadowTrace per-request data structure --------------------------------


@dataclass
class ShadowTrace:
    """Per-request shadow trace. Mutated only by the owning thread.

    Fields are deliberately plain (lists/dicts of primitives) so we can later
    swap the backing store to shared memory without changing call sites.
    """

    rid: str
    role: str
    bootstrap_room: Optional[int] = None
    sampled: bool = False  # True iff selected by sample-rate (vs fault-only)
    created_ts: float = field(default_factory=time.time)
    # Event log: list of (monotonic_ts, stage_code, payload_dict)
    events: List[Tuple[float, int, Dict[str, Any]]] = field(default_factory=list)
    # Decode-step ring (preallocated, head-only writer).
    decode_ring: List[Optional[Tuple[int, float, int, int, int, int]]] = field(
        default_factory=list
    )
    decode_head: int = 0  # next write index (monotonically increasing)
    # Aggregate decode stats (cheap per-step running counters).
    agg_decode_steps: int = 0
    agg_decode_fwd_us_sum: int = 0
    agg_decode_fwd_us_max: int = 0
    agg_retract_count: int = 0

    def __post_init__(self):
        if not self.decode_ring:
            self.decode_ring = [None] * _RING_SIZE

    # --- writers ---------------------------------------------------------

    def event(self, stage_code: int, **payload: Any) -> None:
        self.events.append((time.perf_counter(), stage_code, payload))

    def decode_step(
        self,
        step_idx: int,
        bsz: int,
        token_id: int,
        fwd_us: int,
        retracted: int = 0,
    ) -> None:
        slot = self.decode_head % _RING_SIZE
        self.decode_ring[slot] = (
            step_idx,
            time.perf_counter(),
            bsz,
            token_id,
            fwd_us,
            retracted,
        )
        self.decode_head += 1
        self.agg_decode_steps += 1
        self.agg_decode_fwd_us_sum += fwd_us
        if fwd_us > self.agg_decode_fwd_us_max:
            self.agg_decode_fwd_us_max = fwd_us
        if retracted:
            self.agg_retract_count += 1

    # --- serialize -------------------------------------------------------

    def to_dict(self, verdict: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Reconstruct ring in chronological order.
        last_steps: List[Dict[str, Any]] = []
        if self.decode_head > 0:
            n = min(self.decode_head, _RING_SIZE)
            start = self.decode_head - n
            for i in range(start, self.decode_head):
                slot = self.decode_ring[i % _RING_SIZE]
                if slot is None:
                    continue
                step_idx, ts, bsz, tok, fwd_us, retracted = slot
                last_steps.append(
                    {
                        "step": step_idx,
                        "ts": ts,
                        "bsz": bsz,
                        "tok": tok,
                        "fwd_us": fwd_us,
                        "retracted": retracted,
                    }
                )

        out: Dict[str, Any] = {
            "rid": self.rid,
            "role": self.role,
            "verdict": verdict,
            "pid": _PID,
            "created_ts": self.created_ts,
            "dumped_ts": time.time(),
            "events": [
                {"ts": ts, "stage": stage_name(code), "code": code, **payload}
                for ts, code, payload in self.events
            ],
            "decode_ring_last": last_steps,
            "decode_agg": {
                "steps": self.agg_decode_steps,
                "fwd_us_avg": (
                    self.agg_decode_fwd_us_sum // self.agg_decode_steps
                    if self.agg_decode_steps
                    else 0
                ),
                "fwd_us_max": self.agg_decode_fwd_us_max,
                "retracts": self.agg_retract_count,
            },
        }
        if self.bootstrap_room is not None:
            out["bootstrap_room"] = self.bootstrap_room
        if extra:
            out.update(extra)
        return out


def new_trace(
    rid: str,
    role: Optional[str] = None,
    bootstrap_room: Optional[int] = None,
    sampled: Optional[bool] = None,
) -> Optional[ShadowTrace]:
    """Construct a ShadowTrace iff the feature is enabled AND
    (sampled by rate OR caller requests one explicitly).

    Returns None when disabled or unsampled. Callers MUST check for None
    before invoking writers (the if-None check is the cheap fast path).
    """
    if not _ENABLED:
        return None
    if sampled is None:
        sampled = should_sample(rid)
    # Even if not sampled, we still construct a trace because faults need
    # the buffer; on successful finish we will check `sampled` to decide
    # whether to dump or drop.
    return ShadowTrace(
        rid=rid,
        role=role or _ROLE,
        bootstrap_room=bootstrap_room,
        sampled=sampled,
    )


# ----- Dispatch ---------------------------------------------------------------


def submit_fault(trace: Optional[ShadowTrace], verdict: str, **extra: Any) -> None:
    """Dispatch a trace to the background dumper. No-op if trace is None or
    feature disabled. Safe to call from any thread."""
    if trace is None or not _ENABLED:
        return
    try:
        rec = trace.to_dict(verdict=verdict, extra=extra or None)
    except Exception:  # noqa: BLE001
        logger.exception("shadow_trace: serialization failed for rid=%s", trace.rid)
        return
    _Dumper.instance().submit(rec)


def submit_normal_or_drop(trace: Optional[ShadowTrace], **extra: Any) -> None:
    """Called on successful finish. Dump only if `trace.sampled`, else drop."""
    if trace is None:
        return
    if not trace.sampled:
        return
    submit_fault(trace, verdict="sample", **extra)


# ----- Background dumper ------------------------------------------------------


class _Dumper:
    """Single per-process background thread that drains a queue to a JSONL file."""

    _instance: Optional["_Dumper"] = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls) -> "_Dumper":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self._q: queue.Queue = queue.Queue(maxsize=8192)
        self._thread: Optional[threading.Thread] = None
        self._fp = None
        self._cur_date: Optional[str] = None
        self._stopped = False
        self._dropped = 0
        self._dropped_log_at = 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="sglang-shadow-trace-dumper",
            daemon=True,
        )
        self._thread.start()

    def submit(self, rec: Dict[str, Any]) -> None:
        try:
            self._q.put_nowait(rec)
        except queue.Full:
            self._dropped += 1
            now = time.time()
            if now - self._dropped_log_at > 5.0:
                logger.warning(
                    "shadow_trace: dumper queue full, dropped %d records",
                    self._dropped,
                )
                self._dropped_log_at = now

    def _file_for_today(self):
        today = datetime.utcnow().strftime("%Y%m%d")
        if self._cur_date != today or self._fp is None:
            if self._fp is not None:
                try:
                    self._fp.close()
                except Exception:  # noqa: BLE001
                    pass
            fname = f"sglang-fault-trace-{_ROLE}-{_PID}-{today}.jsonl"
            path = os.path.join(_OUTPUT_DIR, fname)
            self._fp = open(path, "a", encoding="utf-8", buffering=1)  # line-buffered
            self._cur_date = today
        return self._fp

    def _loop(self) -> None:
        while not self._stopped:
            try:
                rec = self._q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                fp = self._file_for_today()
                fp.write(json.dumps(rec, ensure_ascii=False, default=str))
                fp.write("\n")
            except Exception:  # noqa: BLE001
                logger.exception("shadow_trace: write failed")

    def shutdown(self) -> None:
        self._stopped = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:  # noqa: BLE001
                pass


def shutdown_shadow_trace() -> None:
    if _ENABLED:
        _Dumper.instance().shutdown()
