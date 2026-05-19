# Fault-Driven Request Shadow Trace (v0.1)

A per-request shadow trace that is **silently dropped on success** and
**dumped to a local JSONL file on fault**. Designed for postmortem
debugging of HTTP 4xx/5xx and abort errors in production where existing
metrics and logs are too coarse to identify root cause.

This is intentionally decoupled from the OTLP / `RequestStage` /
`ReqTimeStats` path. OTLP is success-biased (sample + buffer + async
export); shadow trace targets zero-loss on-fault expose.

## Quick Start

Enable by setting an output directory; everything else is optional:

```bash
export SGLANG_FAULT_TRACE_DIR=/var/log/sglang-fault-trace
# (optional) sample 1% of normal requests to keep a baseline
export SGLANG_FAULT_TRACE_SAMPLE_RATE=0.01
# (optional) decode-step ring slots per request (last K steps kept on fault)
export SGLANG_FAULT_TRACE_RING_SIZE=64

python -m sglang.launch_server --model-path ... --port 30000
```

When `SGLANG_FAULT_TRACE_DIR` is empty (default), the feature is fully
disabled and all writers short-circuit on a single `is None` check.

Mount `${SGLANG_FAULT_TRACE_DIR}` to a host volume / NAS for retention.

## Output

One JSONL file per process per UTC day:

```
${SGLANG_FAULT_TRACE_DIR}/sglang-fault-trace-${role}-${pid}-${YYYYMMDD}.jsonl
```

`${role}` is `tokenizer` for the tokenizer-manager process, and
`scheduler` / `prefill` / `decode` for the scheduler process (per
disaggregation mode). Each line is a self-contained JSON record.

### Record schema

```json
{
  "rid": "abc-123",
  "role": "decode",
  "verdict": "fault",
  "pid": 12345,
  "created_ts": 1779181755.726,
  "dumped_ts": 1779181755.738,
  "bootstrap_room": 9981,
  "events": [
    {"ts": 12345.001, "stage": "req_received",   "code": 20, "input_len": 512},
    {"ts": 12345.005, "stage": "prefill_finish", "code": 25, "extend_input_len": 512, "batch_size": 8, "first_tok": 21451},
    {"ts": 12345.420, "stage": "retract",        "code": 27, "count": 1},
    {"ts": 12345.421, "stage": "abort_set",      "code": 28, "msg": "...", "status_code": 400},
    {"ts": 12345.422, "stage": "abort_commit",   "code": 29, "reason": "FINISH_ABORT", "status_code": 400}
  ],
  "decode_ring_last": [
    {"step": 1817, "ts": 12345.300, "bsz": 96, "tok": 13, "fwd_us": 1, "retracted": 0},
    "...",
    {"step": 1880, "ts": 12345.420, "bsz": 95, "tok": 0,  "fwd_us": 1, "retracted": 1}
  ],
  "decode_agg": {"steps": 1880, "fwd_us_avg": 1, "fwd_us_max": 4, "retracts": 1},
  "finish_reason_type": "FINISH_ABORT",
  "finish_status_code": 400,
  "output_len": 1880
}
```

`verdict`:
- `"fault"` — request finished with an error (`is_error == True`)
- `"sample"` — request finished normally and was selected by the sample rate

`events` — a chronological log of stage transitions for the request
(tokenize, validate, prefill, retract, abort, finish, etc.). Stage name
and stage code are both included.

`decode_ring_last` — the **last K decode steps** (default K=64) kept in
a fixed-size ring buffer. Long requests overwrite older entries; the
final K steps before the fault are preserved. Each entry carries the
step index, timestamp, batch size, token id, accepted token count
(speculative decoding), and retracted flag.

`decode_agg` — running counters across the entire decode loop (not just
the ring): total steps, average / max per-step value, retract count.

When the request is from a PD-disaggregation deployment, `bootstrap_room`
is included so prefill-side and decode-side records can be joined offline.

### Stage codes

| code | name              | side       | when                                        |
|------|-------------------|------------|---------------------------------------------|
| 1    | http_arrive       | tokenizer  | HTTP request entered TokenizerManager       |
| 2    | tokenize          | tokenizer  | input ids ready                             |
| 4    | http_error        | tokenizer  | HTTP-layer error response                   |
| 5    | http_respond      | tokenizer  | normal response sent                        |
| 6    | abort_from_sched  | tokenizer  | scheduler returned an abort finish reason   |
| 20   | req_received      | scheduler  | Req object constructed                      |
| 22   | validate_fail    | scheduler  | input validation failed                     |
| 24   | prefill_chunk     | scheduler  | one chunk of chunked prefill done           |
| 25   | prefill_finish    | scheduler  | last prefill chunk done, first token chosen |
| 27   | retract           | scheduler  | request retracted (OOM recovery)            |
| 28   | abort_set         | scheduler  | `set_finish_with_abort` called              |
| 29   | abort_commit      | scheduler  | `finished_reason` committed as error        |
| 30   | finish_normal     | scheduler  | normal finish                               |
| 31   | finish_length     | scheduler  | finish due to max-tokens                    |

## Testing the feature

The recipes below assume a running server with the env vars set.

### 1. Verify the feature is enabled

After server start, the log should contain a line like:

```
shadow_trace: enabled role=tokenizer pid=... dir=/var/log/sglang-fault-trace ...
shadow_trace: enabled role=scheduler pid=... dir=/var/log/sglang-fault-trace ...
```

If you don't see these, `SGLANG_FAULT_TRACE_DIR` is not set or the
directory is not writable.

### 2. Trigger a 400 (input validation fault)

Send a request with `max_new_tokens` larger than the configured cap, or
with an empty `prompt`, or with an invalid sampling parameter:

```bash
curl -s http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "", "sampling_params": {"max_new_tokens": 10}}'
```

You should get an HTTP 400. Then:

```bash
ls /var/log/sglang-fault-trace/
# tail the latest tokenizer + scheduler files
tail -n 1 /var/log/sglang-fault-trace/sglang-fault-trace-tokenizer-*-*.jsonl | jq .
tail -n 1 /var/log/sglang-fault-trace/sglang-fault-trace-scheduler-*-*.jsonl | jq .
```

Both records should share the same `rid`. The scheduler-side record
should contain `validate_fail` or `abort_set` events; the tokenizer-side
record should contain `abort_from_sched`.

### 3. Trigger a successful request (sampled baseline)

With `SGLANG_FAULT_TRACE_SAMPLE_RATE=1.0` (set to 1 for testing only),
any successful request will produce a sampled record:

```bash
curl -s http://127.0.0.1:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "hello", "sampling_params": {"max_new_tokens": 8}}'
```

Then check the JSONL file. The record should have `"verdict": "sample"`
and contain decode steps in `decode_ring_last`.

Reset to a low rate (e.g. `0.01`) for production.

### 4. Trigger a forced 503 (queue saturation, requires load)

Saturate the waiting queue with concurrent requests. The scheduler will
abort old requests with `HTTPStatus.SERVICE_UNAVAILABLE`. The records on
both sides will have `finish_status_code: 503`.

### 5. Inspect the ring under decode pressure

Send a request with a very large `max_new_tokens` (e.g. 4096). When it
finishes (or you abort it), the dumped record's `decode_agg.steps`
should equal the actual generated count, while `decode_ring_last` keeps
only the last `SGLANG_FAULT_TRACE_RING_SIZE` (default 64) steps.

### 6. Cross-process aggregation by rid

```bash
RID=$(tail -n 1 /var/log/sglang-fault-trace/sglang-fault-trace-scheduler-*-*.jsonl | jq -r '.rid')
grep -h "\"rid\": \"$RID\"" /var/log/sglang-fault-trace/*.jsonl | jq -s .
```

The output will be an array of 2 records (tokenizer + scheduler) sharing
the same rid, giving the full request timeline.

## Performance Notes

Hot-path overhead is intentionally kept minimal:

- When the feature is disabled (`SGLANG_FAULT_TRACE_DIR` empty), every
  writer site is a single `is None` attribute check.
- When enabled but the request is not sampled, writers still record
  events in memory (needed to capture future faults), but no IO happens
  until the request terminates as a fault.
- All IO is on a single background daemon thread; the scheduler /
  tokenizer event loop is never blocked.
- The decode-step ring is a preallocated list of `RING_SIZE` slots;
  writes are O(1) tuple assignments.

## Limitations (v0.1)

- **No survival across process crashes.** If the scheduler is killed by
  a CUDA error or watchdog, the in-process ring is lost. A future
  iteration may move the backing store to shared memory; the current
  data structures already use plain primitives to make this swap
  straightforward.
- **No global / EPLB context yet.** EPLB rebalance, expert hot-spot
  drift, and DeepEP dispatch anomalies are not recorded in this
  per-request stream. A separate global ring will be added later.
- **No active anomaly triggers.** Currently only `FINISH_ABORT` and
  HTTP error paths trigger a fault dump. Spec-decode accept-rate dips
  and similar internal-consistency signals are not yet wired in.

## File map (for code reviewers)

| Path | Purpose |
|------|---------|
| `python/sglang/srt/observability/shadow_trace.py` | Module: data structure, sampling, background dumper |
| `python/sglang/srt/managers/schedule_batch.py` | `Req.shadow_trace` field, dispatch in `check_finished` / `set_finish_with_abort` / `reset_for_retract` |
| `python/sglang/srt/managers/scheduler.py` | `init_shadow_trace(role)` at process start |
| `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | Prefill chunk events, decode step ring writes |
| `python/sglang/srt/managers/tokenizer_manager.py` | `ReqState.shadow_trace`, http_arrive/tokenize/abort events, normal-finish dump |
| `python/sglang/srt/disaggregation/utils.py` | `prepare_abort` → dispatch shadow trace |

## Related design notes

The internal design rationale (why a separate path from OTLP, layer
breakdown, future evolution toward shared-memory / EPLB ring / active
triggers) is in the engineering note `SGLang Fault-Driven Request
Trace.md` (Obsidian).
