## Purpose

Fix two step-mode failures at the last-request-to-idle boundary:

- [P1] Cancelling the last request between steps can leave its terminal output pending, keeping an Engine API consumer waiting.
- [P2] Completion removes `state_cache` ownership but leaves `InputBatch.states` retaining the request and its private tensors during idle.

Use shared Runner retirement for terminal request resources and a serialized Worker RPC for cancellation. Deliver pending terminal outputs before the Engine waits. Keep successful output tensors owned by their outputs through transport and consumption.

The review follow-up also fixes the request-ID reuse boundary: old finished IDs remain available for Worker cleanup, but only the current terminal Scheduler state can trigger terminal notification. Capture the original output queue before finalization allows ID reuse, so an old result cannot enter a new request's queue. Worker completion and Runner-error paths now retire their own request-to-LoRA bookkeeping without another scheduler wave or cleanup RPC. Update the existing Engine test Scheduler stub.

This follows DIFF-RUNTIME-INV-003 and preserves Scheduler/Worker/output ownership boundaries; it does not introduce another public request-generation protocol. Independent of Pi0.5 D0 PR #8331.

## Test Plan

CPU regressions exercise the real Engine, Scheduler, UniProc Executor, Worker, Runner and InputBatch with a small pipeline. They check terminal delivery, surviving outputs, repeated idle/resume, weakref release and Worker bookkeeping before any later request or shutdown. Focused Scheduler/Engine tests deterministically cover reused IDs and original-queue delivery, while Worker tests preserve nonterminal chunks and unscheduled entries.

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -sv \
  tests/diffusion/test_step_request_lifecycle.py \
  -m 'core_model and cpu' --run-level=core_model
```

Requires Python 3.12, vLLM 0.31.0 and matching project test dependencies; no weights or GPU for CPU regressions. Native GPU scripts, exact commands, frozen-baseline failures and per-rank evidence are provided separately in the supplementary comment/archive.

**vLLM Version:** 0.31.0.
**vLLM-Omni Commit:** `45063b541c5de050a1525b0e350e3382c1a45120`; review baseline `166a5b5b58549b84e92dd45eb2d9d52222dbb411`. The original unfixed lifecycle baseline was `f69b1f2b19cece03f6736c7da093ea25de0e24fd`.

## Test Result

- Review regressions fail on `166a5b5b5854` for stale lifecycle notification, wrong output queue or retained Worker bookkeeping, then pass with the fix.
- Related CPU selections: **284 passed, 4 deselected** across the Scheduler/Engine/RPC, Runner/lifecycle and cancellation commands; cancellation includes a real two-process Gloo test.
- Native OmniVoice (one GPU, mp) and Qwen-Image (2× A100 80GB, TP=2, mp): 13 idle observations per version, including last-request cancellation, a surviving batched peer, repeated idle/resume and three C32 mixed rounds. The follow-up clears all observed request resources/bookkeeping; 55 completed outputs per model match their baseline counterparts bitwise. Real HTTP disconnect cancellation passes for both frontends; Qwen records cleanup and allocator state on both ranks before a later request or shutdown.
- The original native OmniVoice experiment measured **22.5 KiB** of bounded idle tail-batch retention, not unbounded accumulation. This review's Worker metadata is CPU bookkeeping and is not evidence of an additional GPU leak.
- Applicable changed-file pre-commit hooks pass except mypy: the same **52 diagnostics** occur before and after, with no added diagnostics. Full-repository pre-commit also reports unrelated existing failures; logs are included.

Supplementary evidence: [review response and detailed results](https://github.com/vllm-project/vllm-omni/pull/8719#issuecomment-6105742034), [download ZIP](https://raw.githubusercontent.com/Laceprndpm/vllm-omni/2464e717c0f39d772828f12a315db23a386104a9/revision-evidence.zip) and [browse native scripts](https://github.com/Laceprndpm/vllm-omni/tree/2464e717c0f39d772828f12a315db23a386104a9/revision/scripts). The original investigation remains in the [earlier evidence comment](https://github.com/vllm-project/vllm-omni/pull/8719#issuecomment-6094642162).

AI assistance: Codex prepared the investigation, implementation, tests and evidence. Commits include DCO sign-off; human review is required before merging.
