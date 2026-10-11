The review follow-up is in [45063b541](https://github.com/vllm-project/vllm-omni/commit/45063b541c5de050a1525b0e350e3382c1a45120), compared with the reviewed head `166a5b5b58549b84e92dd45eb2d9d52222dbb411`.

1. **Request-ID reuse and terminal delivery.** `pending_finished_request_ids()` and the Scheduler's output-update path now require the current request state to be terminal. Both synchronous and asynchronous empty-schedule paths make the same distinction before cleanup/finalization. The old finished IDs remain available for the next Worker wave, where cleanup precedes admission of the new request. A deterministic regression verifies that a reused ID's first unfinished step remains unfinished even though that wave contains the old cleanup ID.

   There was also a queue-identity window after state removal: a newly admitted request could replace the ID-to-queue mapping before the old terminal output was delivered. The Engine now captures the original queue before finalization and delivers directly to that queue. Scheduler state removal and cancellation-registry retirement share the Engine condition lock; executor `drop_output()` runs outside it. Regressions cover request mode, scheduled step output and unscheduled terminal step output. This preserves the existing ownership and admission boundaries; it does not add a second public generation protocol. See [Scheduler filtering](https://github.com/vllm-project/vllm-omni/blob/45063b541c5de050a1525b0e350e3382c1a45120/vllm_omni/diffusion/sched/base_scheduler.py#L470) and [terminal queue capture](https://github.com/vllm-project/vllm-omni/blob/45063b541c5de050a1525b0e350e3382c1a45120/vllm_omni/diffusion/diffusion_engine.py#L984).

2. **Worker bookkeeping at completion/error-to-idle.** The Worker now removes its request-to-LoRA entries on terminal `RunnerOutput.finished`, including returned error results. If the Runner call raises, it removes the affected scheduled IDs and propagates the exception. Nonterminal output chunks, surviving requests and unscheduled entries remain intact. Runner resource retirement stays with the Runner; this does not unload reusable adapter weights or require another cleanup RPC. The covered error boundary is the Runner call, not arbitrary LoRA-activation or profiler failures.

   The CPU lifecycle tests now exercise normal completion, last-request cancellation, cancellation of A while its batched peer B completes, and Runner error. Completion/cancellation cases repeat with the same ID across idle intervals. Each check occurs after terminal delivery and before another request or shutdown, and checks state/private-tensor weakrefs plus Worker bookkeeping. These use the real Engine, Scheduler, UniProc executor, Worker, Runner and InputBatch with lightweight CPU model computation.

3. **Multiprocessing, native GPU and frontend evidence.** Native OmniVoice validation now uses a separate `mp` Worker on an RTX 4060 Laptop GPU (8,188 MiB), float32/eager, four steps and execution capacity two. It covers 13 idle observations per version: warmup, lone completion, last-request abort, abort A with a same-batch B surviving, six alternating completion/cancellation cycles, and three C32 mixed rounds.

   | Rank 0, before another request or shutdown | Reviewed head `166a5b5b` | Follow-up `45063b541` |
   | --- | --- | --- |
   | Idle observations retaining Worker request metadata | 9, including warmup | 0 |
   | Live observed request states/private tensors | 0 | 0 |
   | CUDA allocated bytes at every idle | 3,545,060,352 | 3,545,060,352 |
   | CUDA reserved bytes at every idle | 4,299,161,600 | 4,299,161,600 |

   All 55 completed native outputs, including the 48 C32 survivors, are bitwise identical before/after and are finite, nonempty 47,040-sample audio tensors. The batched survivor also matches its standalone result for the same prompt/seed. C32 means **32 outstanding submissions with execution capacity two**, not a batch of 32; each round completes 16 and cancels 16. Final before/after round times were 1.777/1.787/1.772 s and 1.702/1.730/1.717 s; `InputBatch.make_batch` host p50 was 0.190/0.170 ms. An earlier reverse-order run measured the opposite timing direction. These short instrumented runs do not establish a stable performance improvement or regression, and both versions already contain the original Runner cleanup, so this is not a benchmark of that cleanup's repacking cost.

   The real `/v1/audio/speech` frontend test returns a valid warmup WAV, then closes a second request's TCP connection during execution. Cancellation propagates through `with_cancellation`, AsyncOmni/Orchestrator, `InlineStageDiffusionClient`, Engine and the separate `mp` Worker. The script never calls `engine.abort()` directly. The target ID is present in the Worker's cache immediately before release; release occurs 61.2 ms after disconnect. The following idle snapshot has empty Runner cache/batch and Worker bookkeeping, dead observed request/private-tensor weakrefs, and allocated/reserved bytes of 3,545,060,352/4,299,161,600.

   **Two-GPU Qwen-Image:** real `mp`, TP=2, SP/CFG/DP=1 on 2× A100-SXM4 80GB, BF16/eager, 512×512, 20 steps, seed 42 and execution capacity two. Both versions run the same 13-observation matrix. On **each rank**, 9 baseline idle observations retain Worker metadata (including warmup); the follow-up has zero retained entries. Observed request states/private tensors are dead in both versions, with no missing request observations. At every native idle point, each rank has **allocated=42,916,440,576 B, reserved=44,379,930,624 B** in both versions.

   All **55 corresponding completed Qwen outputs** have identical before/after tensor hashes, including 48 C32 completions. Each C32 round completes 16 and cancels 16. Qwen standalone versus batched execution is not bitwise equal; the equality claim is specifically the same-configuration before/after comparison. C32 times were 54.630/54.653/54.131 s before and 54.926/54.186/54.214 s after. Each round has 169 `make_batch` calls (167 at batch size two, two at size one); this includes cached repacking, not 169 full rebuilds. Rank-0 host p50 is 0.154–0.170 ms and host total is about 53–56 ms per round. No clear performance change is established.

   **Native image frontend:** `/v1/images/generations` returns a decodable 512×512 warmup image. The next request's TCP connection is closed during execution; it runs only 2/20 steps. Both ranks contain the active target in `release.cache_before`. Disconnect-to-last-rank-release is 274.8 ms before and 276.8 ms after. Both versions then show empty request containers/bookkeeping and dead private weakrefs before any later request or shutdown, with allocated/reserved unchanged from warmup. This confirms real frontend cancellation and the collective cleanup RPC; cancellation already worked in the reviewed head. The disconnected client is not a test of a waiting HTTP consumer; terminal delivery to an active Engine API consumer is covered separately.

   Qwen environment: Python 3.12.3, vLLM 0.31.0, torch 2.13.0+cu130, transformers 5.14.1, diffusers 0.40.0. The installed flashinfer-python/cubin versions differ (0.7.0.post1/0.6.18.post1); the framework catches this import error and disables the optional ring backend. The tested TP=2/SP=1 path runs in this identical environment for both versions; no ring-backend validation is claimed.

   The observer preserves native encode/denoise/decode and cleanup, records weakrefs/scalars, and synchronizes GPU work before allocator sampling. Its read-only snapshot RPC does not run another Runner step or cleanup. It may replace the Worker dispatch loop's last returned local, so the retention assertions specifically cover the named Runner/Worker ownership. Shared reusable `timesteps` buffers are excluded from request-private assertions.

   The original unfixed baseline was `f69b1f2b19cece03f6736c7da093ea25de0e24fd`, where the earlier OmniVoice experiment measured bounded 22.5 KiB idle tail-batch retention. The reviewed `166a5b5b` already fixes that Runner retention. The metadata correction above therefore must not be described as another GPU memory leak or as additional GPU memory savings.

4. **Existing Engine test stub.** `MockScheduler` now implements `pending_finished_request_ids()`, and `test_async_add_req_and_stream_response` passes with the background loop running.

CPU validation totals **284 passed, 4 deselected**, from three non-overlapping selections: Scheduler/Engine/cleanup/RPC (224), step pipeline/lifecycle (50), and cancellation (10, including real two-process Gloo). Applying only the updated tests to `166a5b5b` preserves 5 Scheduler/queue failures, 3 Worker failures and 4 lifecycle failures. Their assertions identify the stale notification, wrong queue or retained Worker bookkeeping; environment/harness startup failures are recorded separately.

Supplementary evidence: [download ZIP](https://raw.githubusercontent.com/Laceprndpm/vllm-omni/2464e717c0f39d772828f12a315db23a386104a9/revision-evidence.zip) (663,936 bytes), [browse scripts](https://github.com/Laceprndpm/vllm-omni/tree/2464e717c0f39d772828f12a315db23a386104a9/revision/scripts). SHA256: `040eea934d20b7dc83b460302bcdc6926bc5fcdc060e1242dadced97fd021018`. It contains scripts, the exact revision patch, manifests, per-rank JSONL, output hashes and before/after logs; it excludes model weights and generated output tensors. Relevant logs are `scheduler-engine-before.log`, `worker-before.log`, `lifecycle-before.log`, and the matching after logs. Native evidence is under `evidence/omnivoice-mp-{before,after}`, `evidence/omnivoice-http-after` and `evidence/qwen/{native,http}-{before,after}`. `evidence/qwen/idle-memory.csv` gives the per-rank allocator observations.

<details>
<summary>Reproduction commands</summary>

Run from the repository root with Python 3.12, vLLM 0.31.0 and matching project test dependencies. CPU regressions need no model weights or GPU:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -sv \
  tests/diffusion/test_diffusion_scheduler.py \
  tests/diffusion/test_diffusion_engine.py \
  tests/diffusion/test_diffusion_engine_cleanup.py \
  tests/diffusion/test_diffusion_engine_rpc_routing.py \
  -m 'core_model and cpu' --run-level=core_model

CUDA_VISIBLE_DEVICES='' python -m pytest -sv \
  tests/diffusion/test_diffusion_step_pipeline.py \
  tests/diffusion/test_step_request_lifecycle.py \
  -m 'core_model and cpu' --run-level=core_model

CUDA_VISIBLE_DEVICES='' python -m pytest -sv \
  tests/diffusion/test_request_cancellation.py \
  -m 'core_model and cpu' --run-level=core_model
```

To reproduce failures on a separate checkout of `166a5b5b58549b84e92dd45eb2d9d52222dbb411`, extract the archive at that checkout's root and apply only its test changes before running the commands:

```bash
git apply --include='tests/**' step-lifecycle-revision/evidence/revision.patch
```

For native single-GPU multiprocessing, extract the archive at the fixed checkout's root, supply the native OmniVoice weights and use a fresh output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:./step-lifecycle-revision/scripts \
python step-lifecycle-revision/scripts/native_lifecycle.py \
  --model /path/to/OmniVoice --model-class OmniVoicePipeline \
  --tp 1 --dtype float32 --steps 4 --c32 \
  --output-dir ./lifecycle-results/omnivoice-after

python step-lifecycle-revision/scripts/http_omnivoice_cancel.py \
  --repo . --model /path/to/OmniVoice \
  --output-dir ./lifecycle-results/omnivoice-http-after
```

Repeat the native matrix from the reviewed checkout with a different output directory for the before comparison; its nonzero exit is expected when it detects retained Worker metadata. OmniVoice weights used here: `k2-fsa/OmniVoice`, revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`.

For the two-GPU run, use native `Qwen/Qwen-Image` weights (revision `75e0b4be04f60ec59a75f475837eced720f823b6`) and fresh output directories:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:./step-lifecycle-revision/scripts \
python step-lifecycle-revision/scripts/native_lifecycle.py \
  --model /path/to/Qwen-Image --model-class QwenImagePipeline \
  --tp 2 --dtype bfloat16 --steps 20 --c32 --c32-rounds 3 \
  --output-dir ./lifecycle-results/qwen-after

python step-lifecycle-revision/scripts/http_qwen_cancel.py \
  --repo . --model /path/to/Qwen-Image --steps 20 --port 18081 \
  --output-dir ./lifecycle-results/qwen-http-after
```

Repeat from the reviewed checkout with new output directories for the baseline. The model file names and sizes were checked (31 files, 57,704,594,653 bytes); no full-content hash verification of the 53.74 GiB weights is claimed.

</details>
