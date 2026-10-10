# Step-request idle lifecycle evidence

This bundle accompanies an independent vLLM-Omni framework fix. It is unrelated to Pi0.5 D0 PR #8331.

- [P1] The last between-step cancellation can leave its terminal result pending at Engine idle.
- [P2] A completed request can remain reachable through `runner.input_batch.states`, including its private tensors.

## Revisions and scope

- Investigation baseline: `f69b1f2b19cece03f6736c7da093ea25de0e24fd`.
- PR base: `b46d0f5de17befe567e653e43c985894cbac8d7b` (one subsequent NPU offloader change; the investigated production paths are unchanged).
- Fix: `bcb3b4b2b86f92203a5430413b73da3561f21132`.
- Python 3.12, vLLM 0.31.0, PyTorch 2.13.0+cu132, Transformers 5.14.1. Use the project's compatible test/runtime dependencies.

The final two-test CPU suite was rerun after the PR-base update. Earlier native-model and expanded-suite logs tested the same production fix on the investigation baseline; later changes only simplified tests/documentation.

## 1. Two CPU regressions (no weights or GPU)

On the fixed checkout, the regression file is already under `tests/diffusion`:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -sv tests/diffusion/test_step_request_lifecycle.py -m 'core_model and cpu' --run-level=core_model
```

To reproduce on the baseline, copy `regression/test_step_request_lifecycle.py` from this bundle into the same location in a separate baseline checkout. Run the identical command. Expected: baseline **2 failures**, fixed **2 passes**. These failures are resource retention and missing terminal output, not import/setup errors.

The framework normally constructs Engine, StepScheduler, UniProc executor, Worker and Runner. A tiny CPU pipeline substitutes model weights; hardware/distributed setup and device statistics are bypassed. Cleanup and scheduling are real. Assertions run at idle before future requests or Engine shutdown.

`regression/logs/` contains the final before/after logs, related CPU results (64 passed, 8 deselected), changed-file pre-commit output and evidence that the 43 mypy errors also exist on baseline. Other applicable changed-file hooks pass.

## 2. Native OmniVoice requests (CUDA and pretrained weights)

Hardware used: RTX 4060 Laptop, 8 GB. Model: `k2-fsa/OmniVoice`, revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`. Download that complete revision separately; `native/model.json` records the two weight-file hashes. Weights and download credentials are not included.

From this extracted bundle, use an absolute checkout path and local model path. Run each command in a separate process, once against baseline and once against the fix:

```bash
PYTHONPATH=/path/to/checkout HF_HUB_OFFLINE=1 python native/repro_abort_notification.py --model /path/to/OmniVoice
PYTHONPATH=/path/to/checkout HF_HUB_OFFLINE=1 python native/repro_tensor_retention.py --model /path/to/OmniVoice
```

Both scripts use native `OmniVoicePipeline` without replacing encode/denoise/decode. The prompt is "Hello, this is a lifecycle test.", with four steps and seed 42. Observers hold only weakrefs/scalars. No new request, manual Runner tick, shutdown, gc.collect or empty_cache triggers release before measurement.

| Observation | Baseline | Fixed |
| --- | --- | --- |
| Last cancellation, consumer after one second idle | Pending | Receives finished+aborted |
| Normal completion | 47,040 finite samples | 47,040 finite samples |
| Completed state / model audio_mask alive at idle | Yes / yes | No / no |
| Idle memory_allocated | 2,182,847,488 B | 2,182,824,448 B |
| Idle memory_reserved | 2,879,389,696 B | 2,879,389,696 B |

Expected exit codes: **1 on baseline**, **0 with the fix**. Logs are in `native/logs/`; machine-readable results are in `native/results.json`.

Optional storage accounting, run under the same conditions:

```bash
PYTHONPATH=/path/to/checkout HF_HUB_OFFLINE=1 python native/account_batch.py --model /path/to/OmniVoice
```

The allocation difference is exactly **23,040 B (22.5 KiB)** across nine unique CUDA storage blocks reachable from InputBatch: 10 KiB of direct batch buffers and 12.5 KiB of request-state tensors. Shared storage is counted once, using allocator block sizes rather than logical tensor bytes. Detailed paths are in `native/account-results.json`.

This establishes idle tail-request retention, not unbounded accumulation. Reserved memory is allocator caching. HTTP hangs, OOM and throughput gains were not measured; MultiProc, Ray and multi-GPU execution were not validated.

## 3. Archived expanded checks (outside the PR regression suite)

`extended/test_step_request_lifecycle.py` is the earlier 276-line validation suite. It covers CPU/CUDA execution-time cancellation, live peers, duplicate aborts, six repeated idle cycles, output validity and denoise failure. Baseline: 10 failures; fixed: 10 passes. Before/after logs are alongside it.

To rerun it, copy the file into a disposable checkout as `tests/diffusion/test_step_request_lifecycle_extended.py` and run:

```bash
python -m pytest -sv tests/diffusion/test_step_request_lifecycle_extended.py
# CPU-only selection:
CUDA_VISIBLE_DEVICES='' python -m pytest -sv tests/diffusion/test_step_request_lifecycle_extended.py -m 'core_model and cpu' --run-level=core_model
```

This expanded suite uses a synthetic pipeline, not pretrained weights. It is retained as investigation evidence, not proposed as additional committed CI coverage.

## Integrity

Run `sha256sum -c SHA256SUMS` from this directory to verify all packaged files. The archive excludes weights, credentials, download logs, cache files, old drafts and duplicate patch bundles.

AI assistance: Codex prepared the investigation, fix, tests and packaging. The PR remains a draft for human review.
