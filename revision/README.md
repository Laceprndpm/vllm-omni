# Step request lifecycle: review follow-up evidence

Supplement to [vllm-project/vllm-omni#8719](https://github.com/vllm-project/vllm-omni/pull/8719).

- Reviewed head: `166a5b5b58549b84e92dd45eb2d9d52222dbb411`.
- Follow-up: `45063b541c5de050a1525b0e350e3382c1a45120`.
- [Download the evidence ZIP](../revision-evidence.zip): architecture/report, exact patch, failing/passing CPU logs, native GPU and HTTP scripts, per-rank traces, allocator counters, output hashes and manifests. It excludes weights and generated output arrays.
- [Browse the observation/reproduction scripts](scripts). Native model encode/denoise/decode are preserved.

CPU selections: **284 passed, 4 deselected**. Native OmniVoice mp (one RTX 4060 Laptop GPU) and Qwen-Image mp/TP2 (two A100-SXM4 80GB) each cover 13 idle boundaries per version, including last cancellation, a surviving batched request, repeated idle/resume and three C32 mixed rounds. C32 is 32 outstanding submissions with execution capacity two. All 55 corresponding completed outputs per model match before/after. Real audio and image serving TCP disconnects also exercise cancellation.

This revision fixes reused-ID terminal notification and Worker bookkeeping. Both compared revisions already contain the original Runner cleanup, so equal GPU allocations do not imply additional memory savings. The original unfixed baseline and its bounded 22.5 KiB tail retention remain documented in the earlier evidence below this directory.

Unzip the archive into a source checkout to access its self-contained `step-lifecycle-revision/README.md`, exact commands and `SHA256SUMS`. Use fresh output directories for every run. CPU tests require Python 3.12, vLLM 0.31.0 and matching project test dependencies; native scripts additionally need the specified hardware and complete pretrained weights. Full environment limitations and pre-commit failures already present on the baseline are included in the reports.

Archive SHA256: `040eea934d20b7dc83b460302bcdc6926bc5fcdc060e1242dadced97fd021018`
