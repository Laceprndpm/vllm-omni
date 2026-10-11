# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real pretrained pipeline + mp Engine lifecycle evidence; no model replacement."""

import argparse
import asyncio
import hashlib
import json
import multiprocessing
import os
import statistics
import threading
import time
from pathlib import Path

import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


async def main(args):
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["LIFECYCLE_EVIDENCE_DIR"] = str(root)
    config = OmniDiffusionConfig(
        model=args.model, model_class_name=args.model_class,
        distributed_executor_backend="mp", step_execution=True,
        engine_backend="lifecycle_observer.LifecycleEngine",
        worker_extension_cls="lifecycle_observer.LifecycleObserver",
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=args.tp),
        num_gpus=args.tp, dtype=getattr(torch, args.dtype), enforce_eager=True, max_num_seqs=2,
    )
    config.enrich_config()
    engine = DiffusionEngine.make_engine(config)
    idle = threading.Event()
    wait, emit = engine._cv.wait, engine._emit_outputs
    cancel_targets, cancelled, waves = set(), set(), []

    def observe_wait(timeout=None):
        if not engine.scheduler.has_requests():
            idle.set()
        return wait(timeout)

    def observe_emit(finished_ids, scheduled_ids, runner_output, **kwargs):
        emit(finished_ids, scheduled_ids, runner_output, **kwargs)
        waves.append(list(scheduled_ids))
        for rid in cancel_targets.intersection(scheduled_ids) - cancelled:
            if rid not in finished_ids:
                engine.abort(rid)
                cancelled.add(rid)

    engine._cv.wait, engine._emit_outputs = observe_wait, observe_emit

    def add(rid, seed=42):
        prompt = "A ceramic teapot on a wooden table."
        if args.model_class == "OmniVoicePipeline":
            prompt = "Hello, this is a lifecycle test."
        else:
            prompt = {"prompt": prompt, "negative_prompt": "blurry", "modalities": ["image"]}
        engine.add_request(OmniDiffusionRequest(
            request_id=rid, prompt=prompt,
            sampling_params=OmniDiffusionSamplingParams(
                height=512, width=512, num_inference_steps=args.steps,
                true_cfg_scale=4.0, seed=seed, extra_args={"seed": seed},
            ),
        ))

    async def consume(rid):
        final = None
        async for output in engine.get_output_stream(rid):
            if not output.finished:
                continue
            final = {"id": rid, "finished": True, "aborted": output.aborted, "error": output.error}
            if output.error is None and not output.aborted:
                tensor = output.output
                if not isinstance(tensor, torch.Tensor):
                    raise AssertionError(f"Unexpected native output type {type(tensor)}")
                array = tensor.detach().float().cpu()
                final.update(
                    shape=list(array.shape), finite=bool(torch.isfinite(array).all()),
                    samples=array.numel(), mean=float(array.mean()), std=float(array.std()),
                    sha256=hashlib.sha256(array.numpy().tobytes()).hexdigest(),
                )
                # Keep a CPU-only correctness artifact; never retain request/device tensors.
                torch.save(array, root / f"{rid}.pt")
        return final

    async def run_case(label, ids, aborted=()):
        cancel_targets.clear()
        cancel_targets.update(aborted)
        waves.clear()
        started = time.perf_counter()
        # Queue both requests before allowing Scheduler to form the first wave.
        with engine._cv:
            idle.clear()
            for rid in ids:
                add(rid)
        tasks = [asyncio.create_task(consume(rid)) for rid in ids]
        await engine._check_and_start_background_loop()
        reached_idle = await asyncio.to_thread(idle.wait, args.timeout)
        if not reached_idle:
            raise TimeoutError(f"{label}: engine failed to reach idle")
        with engine._cv:
            if engine.scheduler.has_requests():
                raise AssertionError("idle event did not match scheduler")
        done, pending = await asyncio.wait(tasks, timeout=30)
        results = [task.result() for task in tasks if task in done]
        snapshot = await asyncio.to_thread(engine.collective_rpc, "lifecycle_snapshot", 120, (label,))
        if snapshot and isinstance(snapshot[0], list):
            if not all(reply == snapshot[0] for reply in snapshot):
                raise AssertionError("Worker snapshot replies disagree")
            snapshot = snapshot[0]
        errors = []
        for rank in snapshot:
            if any(rank[key] for key in ("cache", "batch_ids", "batch_states", "lora")):
                errors.append(f"rank {rank['rank']} retains request metadata")
            for reference in rank["refs"].values():
                if reference["state_alive"] or any(
                    t["alive"] for t in reference["tensors"].values() if t["request_private"]
                ):
                    errors.append(f"rank {rank['rank']} retains {reference['request_id']}")
        if pending:
            errors.append("terminal consumer pending at idle")
        for result in results:
            if not result or result["error"] or result["aborted"] != (result["id"] in aborted):
                errors.append(f"wrong terminal output: {result}")
            elif not result["aborted"] and (not result["finite"] or not result["samples"]):
                errors.append(f"invalid native tensor: {result}")
        if label in ("survivor", "warmup-b2") and not any(set(ids).issubset(wave) for wave in waves):
            errors.append("requests never shared an execution wave")
        report = {
            "case": label, "seconds": time.perf_counter() - started,
            "cancelled": sorted(set(ids) & cancelled), "results": results,
            "batch_sizes": [len(wave) for wave in waves], "ranks": snapshot,
            "errors": errors,
        }
        with (root / "cases.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        print("CASE " + json.dumps({k: v for k, v in report.items() if k != "ranks"}), flush=True)
        for task in pending:
            task.cancel()  # Only after preserving the idle evidence.
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        return report

    reports = []
    try:
        await run_case("warmup-b2", ["warmup-a", "warmup-b"])
        reports.append(await run_case("completion", ["completion"]))
        reports.append(await run_case("last-abort", ["last-abort"], ["last-abort"]))
        reports.append(await run_case("survivor", ["cancel-a", "survivor-b"], ["cancel-a"]))
        # Compare the survivor with an independently completed request with the same seed.
        first = torch.load(root / "completion.pt", weights_only=True)
        survivor = torch.load(root / "survivor-b.pt", weights_only=True)
        difference = (first - survivor).abs()
        print("SURVIVOR_PARITY " + json.dumps({
            "equal": bool(torch.equal(first, survivor)),
            "max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
        }), flush=True)
        for index in range(6):
            rid = f"cycle-{index}"
            reports.append(await run_case(rid, [rid], [rid] if index % 2 else []))
        if args.c32:
            for round_index in range(args.c32_rounds):
                ids = [f"c32-r{round_index}-{index:02}" for index in range(32)]
                reports.append(await run_case(f"c32-capacity-2-round-{round_index}", ids, ids[::2]))
        traces = [json.loads(line) for line in (root / "rank-0.jsonl").read_text().splitlines()]
        prep = [r["prepare_seconds"] for r in traces if r["event"] == "step" and r["prepare_seconds"] is not None]
        pack = [r["make_batch_seconds"] for r in traces if r["event"] == "step" and r["make_batch_calls"]]
        summary = {
            "model": args.model, "model_class": args.model_class, "tp": args.tp,
            "steps": args.steps, "max_num_seqs": 2, "width": 512, "height": 512,
            "dtype": args.dtype, "backend": "mp", "eager": True,
            "failed_cases": [r["case"] for r in reports if r["errors"]],
            "prepare_dispatch_seconds_p50": statistics.median(prep) if prep else None,
            "prepare_dispatch_seconds_total": sum(prep),
            "make_batch_host_seconds_p50": statistics.median(pack) if pack else None,
            "make_batch_host_seconds_total": sum(pack),
            "timing_scope": "host wall time around framework calls; not CUDA kernel duration",
        }
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print("SUMMARY " + json.dumps(summary), flush=True)
        return bool(summary["failed_cases"])
    finally:
        engine.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-class", default="QwenImagePipeline")
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--c32", action="store_true")
    parser.add_argument("--c32-rounds", type=int, default=3)
    raise SystemExit(asyncio.run(main(parser.parse_args())))
