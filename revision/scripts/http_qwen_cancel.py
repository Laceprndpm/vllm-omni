# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native Qwen TP2 serving: close one executing image request's HTTP socket."""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image


def events(root):
    records = []
    for path in root.glob("rank-*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


async def wait_for(predicate, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.05)
    raise TimeoutError("Timed out waiting for native serving evidence")


async def run(args):
    root, repo = Path(args.output_dir).resolve(), Path(args.repo).resolve()
    root.mkdir(parents=True, exist_ok=True)
    overrides = {"0": {
        "devices": "0,1", "max_num_seqs": 2,
        "worker_extension_cls": "lifecycle_observer.LifecycleObserver",
        "engine_backend": "lifecycle_observer.LifecycleEngine",
    }}
    command = [
        sys.executable, "-m", "vllm_omni.entrypoints.cli.main", "serve", args.model,
        "--omni", "--host", "127.0.0.1", "--port", str(args.port),
        "--served-model-name", "lifecycle-qwen", "--distributed-executor-backend", "mp",
        "--tensor-parallel-size", "2", "--step-execution", "--max-num-seqs", "2",
        "--dtype", "bfloat16", "--enforce-eager", "--stage-overrides", json.dumps(overrides),
    ]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join([str(Path(__file__).parent), str(repo), env.get("PYTHONPATH", "")]),
        LIFECYCLE_EVIDENCE_DIR=str(root), PYTHONDONTWRITEBYTECODE="1", VLLM_LOGGING_COLOR="0",
        VLLM_WORKER_MULTIPROC_METHOD="spawn", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    )
    files = ["diffusion_engine.py", "sched/base_scheduler.py", "worker/diffusion_worker.py"]
    manifest = {"command": command, "source": {
        file: hashlib.sha256((repo / "vllm_omni/diffusion" / file).read_bytes()).hexdigest() for file in files
    }}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    payload = {
        "model": "lifecycle-qwen", "prompt": "A ceramic teapot on a wooden table.",
        "negative_prompt": "blurry", "size": "512x512", "num_inference_steps": args.steps,
        "true_cfg_scale": 4.0, "seed": 42, "n": 1, "response_format": "b64_json",
    }
    with (root / "server.log").open("w") as log:
        server = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        writer = None
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{args.port}", timeout=600,
                                         trust_env=False) as client:
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"Native server exited: {server.returncode}; see server.log")
                    try:
                        if (await client.get("/health")).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.5)
                else:
                    raise TimeoutError("Server did not become healthy")
                response = await client.post("/v1/images/generations", json=payload)
                assert response.status_code == 200, response.text
                encoded = response.json()["data"][0]["b64_json"]
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                    assert image.size == (512, 512), image.size
                    image.save(root / "warmup.png")
                warmup_ids = {rid for e in events(root) if e["event"] == "step"
                              for rid, prompt in e.get("new_prompts", {}).items() if payload["prompt"] in prompt}

                def warmup_snapshot(e):
                    return e["event"] == "snapshot" and any(
                        ref["request_id"] in warmup_ids for ref in e["refs"].values()
                    )

                warmup = await wait_for(lambda: {
                    rank: [e for e in events(root) if warmup_snapshot(e) and e["rank"] == rank][-1]
                    for rank in (0, 1)
                } if all(any(warmup_snapshot(e) and e["rank"] == rank for e in events(root))
                         for rank in (0, 1)) else None)
                marker = "A red teapot beside a green book, lifecycle disconnect probe."
                body = json.dumps(payload | {"prompt": marker}).encode()
                _, writer = await asyncio.open_connection("127.0.0.1", args.port)
                writer.write((
                    f"POST /v1/images/generations HTTP/1.1\r\nHost: 127.0.0.1:{args.port}\r\n"
                    f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
                ).encode() + body)
                await writer.drain()
                started = await wait_for(lambda: [e for e in events(root) if e["event"] == "step"
                    and any(marker in prompt for prompt in e.get("new_prompts", {}).values())])
                request_id = next(rid for rid, prompt in started[-1]["new_prompts"].items() if marker in prompt)
                assert request_id in started[-1]["cache"], "Request ended before disconnect"
                disconnected_at = time.time()
                writer.close()
                await writer.wait_closed()
                writer = None
                releases = await wait_for(lambda: [e for e in events(root)
                    if e["event"] == "release" and request_id in e["request_ids"]]
                    if len({e["rank"] for e in events(root)
                        if e["event"] == "release" and request_id in e["request_ids"]}) == 2 else None)
                last_release = max(e["time"] for e in releases)
                assert all(request_id in e["cache_before"] and e["time"] >= disconnected_at for e in releases)
                snapshots = await wait_for(lambda: [e for e in events(root)
                    if e["event"] == "snapshot" and e["time"] >= last_release]
                    if len({e["rank"] for e in events(root)
                        if e["event"] == "snapshot" and e["time"] >= last_release}) == 2 else None)
                idle = {rank: [e for e in snapshots if e["rank"] == rank][-1] for rank in (0, 1)}
                errors = []
                for rank, snapshot in idle.items():
                    if any(snapshot[k] for k in ("cache", "batch_states", "batch_ids", "lora")):
                        errors.append(f"rank {rank}: retained metadata")
                    refs = [ref for ref in snapshot["refs"].values() if ref["request_id"] == request_id]
                    if not refs or any(ref["state_alive"] for ref in refs):
                        errors.append(f"rank {rank}: missing observation or live state")
                    if any(t["alive"] for ref in refs for t in ref["tensors"].values() if t["request_private"]):
                        errors.append(f"rank {rank}: live request tensor")
                result = {
                    "frontend": "native POST /v1/images/generations, TCP closed after first step",
                    "request_id": request_id, "steps": args.steps, "max_num_seqs": 2,
                    "warmup_idle": warmup, "cancel_idle": idle,
                    "disconnect_to_release_seconds": last_release - disconnected_at,
                    "allocated_change": {rank: snap["allocated"] - warmup[rank]["allocated"]
                                         for rank, snap in idle.items()},
                    "observed_before_shutdown": True, "errors": errors,
                }
                (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                print("RESULT " + json.dumps(result), flush=True)
                return bool(errors)
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
                try:
                    await asyncio.to_thread(server.wait, 45)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    await asyncio.to_thread(server.wait)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
