# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Close a real speech HTTP connection while native OmniVoice runs in mp mode.

All model computation, serving routes, abort propagation and cleanup are native.
LifecycleObserver/Engine only record execution and observe the idle boundary.
"""

import argparse
import asyncio
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
import numpy as np
import soundfile as sf
import yaml


def events(root):
    records = []
    for path in root.glob("rank-*.jsonl"):
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A worker may currently be writing the final line.
    return records


async def wait_for(predicate, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.05)
    raise TimeoutError("Timed out waiting for native serving evidence")


async def run(args):
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo).resolve()
    config = yaml.safe_load((repo / "vllm_omni/deploy/omnivoice.yaml").read_text())
    config["stages"][0].update(
        step_execution=True,
        max_num_seqs=2,
        worker_extension_cls="lifecycle_observer.LifecycleObserver",
        engine_backend="lifecycle_observer.LifecycleEngine",
    )
    deploy = root / "deploy.yaml"
    deploy.write_text(yaml.safe_dump(config, sort_keys=False))
    command = [
        sys.executable,
        "-m",
        "vllm_omni.entrypoints.cli.main",
        "serve",
        args.model,
        "--omni",
        "--deploy-config",
        str(deploy),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "lifecycle-omnivoice",
        "--trust-remote-code",
    ]
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join([str(Path(__file__).parent), str(repo), env.get("PYTHONPATH", "")]),
        LIFECYCLE_EVIDENCE_DIR=str(root),
        PYTHONDONTWRITEBYTECODE="1",
        VLLM_LOGGING_COLOR="0",
    )
    source_patch = subprocess.check_output(["git", "diff"], cwd=repo)
    (root / "source.patch").write_bytes(source_patch)
    manifest = {
        "command": command,
        "base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "diff_sha256": hashlib.sha256(source_patch).hexdigest(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (root / "server.log").open("w") as log:
        server = subprocess.Popen(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        writer = None
        try:
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{args.port}", timeout=300, trust_env=False
            ) as client:
                deadline = time.monotonic() + 300
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

                warmup_text = "Hello, this is a lifecycle warmup."
                response = await client.post(
                    "/v1/audio/speech",
                    json={
                        "model": "lifecycle-omnivoice",
                        "input": warmup_text,
                        "response_format": "wav",
                        "seed": 42,
                        "extra_params": {"num_inference_steps": 4},
                    },
                )
                assert response.status_code == 200, response.text
                audio, rate = sf.read(io.BytesIO(response.content))
                assert audio.size > 0 and np.isfinite(audio).all()
                warmup_steps = await wait_for(
                    lambda: [
                        e
                        for e in events(root)
                        if e["event"] == "step"
                        and any(warmup_text in prompt for prompt in e.get("new_prompts", {}).values())
                    ]
                )
                warmup_id = next(
                    rid for rid, prompt in warmup_steps[-1]["new_prompts"].items() if warmup_text in prompt
                )
                warmup = await wait_for(
                    lambda: [
                        e
                        for e in events(root)
                        if e["event"] == "snapshot"
                        and e["time"] >= warmup_steps[-1]["time"]
                        and any(ref["request_id"] == warmup_id for ref in e["refs"].values())
                    ]
                )
                warmup = warmup[-1]
                assert not warmup["cache"] and not warmup["batch_states"] and not warmup["lora"]

                marker = "Please cancel this speech while the model is still generating."
                body = json.dumps(
                    {
                        "model": "lifecycle-omnivoice",
                        "input": marker,
                        "response_format": "wav",
                        "seed": 42,
                        "extra_params": {"num_inference_steps": args.steps},
                    }
                ).encode()
                _, writer = await asyncio.open_connection("127.0.0.1", args.port)
                writer.write(
                    (
                        f"POST /v1/audio/speech HTTP/1.1\r\nHost: 127.0.0.1:{args.port}\r\n"
                        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
                    ).encode()
                    + body
                )
                await writer.drain()
                started = await wait_for(
                    lambda: [
                        e
                        for e in events(root)
                        if e["event"] == "step"
                        and any(marker in prompt for prompt in e.get("new_prompts", {}).values())
                    ]
                )
                request_id = next(rid for rid, prompt in started[-1]["new_prompts"].items() if marker in prompt)
                assert request_id in started[-1]["cache"], "Request already ended before disconnect"
                disconnected_at = time.time()
                writer.close()
                await writer.wait_closed()
                writer = None
                releases = await wait_for(
                    lambda: [e for e in events(root) if e["event"] == "release" and request_id in e["request_ids"]]
                )
                assert releases[-1]["time"] >= disconnected_at
                assert request_id in releases[-1]["cache_before"], "Abort arrived after request already completed"
                snapshots = await wait_for(
                    lambda: [e for e in events(root) if e["event"] == "snapshot" and e["time"] >= releases[-1]["time"]]
                )
                idle = snapshots[-1]
                assert not idle["cache"] and not idle["batch_states"] and not idle["lora"], idle
                refs = [value for value in idle["refs"].values() if value["request_id"] == request_id]
                assert refs and all(not value["state_alive"] for value in refs)
                assert all(
                    not tensor["alive"]
                    for value in refs
                    for tensor in value["tensors"].values()
                    if tensor["request_private"]
                )
                result = {
                    "frontend": "native POST /v1/audio/speech, TCP connection closed after first step",
                    "worker_request_id": request_id,
                    "warmup_audio_samples": int(audio.size),
                    "warmup_sample_rate": rate,
                    "warmup_idle": warmup,
                    "cancel_idle": idle,
                    "disconnect_to_release_seconds": releases[-1]["time"] - disconnected_at,
                    "release": releases[-1],
                    "allocated_change": idle["allocated"] - warmup["allocated"],
                    "observed_before_shutdown": True,
                }
                (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                print("RESULT " + json.dumps(result), flush=True)
        finally:
            if writer is not None:
                writer.close()
                await writer.wait_closed()
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGINT)
                try:
                    await asyncio.to_thread(server.wait, 30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    await asyncio.to_thread(server.wait)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/work/models/OmniVoice")
    parser.add_argument("--repo", default="/work/vllm-omni")
    parser.add_argument("--port", type=int, default=18719)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--output-dir", required=True)
    asyncio.run(run(parser.parse_args()))
