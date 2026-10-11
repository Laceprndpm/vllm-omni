# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Freeze source/script identities and environment before a real model run."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root, repo = Path(args.output_dir), Path(args.repo)
    root.mkdir(parents=True, exist_ok=False)
    scripts = root / "scripts"
    scripts.mkdir()
    hashes = {}
    for name in ("lifecycle_observer.py", "native_lifecycle.py", "http_qwen_cancel.py", "evidence_manifest.py"):
        source = Path(__file__).with_name(name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, scripts / name)
    versions = {}
    for name in ("torch", "vllm", "transformers", "diffusers", "flashinfer-python", "flashinfer-cubin"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    production = {}
    for relative in ("diffusion_engine.py", "sched/base_scheduler.py", "worker/diffusion_worker.py", "worker/diffusion_model_runner.py"):
        source = repo / "vllm_omni/diffusion" / relative
        production[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "phase": args.phase,
        "before_commit": "166a5b5b58549b84e92dd45eb2d9d52222dbb411",
        "after_commit": "45063b541c5de050a1525b0e350e3382c1a45120",
        "applied_patch_sha256": ("1ae8c374450ba04b67c91f43b87736661107384f1ef257f6ce7da94dd5ed5716"
                                  if args.phase == "after" else None),
        "model_revision": "75e0b4be04f60ec59a75f475837eced720f823b6",
        "scripts_sha256": hashes, "production_sha256": production,
        "python": sys.version, "platform": platform.platform(), "packages": versions,
        "hardware": subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv,noheader"
        ], text=True).strip().splitlines(),
        "command": args.command,
        "scope": "Native pretrained model, real mp Engine/Workers; read-only idle snapshot RPC. No future request or shutdown used for cleanup.",
    }
    (root / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
