# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Test-only worker observation: delegate all computation and cleanup unchanged.

Set worker_extension_cls='lifecycle_observer.LifecycleObserver' and
LIFECYCLE_EVIDENCE_DIR to a writable directory visible to all worker ranks.
The lifecycle_snapshot RPC synchronizes GPU work, reads state, and gathers only
scalar metadata over the existing CPU group. It never schedules or cleans up.
"""

import json
import os
import time
import weakref
from pathlib import Path

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.parallel_state import get_world_group
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.platforms import current_omni_platform


def _tensor_refs(value, prefix, depth=0):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif depth < 4 and isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_refs(item, f"{prefix}.{key}", depth + 1)
    elif depth < 4 and isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensor_refs(item, f"{prefix}.{index}", depth + 1)


class LifecycleObserver:
    def _lifecycle_init(self):
        if hasattr(self, "_lifecycle_refs"):
            return
        self._lifecycle_refs = {}
        self._lifecycle_ref_sequence = 0
        self._lifecycle_steps = 0
        self._lifecycle_prepare_times = []
        self._lifecycle_pack_times = []
        self._lifecycle_root = Path(os.environ["LIFECYCLE_EVIDENCE_DIR"])
        self._lifecycle_root.mkdir(parents=True, exist_ok=True)
        # Timing wrapper around the framework's existing packer, not model math.
        prepare = self.model_runner._prepare_batch_inputs

        def observed_prepare(*args, **kwargs):
            started = time.perf_counter()
            try:
                return prepare(*args, **kwargs)
            finally:
                self._lifecycle_prepare_times.append(time.perf_counter() - started)

        self.model_runner._prepare_batch_inputs = observed_prepare
        make_batch = InputBatch.make_batch.__func__

        def observed_make_batch(cls, *args, **kwargs):
            started = time.perf_counter()
            try:
                return make_batch(cls, *args, **kwargs)
            finally:
                self._lifecycle_pack_times.append(time.perf_counter() - started)

        InputBatch.make_batch = classmethod(observed_make_batch)

    def _lifecycle_record(self, payload):
        record = {"rank": dist.get_rank(), "pid": os.getpid(), "time": time.time(), **payload}
        with (self._lifecycle_root / f"rank-{dist.get_rank()}.jsonl").open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def execute_stepwise(self, scheduler_output):
        self._lifecycle_init()
        started = time.perf_counter()
        pack_count = len(self._lifecycle_pack_times)
        output = super().execute_stepwise(scheduler_output)
        self._lifecycle_steps += 1
        for request_id, state in self.model_runner.state_cache.items():
            if any(entry["state"]() is state for entry in self._lifecycle_refs.values()):
                continue
            self._lifecycle_ref_sequence += 1
            key = f"{request_id}@observation-{self._lifecycle_ref_sequence}"
            tensors = {}
            for field in ("extra", "prompt_embeds", "negative_prompt_embeds", "timesteps"):
                for name, tensor in _tensor_refs(getattr(state, field, None), field):
                    tensors[name] = {
                        "ref": weakref.ref(tensor),
                        "device": str(tensor.device),
                        "bytes": tensor.untyped_storage().nbytes(),
                        "shape": list(tensor.shape),
                        "request_private": field != "timesteps",
                    }
            self._lifecycle_refs[key] = {"id": request_id, "state": weakref.ref(state), "tensors": tensors}
        self._lifecycle_record({
            "event": "step",
            "step_id": scheduler_output.step_id,
            "scheduled": list(scheduler_output.scheduled_request_ids),
            "cache": list(self.model_runner.state_cache),
            "lora": list(self._step_lora_state),
            "seconds": time.perf_counter() - started,
            "prepare_seconds": self._lifecycle_prepare_times[-1] if self._lifecycle_prepare_times else None,
            "make_batch_seconds": sum(self._lifecycle_pack_times[pack_count:]),
            "make_batch_calls": len(self._lifecycle_pack_times) - pack_count,
            "new_prompts": {
                req.request_id: str(req.req.prompt)[:600] for req in scheduler_output.scheduled_new_reqs
            },
        })
        return output

    def lifecycle_snapshot(self, label="idle"):
        self._lifecycle_init()
        current_omni_platform.synchronize()
        runner = self.model_runner
        batch = runner.input_batch
        refs = {
            key: {
                "request_id": entry["id"],
                "state_alive": entry["state"]() is not None,
                "tensors": {
                    name: {k: v for k, v in tensor.items() if k != "ref"}
                    | {"alive": tensor["ref"]() is not None}
                    for name, tensor in entry["tensors"].items()
                },
            }
            for key, entry in self._lifecycle_refs.items()
        }
        snapshot = {
            "event": "snapshot", "label": label, "rank": dist.get_rank(), "pid": os.getpid(),
            "cache": list(runner.state_cache),
            "batch_ids": list(batch.request_ids) if batch is not None else [],
            "batch_states": [state.request_id for state in batch.states] if batch is not None else [],
            "lora": list(self._step_lora_state),
            "refs": refs,
            "allocated": torch.accelerator.memory_allocated(),
            "reserved": current_omni_platform.memory_reserved(),
            "steps": self._lifecycle_steps,
            "prepare_seconds_total": sum(self._lifecycle_prepare_times),
            "make_batch_seconds_total": sum(self._lifecycle_pack_times),
            "make_batch_calls": len(self._lifecycle_pack_times),
        }
        self._lifecycle_record(snapshot)
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, snapshot, group=get_world_group().cpu_group)
        return gathered

    def release_step_requests(self, request_ids):
        self._lifecycle_init()
        cache_before = list(self.model_runner.state_cache)
        result = super().release_step_requests(request_ids)
        self._lifecycle_record({
            "event": "release", "request_ids": list(request_ids),
            "cache_before": cache_before,
            "cache": list(self.model_runner.state_cache), "lora": list(self._step_lora_state),
        })
        return result


# Optional serving observer: take a worker snapshot only when the real Engine
# reaches its idle wait. It adds no scheduler/Runner call and changes no cleanup.
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine


class LifecycleEngine(DiffusionEngine):
    def _init_runtime_state(self):
        super()._init_runtime_state()
        self._lifecycle_wave = 0
        self._lifecycle_last_idle_wave = -1
        wait = self._cv.wait

        def observed_wait(timeout=None):
            if (not self.scheduler.has_requests() and self._loop_started
                    and self._lifecycle_last_idle_wave != self._lifecycle_wave):
                self._lifecycle_last_idle_wave = self._lifecycle_wave
                self.executor.collective_rpc(
                    "lifecycle_snapshot", timeout=120,
                    args=(f"engine-idle-wave-{self._lifecycle_wave}",),
                )
            return wait(timeout)

        self._cv.wait = observed_wait

    def _emit_outputs(self, *args, **kwargs):
        result = super()._emit_outputs(*args, **kwargs)
        self._lifecycle_wave += 1
        return result
