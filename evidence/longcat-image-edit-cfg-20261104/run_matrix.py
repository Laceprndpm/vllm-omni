"""Run one frozen SSIM study variant in a fresh Omni process."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def one_worker(value):
    while isinstance(value, list):
        assert len(value) == 1
        value = value[0]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['reference', 'baseline', 'fixed'], required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'plan.json').read_text())
    assert hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest() == (root / 'plan.sha256').read_text().split()[0]
    variant = plan['variants'][args.variant]
    source = (root / variant['source']).resolve()
    hashes = plan['source_file_sha256']['fixed' if args.variant == 'fixed' else 'baseline']
    for relative, expected in hashes.items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected, relative

    import numpy as np
    import torch
    import vllm_omni
    from PIL import Image
    from vllm_omni.diffusion.cache import cachedit
    from vllm_omni.diffusion.models.longcat_image.pipeline_longcat_image_edit import LongCatImageEditPipeline
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs

    assert Path(vllm_omni.__file__).resolve().parent.parent == source
    assert hasattr(cachedit, 'set_separate_cfg') is (args.variant == 'fixed')
    assert LongCatImageEditPipeline.__name__ == 'LongCatImageEditPipeline'
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    for asset in plan['inputs'].values():
        path = root / asset['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256']
        with Image.open(path) as image:
            assert list(image.size) == asset['original_size']
            image.verify()
    expected_versions = {'vllm': '0.33.0', 'torch': '2.14.0', 'diffusers': '0.40.0', 'transformers': '5.19.0', 'cache-dit': '1.5.0'}
    versions = {name: importlib.metadata.version(name) for name in [*expected_versions, 'torchvision', 'numpy', 'scikit-image', 'torchmetrics', 'matplotlib']}
    for name, expected in expected_versions.items():
        assert versions[name] == expected, (name, versions[name])
    model = Path('/root/autodl-fs/omni-9826/models/LongCat-Image-Edit')
    assert (model / 'model_index.json').exists()
    metadata = {
        'variant': args.variant,
        'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_root': str(source),
        'source_file_sha256': hashes,
        'base_commit': plan['base_commit'],
        'fixed_patch_sha256': plan['fix_patch_sha256'] if args.variant == 'fixed' else None,
        'plan_sha256': hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest(),
        'versions': versions,
        'torch_cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip(),
        'model_path': str(model),
        'model_revision': plan['model_revision'],
        'cache_backend': variant['cache_backend'],
        'cache_config': plan['cache_config'],
        'enforce_eager': True,
        'input_conversion': 'PIL RGB, then normal pipeline image preprocessing at requested 1024x1024. No generated image resizing.',
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
        'timing_scope': 'omni.generate only; excludes initialization, warmups, diagnostic RPCs and PNG writes',
    }
    write_json(root / f'preflight-{args.variant}.json', metadata)
    if args.preflight:
        print('PREFLIGHT_OK', args.variant, str(source), flush=True)
        return

    output = root / 'outputs' / args.variant
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'environment.json', metadata)
    tasks = {task['id']: task for task in plan['tasks']}
    jobs = []
    primary = {}
    for round_index, seed in enumerate(plan['seeds'], 1):
        jobs.append({**tasks['cat_to_dog'], 'job_id': f'warmup_r{round_index}', 'role': 'warmup', 'round': round_index, 'seed': 0, 'steps': 2, 'guidance': 4.5})
        for task in plan['tasks']:
            job = {**task, 'job_id': f"{task['id']}_s{seed}", 'role': 'primary', 'round': round_index, 'seed': seed, 'steps': plan['num_inference_steps'], 'guidance': plan['guidance_scale']}
            jobs.append(job)
            primary[job['job_id']] = job
    for name in plan['repeats']:
        jobs.append({**primary[name], 'job_id': name + '_repeat', 'role': 'repeat', 'repeat_of': name})

    omni = Omni(
        model=str(model), cache_backend=variant['cache_backend'], cache_config=plan['cache_config'],
        enable_cache_dit_summary=variant['cache_backend'] is not None,
        enforce_eager=True, enable_diffusion_pipeline_profiler=True, init_timeout=1800,
        worker_extension_cls='matrix_worker.MatrixAuditWorker',
    )
    records = []
    try:
        runtime = one_worker(omni.engine.collective_rpc('matrix_metadata', stage_ids=[0]))
        assert Path(runtime['pipeline_source_path']).is_relative_to(source)
        assert runtime['pipeline_source_sha256'] == hashes['vllm_omni/diffusion/models/longcat_image/pipeline_longcat_image_edit.py']
        metadata['worker_runtime'] = runtime
        write_json(output / 'environment.json', metadata)
        for job in jobs:
            omni.engine.collective_rpc('cache_audit_start', stage_ids=[0])
            with Image.open(root / plan['inputs'][job['input']]['path']) as input_image:
                input_image = input_image.convert('RGB')
            prompt = {'prompt': job['prompt'], 'negative_prompt': plan['negative_prompt'], 'multi_modal_data': {'image': input_image}}
            params = OmniDiffusionSamplingParams(seed=job['seed'], guidance_scale=job['guidance'], num_inference_steps=job['steps'], num_outputs_per_prompt=1, width=plan['width'], height=plan['height'])
            print('MATRIX_START', json.dumps(job), flush=True)
            started = time.perf_counter()
            outputs = omni.generate(prompt, sampling_params_list=[params], use_tqdm=False)
            wall_seconds = time.perf_counter() - started
            state = one_worker(omni.engine.collective_rpc('matrix_snapshot', stage_ids=[0]))
            cfg = job['guidance'] > 1
            assert state['pipeline_num_timesteps'] == job['steps']
            assert state['observed_transformer_forwards'] == job['steps'] * (2 if cfg else 1)
            assert len(state['contexts']) == (2 if variant['cache_backend'] else 0)
            expected_steps = job['steps'] // 2 if args.variant == 'baseline' and not cfg else job['steps']
            for context in state['contexts']:
                assert context['executed_steps'] == expected_steps
                assert context['num_inference_steps'] == job['steps']
                assert context['enable_separate_cfg'] is (args.variant == 'baseline' or cfg)
            images = []
            for result in outputs:
                if getattr(result, 'images', None):
                    images = result.images
                    break
            if not images:
                images = extract_images_from_outputs(outputs)
            assert len(images) == 1
            image = images[0]
            assert image.size == (plan['width'], plan['height'])
            assert image.mode == 'RGB'
            path = output / f"{job['job_id']}.png"
            image.save(path)
            pixels = np.asarray(image)
            record = {
                **job, 'negative_prompt': plan['negative_prompt'], 'input_sha256': plan['inputs'][job['input']]['sha256'],
                'wall_seconds': wall_seconds, 'image_path': str(path), 'image_size': list(image.size),
                'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(), 'image_file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'pixel_mean': float(pixels.mean()), 'pixel_std': float(pixels.std()), 'audit': state,
            }
            records.append(record)
            write_json(output / 'results.json', records)
            print('MATRIX_END', json.dumps({'variant': args.variant, 'job_id': job['job_id'], 'role': job['role'], 'seconds': wall_seconds, 'completed': len(records), 'total': len(jobs)}), flush=True)
    finally:
        omni.close()
    (output / 'COMPLETE').touch()
    print('MATRIX_COMPLETE', args.variant, flush=True)


if __name__ == '__main__':
    main()
