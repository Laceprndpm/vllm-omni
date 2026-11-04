"""Trace one fixed diagnostic case per variant, outside the primary endpoint."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=('reference', 'baseline', 'fixed'), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'plan.json').read_text())
    assert hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest() == (root / 'plan.sha256').read_text().split()[0]
    variant = plan['variants'][args.variant]
    source = (root / variant['source']).resolve()
    import vllm_omni
    assert Path(vllm_omni.__file__).resolve().parent.parent == source
    output = root / 'dispatch-probes' / args.variant
    omni = Omni(
        model='/root/autodl-fs/omni-9826/models/LongCat-Image-Edit',
        cache_backend=variant['cache_backend'], cache_config=plan['cache_config'],
        enable_cache_dit_summary=variant['cache_backend'] is not None,
        enable_diffusion_pipeline_profiler=True,
        enforce_eager=True, init_timeout=1800,
        worker_extension_cls='probe_worker.ProbeWorker',
    )
    try:
        omni.engine.collective_rpc('dispatch_probe_install', args=(str(output),), stage_ids=[0])
        with Image.open(root / 'inputs/cat.png') as image:
            image = image.convert('RGB')
        prompt = {'prompt': plan['tasks'][0]['prompt'], 'negative_prompt': '', 'multi_modal_data': {'image': image}}
        params = OmniDiffusionSamplingParams(seed=7, guidance_scale=1.0, num_inference_steps=2, num_outputs_per_prompt=1, width=1024, height=1024)
        results = omni.generate(prompt, sampling_params_list=[params], use_tqdm=False)
        images = []
        for result in results:
            if getattr(result, 'images', None):
                images = result.images
                break
        if not images:
            images = extract_images_from_outputs(results)
        assert len(images) == 1
        images[0].save(output / 'diagnostic.png')
        (output / 'pixel.sha256').write_text(hashlib.sha256(images[0].tobytes()).hexdigest() + '\n')
        (output / 'COMPLETE').touch()
        print('DISPATCH_PROBE_COMPLETE', args.variant, flush=True)
    finally:
        omni.close()


if __name__ == '__main__':
    main()
