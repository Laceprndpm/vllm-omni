"""Supplementary dispatch evidence; never used by the 24-pair primary study."""

import functools
import json
from pathlib import Path

from matrix_worker import MatrixAuditWorker


class ProbeWorker(MatrixAuditWorker):
    def dispatch_probe_install(self, output_dir):
        import torch

        pipeline = self.model_runner.pipeline
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        metadata = self.matrix_metadata()
        metadata['attention_components'] = {}
        for name in ('transformer', 'text_encoder', 'vae'):
            component = getattr(pipeline, name)
            implementations = set()
            for module in component.modules():
                backend = getattr(module, 'attn_backend', None)
                impl = getattr(module, 'attn_impl_cls', None)
                processor = getattr(module, 'processor', None)
                if backend is not None:
                    implementations.add('backend:' + backend.get_name())
                if impl is not None:
                    implementations.add('impl:' + impl.__module__ + '.' + impl.__name__)
                if processor is not None:
                    implementations.add('processor:' + type(processor).__module__ + '.' + type(processor).__name__)
            config = getattr(component, 'config', None)
            metadata['attention_components'][name] = {
                'implementations': sorted(implementations),
                'config_attn_implementation': getattr(config, '_attn_implementation', None),
            }
        metadata['sdpa_enabled'] = {
            key: getattr(torch.backends.cuda, key + '_sdp_enabled')()
            for key in ('flash', 'mem_efficient', 'math', 'cudnn')
        }
        metadata['scope'] = 'Separate two-step, 1024x1024, no-CFG diagnostic after primary matrix; not part of SSIM or timing statistics.'
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

        def record_component(original_method, label):
            @functools.wraps(original_method)
            def recorded(*args, **kwargs):
                with torch.profiler.record_function(label):
                    return original_method(*args, **kwargs)
            return recorded

        for name in ('transformer', 'text_encoder'):
            component = getattr(pipeline, name)
            component.forward = record_component(component.forward, 'probe::' + name)
        for name in ('encode', 'decode'):
            setattr(pipeline.vae, name, record_component(getattr(pipeline.vae, name), 'probe::vae_' + name))

        # The runner calls pipeline.forward directly. Wrap that entry so the
        # profiler starts and stops on the inference thread, including encoders
        # and VAE decode, without altering any tensor or backend setting.
        original = pipeline.forward

        @functools.wraps(original)
        def profiled(*args, **kwargs):
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
            ) as profile:
                result = original(*args, **kwargs)
                torch.cuda.synchronize()
            profile.export_chrome_trace(str(output / 'trace.json'))
            operators = {event.key: event.count for event in profile.key_averages() if 'attn' in event.key.lower() or 'attention' in event.key.lower()}
            (output / 'attention-operators.json').write_text(json.dumps(operators, indent=2) + '\n')
            return result

        pipeline.forward = profiled
        return {'installed': True, 'output_dir': str(output)}
