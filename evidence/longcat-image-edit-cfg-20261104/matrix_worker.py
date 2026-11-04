"""Read-only worker diagnostics for the paired SSIM experiment."""

import hashlib
import inspect
from pathlib import Path

from audit_worker import CacheAuditWorkerExtension


class MatrixAuditWorker(CacheAuditWorkerExtension):
    def matrix_metadata(self):
        import torch

        pipeline = self.model_runner.pipeline
        components = {}
        for name in ('transformer', 'text_encoder', 'vae'):
            module = getattr(pipeline, name, None)
            if module is None:
                continue
            parameter = next(module.parameters(), None)
            path = Path(inspect.getfile(type(module))).resolve()
            components[name] = {
                'class': f'{type(module).__module__}.{type(module).__name__}',
                'source_path': str(path),
                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'dtype': str(parameter.dtype) if parameter is not None else None,
            }
        attention = sorted({
            f'{type(module).__module__}.{type(module).__name__} -> {type(module.impl).__module__}.{type(module.impl).__name__}'
            for module in pipeline.transformer.modules() if hasattr(module, 'impl')
        })
        path = Path(inspect.getfile(type(pipeline))).resolve()
        return {
            'pipeline_source_path': str(path),
            'pipeline_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'components': components,
            'transformer_attention_implementations': attention,
            'scheduler_class': type(pipeline.scheduler).__name__,
            'scheduler_config': dict(pipeline.scheduler.config),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_version': torch.backends.cudnn.version(),
        }

    def matrix_snapshot(self):
        transformer = self.model_runner.pipeline.transformer
        if hasattr(transformer, '_context_manager'):
            return self.cache_audit_snapshot()
        return {
            'pipeline_num_timesteps': self.model_runner.pipeline._num_timesteps,
            'observed_transformer_forwards': self._audit_transformer_forwards,
            'contexts': [],
        }
