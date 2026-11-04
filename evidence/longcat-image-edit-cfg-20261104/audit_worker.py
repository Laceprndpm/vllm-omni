"""Observe live Cache-DiT state through the supported worker extension API."""


class CacheAuditWorkerExtension:
    def cache_audit_start(self):
        self._audit_transformer_forwards = 0
        transformer = self.model_runner.pipeline.transformer
        if not hasattr(self, '_audit_forward_handle'):
            def count_forward(module, args):
                self._audit_transformer_forwards += 1

            self._audit_forward_handle = transformer.register_forward_pre_hook(count_forward)
        return {'started': True}

    def cache_audit_snapshot(self):
        pipeline = self.model_runner.pipeline
        transformer = pipeline.transformer
        manager = transformer._context_manager
        contexts = []
        for name, context in manager._cached_context_manager.items():
            contexts.append({
                'name': name,
                'executed_steps': context.get_accumulated_executed_steps(),
                'transformer_forwards': context.get_accumulated_transformer_executed_steps(),
                'current_step': context.get_current_step(),
                'current_transformer_step': context.get_current_transformer_step(),
                'cached_steps': context.get_cached_steps(),
                'cfg_cached_steps': context.get_cfg_cached_steps(),
                'enable_separate_cfg': context.cache_config.enable_separate_cfg,
                'num_inference_steps': context.cache_config.num_inference_steps,
                'max_warmup_steps': context.cache_config.max_warmup_steps,
                'max_continuous_cached_steps': context.cache_config.max_continuous_cached_steps,
            })
        return {
            'pipeline_num_timesteps': pipeline._num_timesteps,
            'observed_transformer_forwards': self._audit_transformer_forwards,
            'published_summary_steps': getattr(transformer, '_accumulated_executed_steps', None),
            'published_summary_forwards': getattr(transformer, '_accumulated_transformer_executed_steps', None),
            'contexts': contexts,
        }
