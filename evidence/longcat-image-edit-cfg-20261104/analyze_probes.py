"""Report actual attention dispatch from the separate two-step diagnostics."""

import collections
import json
from pathlib import Path


def attention_name(name):
    name = name.lower()
    return any(token in name for token in ('attention', 'attn', 'fmha', 'flash'))


def main():
    root = Path(__file__).resolve().parent / 'dispatch-probes'
    result = {'scope': 'One separate two-step cat-to-dog request per variant, seed 7, guidance 1.0, 1024x1024; traces excluded from primary SSIM and timing data.', 'variants': {}}
    for variant in ('reference', 'baseline', 'fixed'):
        directory = root / variant
        assert (directory / 'COMPLETE').exists()
        events = json.loads((directory / 'trace.json').read_text())['traceEvents']
        ranges = [event for event in events if event.get('ph') == 'X' and event.get('name', '').startswith('probe::')]
        component_ops = collections.defaultdict(collections.Counter)
        for event in events:
            if event.get('ph') != 'X' or event.get('cat') != 'cpu_op' or not attention_name(event.get('name', '')):
                continue
            for enclosing in ranges:
                if event.get('tid') == enclosing.get('tid') and enclosing['ts'] <= event['ts'] and event['ts'] + event.get('dur', 0) <= enclosing['ts'] + enclosing.get('dur', 0):
                    component_ops[enclosing['name']][event['name']] += 1
        kernels = collections.Counter(event['name'] for event in events if event.get('cat') == 'kernel' and attention_name(event['name']))
        all_kernel_count = sum(event.get('cat') == 'kernel' for event in events)
        metadata = json.loads((directory / 'metadata.json').read_text())
        result['variants'][variant] = {
            'attention_components': metadata['attention_components'],
            'sdpa_enabled': metadata['sdpa_enabled'],
            'component_attention_operators': dict(component_ops),
            'attention_gpu_kernels': dict(kernels),
            'total_gpu_kernel_events': all_kernel_count,
            'pixel_sha256': (directory / 'pixel.sha256').read_text().strip(),
        }
        assert all_kernel_count > 0, 'Profiler did not capture CUDA kernel execution'
        assert component_ops and kernels, 'No attention dispatch events found'
    reference = result['variants']['reference']
    result['comparisons_to_reference'] = {
        variant: {key + '_equal': data[key] == reference[key] for key in (
            'attention_components', 'sdpa_enabled', 'component_attention_operators', 'attention_gpu_kernels', 'pixel_sha256',
        )}
        for variant, data in result['variants'].items() if variant != 'reference'
    }
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['comparisons_to_reference']), flush=True)
    for variant, data in result['variants'].items():
        print(variant, json.dumps(data['attention_components']), 'kernels', len(data['attention_gpu_kernels']), 'total GPU events', data['total_gpu_kernel_events'])


if __name__ == '__main__':
    main()
