"""Retain every paired result from the frozen SSIM study, including regressions."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from tests.e2e.accuracy.helpers import compute_image_ssim_psnr, write_json_artifact


def scheduler_config_for_comparison(config):
    # Diffusers serializes this set of defaulted argument names in process-
    # dependent order. Compare its membership; every numerical field stays
    # strict, and the original worker metadata remains unmodified on disk.
    result = dict(config)
    if '_use_default_values' in result:
        result['_use_default_values'] = sorted(result['_use_default_values'])
    return result


def aggregate(rows):
    before = np.array([r['baseline_ssim'] for r in rows])
    after = np.array([r['fixed_ssim'] for r in rows])
    delta = after - before
    return {
        'n': len(rows), 'baseline_mean': float(before.mean()), 'fixed_mean': float(after.mean()),
        'mean_delta': float(delta.mean()), 'median_delta': float(np.median(delta)),
        'min_delta': float(delta.min()), 'max_delta': float(delta.max()),
        'improved': int((delta > 0).sum()), 'tied': int((delta == 0).sum()), 'regressed': int((delta < 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'plan.json').read_text())
    plan_hash = hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest()
    assert plan_hash == (root / 'plan.sha256').read_text().split()[0]
    variants = ('reference', 'baseline', 'fixed')
    data = {}
    environments = {}
    for variant in variants:
        path = root / 'outputs' / variant / 'results.json'
        if not path.exists():
            if args.partial:
                print('Waiting for', variant)
                return
            raise FileNotFoundError(path)
        data[variant] = {r['job_id']: r for r in json.loads(path.read_text())}
        env = json.loads((path.parent / 'environment.json').read_text())
        assert env['plan_sha256'] == plan_hash
        environments[variant] = env
    for variant in ('baseline', 'fixed'):
        for key in ('versions', 'torch_cuda', 'cudnn', 'gpu', 'model_path', 'model_revision', 'cache_config', 'enforce_eager', 'input_conversion', 'omp_num_threads'):
            assert environments[variant][key] == environments['reference'][key], (variant, key)
        left, right = environments[variant]['worker_runtime'], environments['reference']['worker_runtime']
        for key in ('transformer_attention_implementations', 'scheduler_class', 'matmul_allow_tf32', 'cudnn_allow_tf32', 'cudnn_version'):
            assert left[key] == right[key], (variant, key)
        assert scheduler_config_for_comparison(left['scheduler_config']) == scheduler_config_for_comparison(right['scheduler_config']), (variant, 'scheduler_config')
        for component in ('transformer', 'text_encoder', 'vae'):
            for key in ('class', 'source_sha256', 'dtype'):
                assert left['components'][component][key] == right['components'][component][key], (variant, component, key)
    primary_ids = [f"{task['id']}_s{seed}" for seed in plan['seeds'] for task in plan['tasks']]
    available = [name for name in primary_ids if all(name in data[v] for v in variants)]
    if not args.partial:
        assert len(available) == 24
        for variant in variants:
            assert (root / 'outputs' / variant / 'COMPLETE').exists()
            assert len(data[variant]) == 31
    metrics_dir = root / 'metrics'
    metrics_dir.mkdir(exist_ok=True)
    rows = []
    options = {k: v for k, v in plan['primary_metric'].items() if k != 'library'}
    for name in available:
        records = {v: data[v][name] for v in variants}
        image_hashes = {v: records[v]['image_file_sha256'] for v in variants}
        saved = metrics_dir / f'{name}.json'
        if saved.exists():
            row = json.loads(saved.read_text())
            assert row['image_file_sha256'] == image_hashes
            rows.append(row)
            continue
        for key in ('id', 'input', 'input_sha256', 'prompt', 'seed', 'steps', 'guidance', 'negative_prompt', 'image_size'):
            assert records['reference'][key] == records['baseline'][key] == records['fixed'][key], (name, key)
        images = {}
        arrays = {}
        for variant in variants:
            path = root / 'outputs' / variant / f'{name}.png'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == image_hashes[variant]
            with Image.open(path) as image:
                assert image.mode == 'RGB' and image.size == (1024, 1024)
                images[variant] = image.copy()
                arrays[variant] = np.asarray(image).copy()
            assert hashlib.sha256(arrays[variant].tobytes()).hexdigest() == records[variant]['pixel_sha256']
        row = {k: records['reference'][k] for k in ('job_id', 'id', 'input', 'prompt', 'round', 'seed')}
        row['image_file_sha256'] = image_hashes
        reference = arrays['reference']
        ref_luma = reference @ np.array([0.2126, 0.7152, 0.0722]) / 255.0
        for variant in ('baseline', 'fixed'):
            prediction = arrays[variant]
            row[f'{variant}_ssim'] = float(structural_similarity(reference, prediction, **options))
            psnr = float(peak_signal_noise_ratio(reference, prediction, data_range=255))
            row[f'{variant}_psnr_db'] = psnr if np.isfinite(psnr) else 'inf'
            repo_ssim, repo_psnr = compute_image_ssim_psnr(prediction=images[variant], reference=images['reference'], compare_mode='RGB')
            row[f'{variant}_repository_ssim'] = repo_ssim
            row[f'{variant}_repository_psnr_db'] = repo_psnr if np.isfinite(repo_psnr) else 'inf'
            luma = prediction @ np.array([0.2126, 0.7152, 0.0722]) / 255.0
            row[f'{variant}_mean_luma_bias'] = float(luma.mean() - ref_luma.mean())
            row[f'{variant}_luma_contrast_ratio'] = float(luma.std() / ref_luma.std()) if ref_luma.std() > 0 else None
            row[f'{variant}_seconds'] = records[variant]['wall_seconds']
        row['delta_ssim'] = row['fixed_ssim'] - row['baseline_ssim']
        row['delta_repository_ssim'] = row['fixed_repository_ssim'] - row['baseline_repository_ssim']
        write_json_artifact(saved, row)
        rows.append(row)
    if not rows:
        print('No complete triplets yet')
        return
    summary = {'plan_sha256': plan_hash, 'complete': len(rows) == 24, 'overall': aggregate(rows), 'rows': rows}
    for key, output_key in [('seed', 'by_seed'), ('input', 'by_input'), ('id', 'by_task')]:
        summary[output_key] = {str(value): aggregate([r for r in rows if r[key] == value]) for value in dict.fromkeys(r[key] for r in rows)}
    summary['regressions'] = [r['job_id'] for r in sorted(rows, key=lambda r: r['delta_ssim']) if r['delta_ssim'] < 0]
    summary['worst_case'] = min(rows, key=lambda r: r['delta_ssim'])['job_id']
    summary['best_case'] = max(rows, key=lambda r: r['delta_ssim'])['job_id']
    summary['secondary_metric'] = {
        'name': 'repository compute_image_ssim_psnr / TorchMetrics',
        'baseline_mean': float(np.mean([r['baseline_repository_ssim'] for r in rows])),
        'fixed_mean': float(np.mean([r['fixed_repository_ssim'] for r in rows])),
        'mean_delta': float(np.mean([r['delta_repository_ssim'] for r in rows])),
        'direction_agreement': int(sum(np.sign(r['delta_ssim']) == np.sign(r['delta_repository_ssim']) for r in rows)),
    }
    repeats = []
    for variant in variants:
        for name in plan['repeats']:
            repeated = data[variant].get(name + '_repeat')
            if repeated is not None:
                repeats.append({'variant': variant, 'job_id': name, 'pixel_identical': repeated['pixel_sha256'] == data[variant][name]['pixel_sha256']})
    summary['repeatability'] = repeats
    write_json_artifact(root / ('partial-summary.json' if args.partial else 'summary.json'), summary)
    print(json.dumps({'complete_pairs': len(rows), **summary['overall'], 'by_seed': summary['by_seed'], 'repeatability_completed': len(repeats)}), flush=True)
    if args.partial:
        return
    assert len(repeats) == 9
    round_deltas = [summary['by_seed'][str(seed)]['mean_delta'] for seed in plan['seeds']]
    bootstrap = [np.mean([round_deltas[i] for i in sample]) for sample in itertools.product(range(4), repeat=4)]
    summary['seed_cluster_bootstrap_95_interval'] = [float(x) for x in np.percentile(bootstrap, [2.5, 97.5])]
    summary['bootstrap_scope'] = 'Resample four seed rounds jointly across six fixed editing tasks; all 4^4 ordered bootstrap samples. Exploratory interval from only four seeds, conditional on these selected tasks.'
    summary['previous_single_case_anchor'] = {}
    for variant, old_label in [('reference', 'uncached'), ('baseline', 'audited'), ('fixed', 'fixed_audited')]:
        old_path = root.parent / 'results' / old_label / 'results.json'
        old = next(r for r in json.loads(old_path.read_text()) if r['name'] == 'no_cfg_first')
        summary['previous_single_case_anchor'][variant] = data[variant]['cat_to_dog_s42']['pixel_sha256'] == old['pixel_sha256']
    write_json_artifact(root / 'summary.json', summary)
    csv_fields = [key for key in rows[0] if key != 'image_file_sha256']
    with (root / 'paired-results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    make_plot(root, plan, rows)
    make_report(root, plan, summary)
    print('ANALYSIS_COMPLETE', flush=True)


def make_plot(root, plan, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    colors = {'cat': '#2878B5', 'dog': '#E07B39', 'brooch': '#4E9A63'}
    values = [r[k] for r in rows for k in ('baseline_ssim', 'fixed_ssim')]
    bounds = (min(values) - 0.01, max(values) + 0.01)
    axes[0].plot(bounds, bounds, color='#777777', linestyle='--', linewidth=1)
    for name, color in colors.items():
        selected = [r for r in rows if r['input'] == name]
        axes[0].scatter([r['baseline_ssim'] for r in selected], [r['fixed_ssim'] for r in selected], color=color, label=name, s=43)
    axes[0].set(xlabel='Before fix: SSIM vs uncached', ylabel='After fix: SSIM vs uncached', xlim=bounds, ylim=bounds, title='24 matched image / edit / seed cases')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].legend()
    tasks = [task['id'] for task in plan['tasks']]
    axes[1].axhline(0, color='#777777', linewidth=1)
    for seed_index, seed in enumerate(plan['seeds']):
        selected = {r['id']: r for r in rows if r['seed'] == seed}
        xs = np.arange(len(tasks)) + (seed_index - 1.5) * 0.12
        axes[1].scatter(xs, [selected[task]['delta_ssim'] for task in tasks], label=f'seed {seed}', s=36)
    axes[1].set_xticks(np.arange(len(tasks)), [name.replace('_', '\n') for name in tasks], fontsize=9)
    axes[1].set(ylabel='SSIM after minus SSIM before', title='Every paired change, including regressions')
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
    fig.savefig(root / 'ssim-comparison.png', dpi=200)
    fig.savefig(root / 'ssim-comparison.pdf')
    plt.close(fig)


def make_report(root, plan, summary):
    overall = summary['overall']
    interval = summary['seed_cluster_bootstrap_95_interval']
    lines = [
        'LongCat-Image-Edit：四轮配对 SSIM 验证', '',
        '结论：' + ('本次固定测试矩阵的平均 SSIM 提高。' if overall['mean_delta'] > 0 else '本次测试未证实平均 SSIM 提升。'),
        f"全部 {overall['n']} 组：均值 {overall['baseline_mean']:.6f} -> {overall['fixed_mean']:.6f}，差值 {overall['mean_delta']:+.6f}。",
        f"逐样例：提升 {overall['improved']}，持平 {overall['tied']}，下降 {overall['regressed']}；差值中位数 {overall['median_delta']:+.6f}。",
        f"最小/最大差值：{overall['min_delta']:+.6f} / {overall['max_delta']:+.6f}。",
        f"按 seed 轮次重采样的 95% 区间：[{interval[0]:+.6f}, {interval[1]:+.6f}]。",
        '该区间只有 4 个 seed 轮次，且以选定的 6 个编辑任务为条件，不能外推为所有输入的保证。',
        '', '每轮结果（每轮 6 组）：', 'seed     before       after        delta        improved / regressed',
    ]
    for seed, result in summary['by_seed'].items():
        lines.append(f"{seed:<9}{result['baseline_mean']:<13.6f}{result['fixed_mean']:<13.6f}{result['mean_delta']:<+13.6f}{result['improved']} / {result['regressed']}")
    lines += ['', '按输入图分组（每图 8 组）：']
    for name, result in summary['by_input'].items():
        lines.append(f"{name}: {result['baseline_mean']:.6f} -> {result['fixed_mean']:.6f}, delta={result['mean_delta']:+.6f}, improved={result['improved']}, regressed={result['regressed']}")
    lines += ['', '按编辑任务分组（每任务 4 个 seed）：']
    for name, result in summary['by_task'].items():
        lines.append(f"{name}: {result['baseline_mean']:.6f} -> {result['fixed_mean']:.6f}, delta={result['mean_delta']:+.6f}")
    secondary = summary['secondary_metric']
    lines += [
        '', f"仓库 TorchMetrics SSIM 交叉核对：{secondary['baseline_mean']:.6f} -> {secondary['fixed_mean']:.6f}；变化方向一致 {secondary['direction_agreement']}/24。",
        f"重复性：{sum(r['pixel_identical'] for r in summary['repeatability'])}/9 个重复样例逐像素一致；重复样例不计入 24 组统计。",
        f"与此前 seed=42 的单例跨进程核对：{json.dumps(summary['previous_single_case_anchor'])}。",
        '', f"最小差值样例：{summary['worst_case']}；最大差值样例：{summary['best_case']}。",
        '所有下降样例：' + (', '.join(summary['regressions']) or '无'),
        '', '实验合同：',
        f"基线：{plan['base_commit']}", f"修复补丁 SHA256：{plan['fix_patch_sha256']}",
        f"冻结计划 SHA256：{summary['plan_sha256']}",
        '单卡 NVIDIA A800 80GB；相同 runtime、权重、seed、正常图像预处理及输出解码。',
        '3 张输入图、每图 2 个编辑指令，seed=[7,42,123,2026]；guidance=1.0，negative_prompt=""。',
        '50 步、1024x1024、默认 Cache-DiT 配置、enforce_eager=True。',
        '无缓存参考和修复前缓存使用独立原始 main 快照；修复后使用已验证补丁。',
        '每轮前有 2 步 CFG warmup，以覆盖 CFG 到无 CFG 的切换；warmup 排除在统计外。',
        '主指标沿用前次 scikit-image SSIM：RGB、data_range=255、Gaussian sigma=1.5、use_sample_covariance=False。',
        '共享 compute_image_ssim_psnr 的 TorchMetrics 分数单独报告，不与主指标混合。',
        '亮度和对比度诊断保留在 CSV；它们与 SSIM 均不直接证明主观画质或漂白故障。',
        '未调整提示词、seed、缓存参数或修复代码来挑选高分；全部 24 组均保留。',
        '', '重跑：source ../env.sh，然后 bash run_all.sh（需新的空 outputs 目录，A800及已缓存权重）。',
        '分析：PYTHONPATH=../vllm-omni python analyze_matrix.py',
        '完整逐样例结果：paired-results.csv、summary.json、metrics/*.json。',
        '原图与生成图：inputs/、outputs/{reference,baseline,fixed}/*.png。',
        '环境、实际 worker 源码及组件 dtype：各 outputs 子目录的 environment.json。',
        'scheduler 数值配置逐项相同；只对 _use_default_values 参数名集合的序列化顺序做无损归一化后比较。',
        '主实验的 attention 属性探针返回空列表，不能作为 kernel 一致的证据；另留独立两步 dispatch-probes 追踪，不计入主实验指标。',
        '图表：ssim-comparison.png、ssim-comparison.pdf。',
    ]
    (root / 'report.txt').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
