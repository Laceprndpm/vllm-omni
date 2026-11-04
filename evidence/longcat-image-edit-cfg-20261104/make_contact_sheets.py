"""Scientific comparison figures only; source PNGs and metrics stay unchanged."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image


def draw(root, rows, path, title):
    figure, axes = plt.subplots(len(rows), 3, figsize=(11, 3.6 * len(rows)), squeeze=False, constrained_layout=True)
    for row_index, row in enumerate(rows):
        for column, variant in enumerate(('reference', 'baseline', 'fixed')):
            axis = axes[row_index, column]
            with Image.open(root / 'outputs' / variant / (row['job_id'] + '.png')) as image:
                axis.imshow(image)
            label = {'reference': 'Uncached reference', 'baseline': 'Before fix', 'fixed': 'After fix'}[variant]
            if variant != 'reference':
                label += f" | SSIM={row[variant + '_ssim']:.6f}"
            axis.set_title(label, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(row['job_id'].replace('_s', '\nseed ') + f"\nDelta={row['delta_ssim']:+.6f}", fontsize=9)
    figure.suptitle(title, fontsize=14)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    root = Path(__file__).resolve().parent
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['complete'] and len(summary['rows']) == 24
    directory = root / 'figures'
    directory.mkdir(exist_ok=True)
    for seed in summary['by_seed']:
        rows = [row for row in summary['rows'] if str(row['seed']) == seed]
        draw(root, rows, directory / f'seed-{seed}.png', f'All six edits, seed {seed}; original 1024 x 1024 outputs')
    lowest = sorted(summary['rows'], key=lambda row: row['delta_ssim'])[:3]
    draw(root, lowest, directory / 'lowest-deltas.png', 'Three lowest paired SSIM changes; no cases excluded')


if __name__ == '__main__':
    main()
