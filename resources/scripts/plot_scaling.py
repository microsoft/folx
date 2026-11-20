"""Plots the FermiNet and deep set scaling benchmarks side by side (Figure 2 of the paper).

Usage: python resources/scripts/plot_scaling.py
  (after python resources/scripts/ferminet.py and python resources/scripts/deepset.py)
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from common import RESULTS_DIR

plt.rcParams.update(
    {
        'font.family': 'serif',
        'font.serif': ['cmr10'],
        'mathtext.fontset': 'cm',
        'axes.formatter.use_mathtext': True,
        'axes.unicode_minus': False,
        'font.size': 8,
    }
)

STYLE = {
    'hessian_loop': dict(color='#666666', marker='s', ls='--'),
    'folx_sparse': dict(color='#2a6fb0', marker='o'),
    'folx_sparse_chunked': dict(color='#3f8f4f', marker='D', ls=':'),
    'folx_dense_chunked': dict(color='#d9822b', marker='^', ls='-.'),
}
LABELS = {
    'hessian_loop': 'Hessian loop',
    'folx_dense_chunked': 'folx, dense',
    'folx_sparse_chunked': 'folx, sparse',
}


PANELS = {'ferminet': '(a) FermiNet', 'deepset': '(b) Deep set'}


def plot_panel(ax, name, title):
    data = json.loads((RESULTS_DIR / f'{name}.json').read_text())
    batch = data['config']['batch']
    results = {int(n): r for n, r in data['results'].items()}
    for method, label in LABELS.items():
        ns = [n for n in sorted(results) if 'run' in results[n].get(method, {})]
        time = [results[n][method]['run'] / batch * 1e3 for n in ns]
        ax.plot(ns, time, label=label, ms=4, **STYLE[method])

    ax.set_title(title, loc='left', fontsize=8)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('Number of electrons $N$')
    ax.set_xticks(sorted(results))
    ax.set_xticklabels([str(n) for n in sorted(results)])
    ax.minorticks_off()
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.4, 2.3))
    for ax, (name, title) in zip(axes, PANELS.items()):
        plot_panel(ax, name, title)
    axes[0].set_ylabel('Time per Laplacian (ms)')
    axes[0].legend(frameon=False, handlelength=2.2, loc='upper left')
    fig.tight_layout()

    out = Path(__file__).parents[1] / 'figures' / 'scaling.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
