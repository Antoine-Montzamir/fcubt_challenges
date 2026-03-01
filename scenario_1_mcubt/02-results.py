"""
Generate figures for scenario 1 — mCUBT method.

Mirrors the structure of scenario_1/09-results.py but focuses on mCUBT,
and includes a head-to-head comparison with fCUBT (scenario_1 results).

Figures produced (in scenario_1_mcubt/figures/):
    ARI_mcubt.pdf              — ARI distribution of mCUBT over N_SIM runs
    n_clusters_mcubt.pdf       — Estimated number of clusters distribution
    root_k_mcubt.pdf           — K̂ chosen at the root node (unique to mCUBT)
    comptime_mcubt.pdf         — Computation time distribution
    comparison_fcubt_mcubt.pdf — ARI comparison: fCUBT vs mCUBT (side-by-side)
    n_clusters_comparison.pdf  — n_clusters: fCUBT vs mCUBT (side-by-side)

Run from scenario_1_mcubt/:
    python 02-results.py
Or from the project root:
    python scenario_1_mcubt/02-results.py
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

RESULTS_DIR  = os.path.join(SCRIPT_DIR, 'results')
FIGURES_DIR  = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# fCUBT results from scenario_1 (for comparison)
FCUBT_RESULTS = os.path.join(ROOT_DIR, 'scenario_1', 'results', 'results_fcubt.pkl')
FCUBT_COMPTIME = os.path.join(ROOT_DIR, 'scenario_1', 'results', 'results_fcubt_comptime.pkl')

# ---------------------------------------------------------------------------
# Matplotlib — pdf backend with LaTeX text
# ---------------------------------------------------------------------------
matplotlib.use('pdf')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
})

COLORS = [
    '#377eb8', '#ff7f00', '#4daf4a',
    '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00',
]
sns.set_palette(sns.color_palette(COLORS))

# ---------------------------------------------------------------------------
# Load mCUBT results
# ---------------------------------------------------------------------------
mcubt_path = os.path.join(RESULTS_DIR, 'results_mcubt.pkl')
if not os.path.exists(mcubt_path):
    raise FileNotFoundError(
        f'{mcubt_path} not found.\n'
        'Run 01-run.py first to generate the simulation results.'
    )

with open(mcubt_path, 'rb') as f:
    results_mcubt = pickle.load(f)

n_clusters_mcubt = np.array([r['n_clusters'] for r in results_mcubt])
ARI_mcubt        = np.array([r['ARI']        for r in results_mcubt])
comptime_mcubt   = np.array([r['comp_time']  for r in results_mcubt])
root_k_mcubt     = np.array([r['root_k']     for r in results_mcubt])

# ---------------------------------------------------------------------------
# Load fCUBT results (optional — used for comparison figures)
# ---------------------------------------------------------------------------
fcubt_available = os.path.exists(FCUBT_RESULTS)
if fcubt_available:
    with open(FCUBT_RESULTS, 'rb') as f:
        results_fcubt = pickle.load(f)
    n_clusters_fcubt = np.array([r['n_clusters'] for r in results_fcubt])
    ARI_fcubt        = np.array([r['ARI']        for r in results_fcubt])
else:
    print(f'[warning] fCUBT results not found at {FCUBT_RESULTS}; '
          'comparison figures will be skipped.')

fcubt_comptime_available = os.path.exists(FCUBT_COMPTIME)
if fcubt_comptime_available:
    with open(FCUBT_COMPTIME, 'rb') as f:
        results_fcubt_comptime = pickle.load(f)
    comptime_fcubt = np.array([r['comp_time'] for r in results_fcubt_comptime])

# ---------------------------------------------------------------------------
# Figure 1 — ARI distribution of mCUBT
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
ax.violinplot(ARI_mcubt, vert=False, showmedians=True)
ax.set_xlabel('ARI', size=14)
ax.set_yticks([])
ax.set_xlim((0, 1))
ax.set_title(r'\texttt{mCUBT} — ARI distribution', size=13)
fig.savefig(os.path.join(FIGURES_DIR, 'ARI_mcubt.pdf'), format='pdf')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2 — Estimated number of clusters
# ---------------------------------------------------------------------------
counts = pd.Series(n_clusters_mcubt).value_counts().sort_index()

fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
ax.bar(counts.index.astype(str), counts.values / counts.sum(),
       color=COLORS[0], edgecolor='white')
ax.axvline(x=str(5), color='red', linestyle='-.', linewidth=1.5,
           label='True $K=5$')
ax.set_xlabel('Estimated number of clusters', size=13)
ax.set_ylabel('Proportion', size=13)
ax.set_title(r'\texttt{mCUBT} — estimated $\hat{K}$', size=13)
ax.legend(fontsize=11)
fig.savefig(os.path.join(FIGURES_DIR, 'n_clusters_mcubt.pdf'), format='pdf')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3 — Root K̂ distribution (unique to mCUBT)
# ---------------------------------------------------------------------------
root_counts = pd.Series(root_k_mcubt).value_counts().sort_index()

fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
ax.bar(root_counts.index.astype(str), root_counts.values / root_counts.sum(),
       color=COLORS[1], edgecolor='white')
ax.set_xlabel(r'$\hat{K}$ at root node', size=13)
ax.set_ylabel('Proportion', size=13)
ax.set_title(r'\texttt{mCUBT} — root branching factor', size=13)
fig.savefig(os.path.join(FIGURES_DIR, 'root_k_mcubt.pdf'), format='pdf')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4 — Computation time
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
ax.violinplot(comptime_mcubt, vert=False, showmedians=True)
ax.set_xlabel('Computation time (s)', size=13)
ax.set_yticks([])
ax.set_title(r'\texttt{mCUBT} — computation time', size=13)
fig.savefig(os.path.join(FIGURES_DIR, 'comptime_mcubt.pdf'), format='pdf')
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 5 — ARI comparison: fCUBT vs mCUBT
# ---------------------------------------------------------------------------
if fcubt_available:
    df_ari = pd.DataFrame({
        r'\texttt{fCUBT}': ARI_fcubt,
        r'\texttt{mCUBT}': ARI_mcubt,
    })

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    bplot = sns.boxplot(data=df_ari, orient='h', ax=ax)
    bplot.set_yticklabels(bplot.get_yticklabels(), size=13)
    bplot.patches[0].set_facecolor(COLORS[0])
    bplot.patches[1].set_facecolor(COLORS[1])
    ax.set_xlabel('ARI', size=14)
    ax.set_xlim((0, 1))
    ax.set_title('ARI — scenario 1', size=13)
    fig.savefig(os.path.join(FIGURES_DIR, 'comparison_fcubt_mcubt.pdf'),
                format='pdf')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 6 — n_clusters comparison: fCUBT vs mCUBT
# ---------------------------------------------------------------------------
if fcubt_available:
    ks = sorted(set(n_clusters_fcubt.tolist() + n_clusters_mcubt.tolist()))
    x = np.arange(len(ks))
    width = 0.35

    fcubt_frac = np.array(
        [np.mean(n_clusters_fcubt == k) for k in ks]
    )
    mcubt_frac = np.array(
        [np.mean(n_clusters_mcubt == k) for k in ks]
    )

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.bar(x - width / 2, fcubt_frac, width,
           label=r'\texttt{fCUBT}', color=COLORS[0], edgecolor='white')
    ax.bar(x + width / 2, mcubt_frac, width,
           label=r'\texttt{mCUBT}', color=COLORS[1], edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks], size=12)
    ax.set_xlabel('Estimated number of clusters', size=13)
    ax.set_ylabel('Proportion', size=13)
    ax.set_title(r'Estimated $\hat{K}$ — scenario 1', size=13)
    ax.axvline(x=ks.index(5) if 5 in ks else -1,
               color='red', linestyle='-.', linewidth=1.5,
               label='True $K=5$')
    ax.legend(fontsize=11)
    fig.savefig(os.path.join(FIGURES_DIR, 'n_clusters_comparison.pdf'),
                format='pdf')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 7 — Computation time comparison (if fCUBT comptime available)
# ---------------------------------------------------------------------------
if fcubt_comptime_available:
    df_time = pd.DataFrame({
        r'\texttt{fCUBT}': comptime_fcubt,
        r'\texttt{mCUBT}': comptime_mcubt,
    })

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    bplot = sns.boxplot(data=df_time, orient='h', ax=ax)
    bplot.set_yticklabels(bplot.get_yticklabels(), size=13)
    bplot.patches[0].set_facecolor(COLORS[0])
    bplot.patches[1].set_facecolor(COLORS[1])
    ax.set_xlabel('Computation time (s)', size=14)
    ax.set_title('Computation time — scenario 1', size=13)
    fig.savefig(os.path.join(FIGURES_DIR, 'comptime_comparison.pdf'),
                format='pdf')
    plt.close(fig)

# ---------------------------------------------------------------------------
# Summary statistics (printed)
# ---------------------------------------------------------------------------
print('\n=== mCUBT — scenario 1 summary ===')
print(f'  N simulations   : {len(results_mcubt)}')
print(f'  ARI  mean ± std : {ARI_mcubt.mean():.3f} ± {ARI_mcubt.std():.3f}')
print(f'  ARI  median     : {np.median(ARI_mcubt):.3f}')
print(f'  n_clusters mode : {pd.Series(n_clusters_mcubt).mode()[0]}')
print(f'  root_k mode     : {pd.Series(root_k_mcubt).mode()[0]}')
print(f'  comp time mean  : {comptime_mcubt.mean():.2f}s')
if fcubt_available:
    print(f'\n=== fCUBT vs mCUBT ===')
    print(f'  fCUBT ARI mean  : {ARI_fcubt.mean():.3f}')
    print(f'  mCUBT ARI mean  : {ARI_mcubt.mean():.3f}')
print(f'\nFigures saved to {FIGURES_DIR}')
