"""
Generate comparison figures for scenario 1 — fCUBT with BIC / AIC / ICL / Stability.

Figures produced in scenario_1_metriques/figures/:
    ARI_metriques.pdf         — ARI boxplot for all four criteria
    n_clusters_metriques.pdf  — estimated K̂ grouped bar chart
    comparison_bic_aic.pdf    — ARI + K̂: BIC vs AIC  (2 panels)
    comparison_bic_icl.pdf    — ARI + K̂: BIC vs ICL  (2 panels)
    comparison_bic_stability.pdf — ARI + K̂: BIC vs Stability (2 panels)

Run from the project root:
    python scenario_1_metriques/02-results.py
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Matplotlib
# ---------------------------------------------------------------------------
matplotlib.use('pdf')
matplotlib.rcParams.update({'font.family': 'serif', 'text.usetex': True})

COLORS = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
          '#a65628', '#984ea3', '#999999', '#e41a1c']
sns.set_palette(sns.color_palette(COLORS))

TRUE_K = 5

LABELS = {
    'bic':       r'\texttt{fCUBT-BIC}',
    'aic':       r'\texttt{fCUBT-AIC}',
    'icl':       r'\texttt{fCUBT-ICL}',
    'stability': r'\texttt{fCUBT-Stab}',
}

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_pkl(name):
    path = os.path.join(RESULTS_DIR, f'results_{name}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found. Run 01-run.py first.')
    with open(path, 'rb') as f:
        res = pickle.load(f)
    print(f'Loaded {len(res):>4d} results for {LABELS[name]}.')
    return res


ORDER = ['bic', 'aic', 'icl', 'stability']
results = {k: load_pkl(k) for k in ORDER}

ARI     = {k: np.array([r['ARI']        for r in v]) for k, v in results.items()}
N_CLUST = {k: np.array([r['n_clusters'] for r in v]) for k, v in results.items()}

# ---------------------------------------------------------------------------
# Figure 1 — ARI comparison (all criteria, horizontal boxplot)
# ---------------------------------------------------------------------------
df_ari = pd.DataFrame({LABELS[k]: ARI[k] for k in ORDER})

fig, ax = plt.subplots(figsize=(6, 1 + 1.2 * len(ORDER)), constrained_layout=True)
bplot = sns.boxplot(data=df_ari, orient='h', ax=ax)
for i, patch in enumerate(bplot.patches[:len(ORDER)]):
    patch.set_facecolor(COLORS[i])
bplot.set_yticklabels(bplot.get_yticklabels(), size=13)
ax.set_xlabel('ARI', size=14)
ax.set_xlim((0, 1))
ax.set_title('ARI — scenario 1', size=13)
fig.savefig(os.path.join(FIGURES_DIR, 'ARI_metriques.pdf'), format='pdf')
plt.close(fig)
print('Saved ARI_metriques.pdf')

# ---------------------------------------------------------------------------
# Figure 2 — Estimated K̂ grouped bar chart (all criteria)
# ---------------------------------------------------------------------------
all_ks = sorted({k for key in ORDER for k in N_CLUST[key].tolist()})
x = np.arange(len(all_ks))
width = 0.8 / len(ORDER)

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for i, key in enumerate(ORDER):
    fracs = np.array([np.mean(N_CLUST[key] == k) for k in all_ks])
    ax.bar(x + (i - len(ORDER) / 2 + 0.5) * width, fracs, width,
           label=LABELS[key], color=COLORS[i], edgecolor='white')
if TRUE_K in all_ks:
    ax.axvline(x=all_ks.index(TRUE_K), color='red',
               linestyle='-.', linewidth=1.5, label=f'True $K={TRUE_K}$')
ax.set_xticks(x)
ax.set_xticklabels([str(k) for k in all_ks], size=12)
ax.set_xlabel('Estimated number of clusters', size=13)
ax.set_ylabel('Proportion', size=13)
ax.set_title(r'Estimated $\hat{K}$ — scenario 1', size=13)
ax.legend(fontsize=10)
fig.savefig(os.path.join(FIGURES_DIR, 'n_clusters_metriques.pdf'), format='pdf')
plt.close(fig)
print('Saved n_clusters_metriques.pdf')

# ---------------------------------------------------------------------------
# Figures 3-5 — BIC vs each alternative (2-panel: ARI + K̂)
# ---------------------------------------------------------------------------
for i, key in enumerate(['aic', 'icl', 'stability']):
    color = COLORS[i + 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    # ARI boxplot
    ax = axes[0]
    df = pd.DataFrame({LABELS['bic']: ARI['bic'], LABELS[key]: ARI[key]})
    bplot = sns.boxplot(data=df, orient='h', ax=ax)
    bplot.patches[0].set_facecolor(COLORS[0])
    bplot.patches[1].set_facecolor(color)
    bplot.set_yticklabels(bplot.get_yticklabels(), size=12)
    ax.set_xlabel('ARI', size=13)
    ax.set_xlim((0, 1))
    ax.set_title('ARI', size=13)

    # K̂ grouped bar chart
    ax = axes[1]
    ks    = sorted(set(N_CLUST['bic'].tolist() + N_CLUST[key].tolist()))
    x     = np.arange(len(ks))
    width = 0.35
    ax.bar(x - width / 2, [np.mean(N_CLUST['bic'] == k) for k in ks], width,
           label=LABELS['bic'], color=COLORS[0], edgecolor='white')
    ax.bar(x + width / 2, [np.mean(N_CLUST[key] == k) for k in ks], width,
           label=LABELS[key], color=color, edgecolor='white')
    if TRUE_K in ks:
        ax.axvline(x=ks.index(TRUE_K), color='red',
                   linestyle='-.', linewidth=1.5, label=f'True $K={TRUE_K}$')
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks], size=11)
    ax.set_xlabel('Estimated number of clusters', size=12)
    ax.set_ylabel('Proportion', size=12)
    ax.set_title(r'Estimated $\hat{K}$', size=13)
    ax.legend(fontsize=10)

    fig.suptitle(f'{LABELS["bic"]} vs {LABELS[key]} — scenario 1', size=13)
    out = os.path.join(FIGURES_DIR, f'comparison_bic_{key}.pdf')
    fig.savefig(out, format='pdf')
    plt.close(fig)
    print(f'Saved comparison_bic_{key}.pdf')

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print('\n=== Scenario 1 — model selection criteria summary ===')
for key in ORDER:
    a  = ARI[key]
    nc = N_CLUST[key]
    print(f'  {LABELS[key]:25s}  '
          f'ARI {a.mean():.3f} ± {a.std():.3f}  '
          f'median {np.median(a):.3f}  '
          f'mode K̂={pd.Series(nc).mode()[0]}')
print(f'\nFigures saved to {FIGURES_DIR}')
