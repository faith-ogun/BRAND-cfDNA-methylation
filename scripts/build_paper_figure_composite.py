"""Composite Figure 2: five panels.

Layout:
  Row 1: A (baseline heatmap) | B (follow-up heatmap)
  Row 2: C (baseline ROC) | D (follow-up ROC) | E (continuous HR plot)

300 dpi PNG, paper-figure ready. Panel labels in bold sans-serif.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent
SENS       = REPO / 'results' / 'sensitivity_analysis'
OUT        = REPO / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

DPI = 300
C_RESPONDER = '#2c7bb6'
C_NONRESP   = '#c0392b'
C_BASELINE  = '#2c7bb6'
C_FU        = '#c0392b'


def load_data():
    base = pd.read_csv(SENS / 'dmr36_matrix.csv', index_col='patient_id')
    fu   = pd.read_csv(SENS / 'dmr36_matrix_fu.csv', index_col='patient_id')
    fu   = fu.loc[base.index]
    base_pred = pd.read_csv(SENS / 'sensitivity_36dmr.csv').set_index('patient_id').loc[base.index]
    fu_pred   = pd.read_csv(SENS / 'fu_alone_perpatient.csv').set_index('patient_id').loc[base.index]
    dyn       = pd.read_csv(SENS / 'fu_dynamic_results.csv').set_index('patient_id').loc[base.index]
    return base, fu, base_pred, fu_pred, dyn


def roc_with_bootstrap(y_true, y_score, n_boot=2000, seed=42):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_pt = auc(fpr, tpr)
    rng = np.random.default_rng(seed)
    n = len(y_true); idx = np.arange(n)
    grid = np.linspace(0, 1, 200)
    tpr_b = np.zeros((n_boot, len(grid)))
    auc_b = np.zeros(n_boot)
    for i in range(n_boot):
        s = rng.choice(idx, size=n, replace=True)
        if len(np.unique(y_true[s])) < 2:
            tpr_b[i] = np.nan; auc_b[i] = np.nan; continue
        f_, t_, _ = roc_curve(y_true[s], y_score[s])
        tpr_b[i] = np.interp(grid, f_, t_)
        auc_b[i] = auc(f_, t_)
    tlo = np.nanpercentile(tpr_b, 2.5, axis=0)
    thi = np.nanpercentile(tpr_b, 97.5, axis=0)
    alo, ahi = np.nanpercentile(auc_b, [2.5, 97.5])
    return fpr, tpr, auc_pt, grid, tlo, thi, alo, ahi


def add_panel_label(ax, label, x=-0.12, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=18, fontweight='bold',
            ha='left', va='bottom')


def draw_heatmap(ax_bar, ax, matrix_df, base_pred, show_ylabels=True):
    actual = base_pred['actual'].astype(int).values
    region_short = [r.split(':')[0] + ':' + r.split(':')[1].split('-')[0] for r in matrix_df.columns]
    X = matrix_df.values.astype(float)
    z = (X - X.mean(axis=0)) / X.std(axis=0)

    r_idx  = np.where(actual == 0)[0]
    nr_idx = np.where(actual == 1)[0]
    def order_within(idx):
        if len(idx) < 2: return idx
        d = linkage(z[idx], method='average', metric='euclidean')
        return idx[leaves_list(d)]
    r_ord  = order_within(r_idx)
    nr_ord = order_within(nr_idx)
    col_order = np.concatenate([r_ord, nr_ord])
    row_order = leaves_list(linkage(z.T, method='average', metric='euclidean'))
    z_plot = z[col_order][:, row_order].T

    # group bar
    bar = np.zeros((1, len(col_order)))
    bar[0, len(r_ord):] = 1
    ax_bar.imshow(bar, aspect='auto',
                  cmap=plt.matplotlib.colors.ListedColormap([C_RESPONDER, C_NONRESP]))
    ax_bar.set_xticks([]); ax_bar.set_yticks([])
    ax_bar.set_ylabel('Group', rotation=0, ha='right', va='center', fontsize=8)

    # heatmap
    im = ax.imshow(z_plot, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
    if show_ylabels:
        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels([region_short[i] for i in row_order], fontsize=5)
        ax.set_ylabel('36 stability-filtered DMRs', fontsize=9)
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(f'Patients (DCB n={len(r_ord)} | no DCB n={len(nr_ord)})',
                  fontsize=9)
    return im


def draw_roc(ax, y, score, label, colour, title):
    fpr, tpr, aucpt, grid, tlo, thi, alo, ahi = roc_with_bootstrap(y, score)
    ax.fill_between(grid, tlo, thi, color=colour, alpha=0.18, label='95% CI')
    ax.plot(fpr, tpr, color=colour, lw=2,
            label=f'AUC = {aucpt:.2f}\n95% CI {alo:.2f} to {ahi:.2f}')
    ax.plot([0, 1], [0, 1], color='grey', ls='--', lw=1, label='Chance')
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel('1 - specificity', fontsize=9)
    ax.set_ylabel('Sensitivity', fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(loc='lower right', fontsize=7.5, frameon=True)
    ax.grid(alpha=0.25)


def draw_panel_D(ax, dyn):
    # Plot in benefit direction (HR < 1 = benefit), matching the manuscript.
    # Define Δprob_R = -Δprob_NR: how much more responder-like did the score get
    # from baseline to FU. Positive Δprob_R means the patient moved toward
    # responder during treatment. Same Cox model as the analysis text, just the
    # opposite sign on the predictor.
    df = dyn[['delta_prob_NR', 'PFS_Months', 'pfs_event']].copy()
    df['delta_prob_R'] = -df['delta_prob_NR']
    df = df.rename(columns={'PFS_Months': 'T', 'pfs_event': 'E'})
    cox = CoxPHFitter().fit(df[['T', 'E', 'delta_prob_R']],
                            duration_col='T', event_col='E')
    s = cox.summary.loc['delta_prob_R']
    coef = float(s['coef']); se = float(s['se(coef)']); p = float(s['p'])
    hr_p = float(np.exp(coef * 0.1))
    lo_p = float(np.exp((coef - 1.96 * se) * 0.1))
    hi_p = float(np.exp((coef + 1.96 * se) * 0.1))

    deltas = df['delta_prob_R'].values
    pad = 0.02
    x = np.linspace(deltas.min() - pad, deltas.max() + pad, 400)
    hr = np.exp(coef * x)
    hr_lo = np.minimum(np.exp((coef - 1.96 * se) * x), np.exp((coef + 1.96 * se) * x))
    hr_hi = np.maximum(np.exp((coef - 1.96 * se) * x), np.exp((coef + 1.96 * se) * x))

    ax.fill_between(x, hr_lo, hr_hi, color=C_BASELINE, alpha=0.18, label='95% CI')
    ax.plot(x, hr, color=C_BASELINE, lw=2, label='Predicted HR')
    ax.axhline(1.0, color='grey', ls='--', lw=1)
    ax.axvline(0.0, color='grey', ls=':', lw=0.8)
    for d in deltas:
        ax.plot([d, d], [0.0, 0.03], color='black', lw=0.5, alpha=0.7,
                transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_yscale('log')
    ax.set_xlabel('Δ predicted P(response), FU minus baseline', fontsize=9)
    ax.set_ylabel('Hazard ratio vs Δ = 0 (log)', fontsize=9)
    ax.set_title('Continuous benefit of Δprob_R on PFS', fontsize=10)
    ax.grid(alpha=0.3, which='both')

    annot = (f'HR per 0.1 unit = {hr_p:.2f}\n'
             f'95% CI {lo_p:.2f} to {hi_p:.2f}\n'
             f'p = {p:.3f}, n = {len(df)}\n'
             f'HR < 1 = benefit')
    ax.text(0.02, 0.04, annot, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8, bbox=dict(boxstyle='round', facecolor='white',
                                  edgecolor='grey', alpha=0.92))
    ax.legend(loc='upper right', fontsize=7.5, frameon=True)


def main():
    base, fu, base_pred, fu_pred, dyn = load_data()
    y = base_pred['actual'].astype(int).values
    s_b = base_pred['prob_nonresponder'].values
    s_f = fu_pred['prob_nonresponder_fu'].values

    fig = plt.figure(figsize=(16, 13))
    gs_outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 0.85], hspace=0.30, figure=fig)

    # Top row: two heatmaps side by side (A baseline, B follow-up)
    gs_top = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_outer[0],
                                              height_ratios=[0.04, 1.0],
                                              hspace=0.04, wspace=0.10)
    ax_bar_A = fig.add_subplot(gs_top[0, 0])
    ax_A     = fig.add_subplot(gs_top[1, 0])
    ax_bar_B = fig.add_subplot(gs_top[0, 1])
    ax_B     = fig.add_subplot(gs_top[1, 1])

    im_A = draw_heatmap(ax_bar_A, ax_A, base, base_pred, show_ylabels=True)
    im_B = draw_heatmap(ax_bar_B, ax_B, fu, base_pred, show_ylabels=False)
    ax_bar_A.set_title('Baseline methylation', fontsize=11, pad=14)
    ax_bar_B.set_title('Follow-up methylation', fontsize=11, pad=14)

    add_panel_label(ax_bar_A, 'A', x=-0.06, y=2.5)
    add_panel_label(ax_bar_B, 'B', x=-0.03, y=2.5)

    # Shared colourbar to the right of B
    cbar_ax = fig.add_axes([0.905, 0.56, 0.011, 0.28])
    cbar = fig.colorbar(im_B, cax=cbar_ax)
    cbar.set_label('Methylation z-score', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Group legend above the colourbar, well clear of it
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=C_RESPONDER, label='DCB (PFS ≥ 6 mo)'),
               Patch(facecolor=C_NONRESP, label='no DCB (PFS < 6 mo)')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.895, 0.92),
               frameon=False, fontsize=8)

    # Bottom row: C (baseline ROC), D (FU ROC), E (continuous HR)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[1], wspace=0.32)
    ax_C = fig.add_subplot(gs_bot[0])
    ax_D = fig.add_subplot(gs_bot[1])
    ax_E = fig.add_subplot(gs_bot[2])

    draw_roc(ax_C, y, s_b, 'Baseline', C_BASELINE,
             'ROC: 36-DMR panel at baseline')
    draw_roc(ax_D, y, s_f, 'Follow-up', C_FU,
             'ROC: 36-DMR panel at follow-up')
    draw_panel_D(ax_E, dyn)

    add_panel_label(ax_C, 'C')
    add_panel_label(ax_D, 'D')
    add_panel_label(ax_E, 'E')

    fig.suptitle('cfDNA methylation: 36-DMR panel performance and dynamic change',
                 fontsize=13, y=0.995, fontweight='bold')

    out = OUT / 'figure2_composite.png'
    plt.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')

    # Standalone Panel E (benefit-direction continuous HR plot)
    fig_e, ax_e = plt.subplots(figsize=(7, 5.5))
    draw_panel_D(ax_e, dyn)
    plt.tight_layout()
    out_e = OUT / 'panelE_continuous_hr_benefit.png'
    plt.savefig(out_e, dpi=DPI, bbox_inches='tight')
    plt.close(fig_e)
    print(f'wrote {out_e}')


if __name__ == '__main__':
    main()
