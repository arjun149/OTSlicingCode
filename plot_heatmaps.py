import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_tlb_heatmap(tlb_cost, adata_source, adata_target, source_name, target_name):
    s_source = sorted(adata_source.obs['slice'].unique())
    s_target = sorted(adata_target.obs['slice'].unique())
    n_source = adata_source.obs['slice'].value_counts()
    n_target = adata_target.obs['slice'].value_counts()

    ro = n_source.loc[s_source].sort_values().index
    co = n_target.loc[s_target].sort_values().index
    cs = tlb_cost.loc[ro, co]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cs, cmap='viridis')
    ax.set_xticks(range(len(co)), [f"{c}\nn={n_target[c]}" for c in co], fontsize=8, rotation=45)
    ax.set_yticks(range(len(ro)), [f"{r} (n={n_source[r]})" for r in ro], fontsize=8)
    ax.set_xlabel(target_name)
    ax.set_ylabel(source_name)
    ax.set_title(f'fTLB cost: {source_name} -> {target_name}')
    plt.colorbar(im, label='fTLB cost')
    plt.tight_layout()
    plt.show()


def plot_attention_heatmap(csv_path, title=None, mask_diagonal=True):
    
    att = pd.read_csv(csv_path, index_col=0)
    att.columns = att.columns.astype(int)
    att.index = att.index.astype(int)

    values = att.values.astype(float).copy()
    if mask_diagonal:
        np.fill_diagonal(values, np.nan)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(values, cmap='magma')
    ax.set_xticks(range(len(att.columns)), att.columns, fontsize=7, rotation=90)
    ax.set_yticks(range(len(att.index)), att.index, fontsize=7)
    ax.set_xlabel('target slice')
    ax.set_ylabel('source slice')
    diag_note = " (diagonal masked)" if mask_diagonal else ""
    ax.set_title(title or f'STAIR attention{diag_note}: {csv_path}')
    plt.colorbar(im, label='attention score (higher = more similar)')
    plt.tight_layout()
    plt.show()

    return att
