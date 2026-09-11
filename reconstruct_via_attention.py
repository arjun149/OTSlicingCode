import os
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from STAIR.loc_prediction import sort_slices

REAL_SECTION_THICKNESS_UM = 15


def reconstruct_z_via_attention(adata, attention_csv_path, start_slice,
                                 slice_col='slice', real_spacing_um=REAL_SECTION_THICKNESS_UM):
    atte = pd.read_csv(attention_csv_path, index_col=0)
    atte.columns = atte.columns.astype(str)
    atte.index = atte.index.astype(str)

    slices_present = sorted(adata.obs[slice_col].unique(), key=lambda x: int(x))
    if str(start_slice) not in atte.index:
        raise ValueError(f"start_slice={start_slice} not in attention matrix. "
                          f"Available: {list(atte.index)}")

    print(f"Running sort_slices, root={start_slice}...")
    dists = sort_slices(atte, start=str(start_slice))
    print("Raw accumulated distances (STAIR's own relative ordering):")
    for k, v in sorted(dists.items(), key=lambda kv: kv[1]):
        print(f"  {k}: {v:.4f}")

    
    adata.obs['z_rec_raw'] = adata.obs[slice_col].astype(str).map(dists).astype(float)

    
    z_norm = (adata.obs['z_rec_raw'] - adata.obs['z_rec_raw'].min()) / \
             (adata.obs['z_rec_raw'].max() - adata.obs['z_rec_raw'].min())

    
    n_slices = len(slices_present)
    real_total_span_um = (n_slices - 1) * real_spacing_um
    adata.obs['z_rec_um'] = z_norm * real_total_span_um

    print(f"\nRescaled to real physical span: 0 to {real_total_span_um} um "
          f"({n_slices} slices x {real_spacing_um}um)")
    print("(This is STAIR's RELATIVE ordering, anchored to OUR confirmed REAL scale --")
    print(" not naive uniform spacing, and not borrowed ground truth.)")

    return adata, dists


if __name__ == "__main__":
    timepoint = sys.argv[1] if len(sys.argv) > 1 else "zf10"
    start_slice = sys.argv[2] if len(sys.argv) > 2 else "1"

    RESULT_PATH = rf"C:\Users\Arjun\OneDrive\Desktop\SlicingBias\STAIR\results\{timepoint}"
    CHECKPOINT_PATH = rf"{RESULT_PATH}\{timepoint}_emb_aligned_checkpoint.h5ad"
    ATTENTION_CSV = f"{RESULT_PATH}/{timepoint}_slice_attention.csv"

    print(f"Loading {CHECKPOINT_PATH}...")
    adata = sc.read_h5ad(CHECKPOINT_PATH)

    adata, dists = reconstruct_z_via_attention(adata, ATTENTION_CSV, start_slice)

    
    adata.obs['z_naive_um'] = (adata.obs['slice'].astype(int) - 1) * REAL_SECTION_THICKNESS_UM

    xy = adata.obsm['transform_fine'] if 'transform_fine' in adata.obsm else adata.obsm['spatial']

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(xy[:, 0], xy[:, 1], adata.obs['z_naive_um'], s=1,
                c=adata.obs['z_naive_um'], cmap='viridis')
    ax1.set_title(f'{timepoint}: naive uniform z (index x {REAL_SECTION_THICKNESS_UM}um)')
    ax1.set_zlabel('z (um)')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(xy[:, 0], xy[:, 1], adata.obs['z_rec_um'], s=1,
                c=adata.obs['z_rec_um'], cmap='viridis')
    ax2.set_title(f'{timepoint}: STAIR attention-ordered z (real {REAL_SECTION_THICKNESS_UM}um scale)')
    ax2.set_zlabel('z (um)')
    
    adata.obs['_pos'] = np.arange(len(adata.obs))
    centroids = adata.obs.groupby('slice', observed=True).apply(lambda g: pd.Series({
        'x': xy[g['_pos'].values, 0].mean(),
        'y': xy[g['_pos'].values, 1].mean(),
        'z': g['z_rec_um'].mean()
    }))
    for slice_id, row in centroids.iterrows():
        ax2.text(row['x'], row['y'], row['z'], str(slice_id), fontsize=8)
    

    out_path = f"{RESULT_PATH}/{timepoint}_z_comparison_naive_vs_attention.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved comparison: {out_path}")

    adata.write_h5ad(f"{RESULT_PATH}/{timepoint}_attention_reconstructed.h5ad")
    
print(adata.obs['slice'].value_counts().sort_index())
