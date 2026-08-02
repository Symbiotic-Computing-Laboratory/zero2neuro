"""
plot_true_vs_pred_grid.py
-------------------------
Plots a massive 7-Year by N-Location grid of true vs predicted mosquito counts.
The far left column ("All") shows the summed prediction across all locations for that year.
"""

import pickle
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===========================================================================
# USER SETTINGS
# ===========================================================================

RESULTS_PKL = 'results/mosquito_tensor_R00_results.pkl'
DATASET_DIR = 'dataset'

# Which dataset split to plot? ('training', 'validation', 'testing')
DATA_SPLIT  = 'training'

N_YEARS     = 3     # total years in the dataset (2022-2024)
FIRST_YEAR  = 2022    # first year in the dataset
EPIWEEKS    = list(range(19, 49))   # 30 epiweek labels

# Fold reconstruction settings (Must match zero2neuro defaults)
DATA_SEED = 1138
N_FOLDS = 5
DATA_ROTATION = 0

LOG_TO_WANDB = True

BG      = '#0f1117'
PANEL   = '#1a1d2e'
GRID_C  = '#2a2d3e'
TRUE_C  = '#4fc3f7'   # sky blue
PRED_C  = '#f06292'   # pink
TEXT_C  = '#e0e0e0'

# ===========================================================================

def main():
    print(f'Loading {RESULTS_PKL} ...')
    with open(RESULTS_PKL, 'rb') as f:
        res = pickle.load(f)

    outs_key = f'outs_{DATA_SPLIT}'
    pred_key = f'predict_{DATA_SPLIT}'

    if outs_key not in res:
        raise ValueError(f'{outs_key} not found in results pkl. Did you log the {DATA_SPLIT} set?')

    outs = res[outs_key]       
    pred = res[pred_key]    

    # ---------------------------------------------------------------------------
    # Reconstruct Location Ordering
    # ---------------------------------------------------------------------------
    print("Reconstructing dataset split ordering...")
    # 2. Hardcoded Splits
    custom_train = ['R117-C385', 'R118-C380', 'R132-C372', 'R132-C381', 'R135-C383', 'R140-C378']
                   
    loc_to_fold = {}
    for i, loc in enumerate(custom_train):
        loc_to_fold[loc] = 0  # Fold 0
        
    active_locations = sorted(custom_train)

    if DATA_SPLIT == 'training':
        target_folds = [0]
    elif DATA_SPLIT == 'validation':
        target_folds = [3]
    else:
        target_folds = [4]

    ordered_locs = []
    for fold_idx in target_folds:
        # We must append them exactly in the alphabetical order they were processed 
        # in prepare_data_prior_selected.py (which iterated over sorted active_locations)
        fold_locs = [loc for loc in active_locations if loc_to_fold[loc] == fold_idx]
        ordered_locs.extend(fold_locs)

    n_locs = len(ordered_locs)
    expected_rows = n_locs * N_YEARS
    print(f"Locations in {DATA_SPLIT} split: {n_locs}")
    
    if outs.shape[0] != expected_rows:
        print(f"WARNING: The results.pkl has {outs.shape[0]} rows, but the reconstruction expects {expected_rows}.")
        print("If you used the v6 data masking threshold, this grid script might not align properly if years were deleted.")

    # ---------------------------------------------------------------------------
    # Generate Grid
    # ---------------------------------------------------------------------------
    n_cols = n_locs + 1  # +1 for the "ALL" column
    n_rows = N_YEARS
    
    # 2.5 inches per column, 2.5 inches per row
    fig_width = max(10, n_cols * 2.5)
    fig_height = max(5, n_rows * 2.5)
    
    print(f"Generating massive {n_rows} x {n_cols} grid (Size: {fig_width}\" x {fig_height}\")...")
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_width, fig_height), squeeze=False)
    fig.patch.set_facecolor(BG)

    for y in range(N_YEARS):
        year_label = FIRST_YEAR + y
        
        all_true = np.zeros(len(EPIWEEKS))
        all_pred = np.zeros(len(EPIWEEKS))
        
        # Plot individual locations
        for c_idx, loc in enumerate(ordered_locs):
            # Because arrays are stacked fold by fold, and alphabetically inside folds:
            # The row index for a specific location and year is:
            loc_row_idx = c_idx * N_YEARS + y
            
            # Extract data
            y_t = outs[loc_row_idx, :, 0]
            y_p = pred[loc_row_idx, :, 0]
            
            all_true += y_t
            all_pred += y_p
            
            ax = axes[y, c_idx + 1]
            is_bottom = (y == N_YEARS - 1)
            plot_on_axis(ax, y_t, y_p, is_bottom)
            
            # Put Location Title on the top row only
            if y == 0:
                ax.set_title(loc, color=TEXT_C, fontsize=12, pad=10)

        # Plot "ALL" column
        ax_all = axes[y, 0]
        is_bottom_all = (y == N_YEARS - 1)
        plot_on_axis(ax_all, all_true, all_pred, is_bottom_all)
        
        if y == 0:
            ax_all.set_title("ALL LOCATIONS", color='#ffd54f', fontsize=14, fontweight='bold', pad=10)
            ax_all.legend(facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=10, loc='upper right')
            
        # Put Year Label on the Y-Axis of the "ALL" column
        ax_all.set_ylabel(f'Year {year_label}', color=TEXT_C, fontsize=14, fontweight='bold', labelpad=15)

    # Global Formatting
    fig.suptitle(f'True vs Predicted Culex pipiens  |  {DATA_SPLIT.capitalize()} Set', 
                 fontsize=24, color=TEXT_C, y=1.02)
                 
    plt.tight_layout()
    out_path = f'true_vs_pred_grid_{DATA_SPLIT}.png'
    print(f"Saving high-resolution PNG to {out_path}...")
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print("Done!")

    if LOG_TO_WANDB:
        import wandb
        run = wandb.init(project='Mosquito_Prediction',
                         name=f'true_vs_pred_grid_{DATA_SPLIT}',
                         job_type='analysis')
        wandb.log({f'true_vs_pred_grid_{DATA_SPLIT}': wandb.Image(out_path)})
        wandb.finish()
        print('Logged to Weights & Biases.')

def plot_on_axis(ax, y_t, y_p, is_bottom_row):
    ax.set_facecolor(PANEL)
    ax.plot(EPIWEEKS, y_t, color=TRUE_C, linewidth=2.0, label='True')
    ax.plot(EPIWEEKS, y_p, color=PRED_C, linewidth=2.0, linestyle='--', label='Pred')
    ax.fill_between(EPIWEEKS, y_t, y_p, alpha=0.15, color='white')
    
    ax.tick_params(colors=TEXT_C, labelsize=9)
    ax.spines[:].set_color(GRID_C)
    ax.grid(color=GRID_C, linewidth=0.5, linestyle=':')
    
    # Custom ticks
    ticks = [20, 25, 30, 35, 40, 45]
    ax.set_xticks(ticks)
    if is_bottom_row:
        ax.set_xticklabels([str(t) for t in ticks], color=TEXT_C)
        ax.set_xlabel('Epiweek', color=TEXT_C, fontsize=11)
    else:
        ax.set_xticklabels([])

if __name__ == '__main__':
    main()
