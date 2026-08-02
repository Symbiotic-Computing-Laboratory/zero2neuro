"""
plot_true_vs_pred.py
--------------------
Plot true vs predicted Culex pipiens counts over epiweek
for a chosen year, averaged across all validation locations.

Requires the network to have been trained with --log_validation_set
(and --log_training_set / --log_testing_set) so that results pkl contains 
'outs_validation' and 'predict_validation' etc.

Usage:
    python plot_true_vs_pred.py

To pick a different year, change YEAR below.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===========================================================================
# USER SETTINGS  ← change these
# ===========================================================================

RESULTS_PKL = 'results/mosquito_tensor_R00_results.pkl'

# Which year to plot?
# The data spans 2018-2024 (7 years total, index 0-6):
#   index 0 = 2018
#   index 1 = 2019
#   index 2 = 2020
#   index 3 = 2021
#   index 4 = 2022  
#   index 5 = 2023
#   index 6 = 2024
YEAR_IDX = 6      # <--- CHANGE THIS to pick a different year

# Which dataset split to plot?
# Choose from: 'training', 'validation', 'testing'
DATA_SPLIT  = 'training'

# Which location to plot?
# pick form result file
# infuteure define ID (I am not sure)

LOC_ID = 'R124-C379'

N_YEARS     = 7       # total years in the dataset (2018-2024)
FIRST_YEAR  = 2018    # first year in the dataset
EPIWEEKS    = list(range(19, 49))   # 30 epiweek labels

LOG_TO_WANDB = True    # set True to also log the plot to Weights & Biases

# ===========================================================================

year = FIRST_YEAR + YEAR_IDX
print(f'Loading {RESULTS_PKL} ...')

with open(RESULTS_PKL, 'rb') as f:
    res = pickle.load(f)

# ---------------------------------------------------------------------------
# Data Extraction & Auto-Detection
# ---------------------------------------------------------------------------

# 1. ALWAYS extract the partition sum based on DATA_SPLIT
outs_key = f'outs_{DATA_SPLIT}'
pred_key = f'predict_{DATA_SPLIT}'

assert outs_key in res, (
    f'{outs_key} not found in results pkl.\n'
    f'Make sure you trained with --log_{DATA_SPLIT}_set in experiment.txt '
    'and re-ran the training command.'
)

outs = res[outs_key]       
pred = res[pred_key]    

N_examples = outs.shape[0]
print(f'outs shape   : {outs.shape}')
print(f'pred shape   : {pred.shape}')
print(f'Plotting year: {year}  (index {YEAR_IDX} of {N_YEARS})')

year_mask = (np.arange(N_examples) % N_YEARS) == YEAR_IDX
true_year_part = outs[year_mask]   
pred_year_part = pred[year_mask]   

n_locs_part = true_year_part.shape[0]
print(f'Locations in {DATA_SPLIT} fold for this year: {n_locs_part}')

# Sum across locations → shape: (30,)
y_true_part = true_year_part.sum(axis=0).squeeze()
y_pred_part = pred_year_part.sum(axis=0).squeeze()
plot_label_part = 'Sum across locations'

# 2. If LOC_ID is provided, extract the specific location
y_true_loc = None
y_pred_loc = None
loc_split = None

if LOC_ID is not None:
    # Load the original dataset pickle for this location to get its true values
    loc_pkl_path = f'dataset/{LOC_ID}.pkl'
    try:
        with open(loc_pkl_path, 'rb') as f_loc:
            loc_data = pickle.load(f_loc)
    except FileNotFoundError:
        raise ValueError(f"Could not find dataset file for '{LOC_ID}' at {loc_pkl_path}")
        
    # Extract the specific year's true values for this location
    target_true = loc_data['Culex.pipiens'][YEAR_IDX, :, 0]
    
    # Search through ALL splits to find which one contains this location
    found_split = None
    for split in ['training', 'validation', 'testing']:
        o_key = f'outs_{split}'
        if o_key not in res:
            continue
            
        outs_split = res[o_key]
        pred_split = res[f'predict_{split}']
        
        N_ex = outs_split.shape[0]
        y_mask = (np.arange(N_ex) % N_YEARS) == YEAR_IDX
        true_yr = outs_split[y_mask]
        pred_yr = pred_split[y_mask]
        
        for i in range(true_yr.shape[0]):
            # We use allclose to handle floating point comparisons safely
            if np.allclose(true_yr[i, :, 0], target_true, equal_nan=True):
                y_true_loc = true_yr[i].squeeze()
                y_pred_loc = pred_yr[i].squeeze()
                loc_split = split
                found_split = split
                break
        if found_split:
            break
            
    if not found_split:
        raise ValueError(f"Station '{LOC_ID}' was not found in ANY dataset split in the results file!")
        
    print(f"Auto-detected Station '{LOC_ID}' in the '{loc_split}' set.")
    plot_label_loc = f'Station {LOC_ID}'


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
BG      = '#0f1117'
PANEL   = '#1a1d2e'
GRID_C  = '#2a2d3e'
TRUE_C  = '#4fc3f7'   # sky blue
PRED_C  = '#f06292'   # pink
TEXT_C  = '#e0e0e0'

# Setup plot function to avoid repeating code
def plot_on_axis(ax, y_t, y_p, label, title, n_locs):
    ax.set_facecolor(PANEL)
    ax.plot(EPIWEEKS, y_t, color=TRUE_C, linewidth=2.5,
            marker='o', markersize=5, label=f'True ({label})')
    ax.plot(EPIWEEKS, y_p, color=PRED_C, linewidth=2.5,
            marker='s', markersize=5, linestyle='--',
            label=f'Predicted ({label})')
    ax.fill_between(EPIWEEKS, y_t, y_p,
                    alpha=0.15, color='white', label='Error region')
    ax.set_title(title, color=TEXT_C, fontsize=13, pad=12)
    ax.set_xlabel('Epiweek', color=TEXT_C, fontsize=11)
    ax.set_ylabel('Mosquito count', color=TEXT_C, fontsize=11)
    ax.tick_params(colors=TEXT_C)
    ax.spines[:].set_color(GRID_C)
    ax.grid(color=GRID_C, linewidth=0.8, linestyle='--')
    ax.set_xticks(EPIWEEKS)
    ax.set_xticklabels([str(e) for e in EPIWEEKS], fontsize=8, color=TEXT_C)
    ax.legend(facecolor=PANEL, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=10)
    
    mae = np.mean(np.abs(y_t - y_p))
    mae_label = f'MAE = {mae:.2f}  |  n_locs = {n_locs}'
    ax.text(0.98, 0.96, mae_label, transform=ax.transAxes, ha='right', va='top',
            color='#ffd54f', fontsize=9, bbox=dict(facecolor=PANEL, edgecolor='none', alpha=0.7))

if LOC_ID is not None:
    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    
    # Left Subplot: Partition
    title_part = f'Culex pipiens  |  Year {year}  |  {DATA_SPLIT.capitalize()} set'
    plot_on_axis(axes[0], y_true_part, y_pred_part, plot_label_part, title_part, n_locs_part)
    
    # Right Subplot: Location
    title_loc = f'Culex pipiens  |  Year {year}  |  {loc_split.capitalize()} set  |  {LOC_ID}'
    plot_on_axis(axes[1], y_true_loc, y_pred_loc, plot_label_loc, title_loc, 1)
else:
    # Plot single partition
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    title_part = f'Culex pipiens  |  Year {year}  |  {DATA_SPLIT.capitalize()} set'
    plot_on_axis(ax, y_true_part, y_pred_part, plot_label_part, title_part, n_locs_part)

loc_str = f'_{LOC_ID}' if LOC_ID is not None else '_sum'
out_path = f'true_vs_pred_{year}_{DATA_SPLIT}{loc_str}.png'
plt.tight_layout(pad=1.5)
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
print(f'Saved -> {out_path}')

# ---------------------------------------------------------------------------
# Optional: log to Weights & Biases
# ---------------------------------------------------------------------------
if LOG_TO_WANDB:
    import wandb
    
    # 1. Initialize a new "run" in your project. This acts like a new experiment 
    #    or log entry specifically for storing this plot.
    run = wandb.init(project='Mosquito_Prediction',
                     name=f'true_vs_pred_{year}_{DATA_SPLIT}{loc_str}',
                     job_type='analysis')
                     
    # 2. Upload the PNG image to Weights & Biases. 
    #    wandb.Image() converts the local file into a format W&B can display on the dashboard.
    wandb.log({f'true_vs_pred_{year}_{DATA_SPLIT}{loc_str}': wandb.Image(out_path)})
    
    # 3. Close the run so it doesn't hang in the background. 
    #    This tells W&B we are finished uploading for this script.
    wandb.finish()
    
    print('Logged to Weights & Biases.')
