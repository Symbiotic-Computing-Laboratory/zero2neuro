import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob

# ===========================================================================
# USER SETTINGS
# ===========================================================================
DATASET_DIR = 'dataset'
YEARS = list(range(2022, 2025))  # 2022 to 2024
DATA_SEED = 1138                   # Default seed from zero2neuro
N_FOLDS = 5                      # Default n_folds
DATA_ROTATION = 0                # Which rotation was used for the split

BG      = '#0f1117'
PANEL   = '#1a1d2e'
TEXT_C  = '#e0e0e0'
GRID_C  = '#2a2d3e'

def main():
    print("Loading locations...")
    
    # 1. Find all location pkl files in the dataset folder
    pkl_files = glob.glob(os.path.join(DATASET_DIR, '*.pkl'))
    locations = []
    for f in pkl_files:
        basename = os.path.basename(f)
        if not basename.startswith('fold_'):
            locations.append(basename.replace('.pkl', ''))
            
    locations = sorted(locations)
    n_locations = len(locations)
    print(f"Found {n_locations} stations.")
    
    if n_locations == 0:
        print("No station .pkl files found. Have you run the data prep script?")
        return
    
    # 2. Hardcoded Splits
    custom_train = ['R117-C385', 'R118-C380', 'R132-C372', 'R132-C381', 'R135-C383']
    custom_val = ['R140-C378']
    custom_test = ['R120-C374', 'R121-C369', 'R122-C370', 'R127-C370', 'R127-C383', 
                   'R128-C376', 'R130-C366', 'R130-C376', 'R132-C380', 'R149-C343']
                   
    splits = {
        'training': custom_train,
        'validation': custom_val,
        'testing': custom_test,
    }
    
    # 3. Process each split and plot
    for split_name, locs_in_split in splits.items():
        print(f"\nProcessing {split_name} set ({len(locs_in_split)} locations):")
        print(f"Locations: {locs_in_split}")
        if len(locs_in_split) == 0:
            print(f"No locations in {split_name} set.")
            continue
            
        print(f"\nProcessing {split_name} set ({len(locs_in_split)} locations)...")
        
        # Matrix shape: (n_years, n_locs)
        heatmap_data = np.zeros((len(YEARS), len(locs_in_split)))
        heatmap_nonzero = np.zeros((len(YEARS), len(locs_in_split)))
        
        for x_idx, loc in enumerate(locs_in_split):
            pkl_path = os.path.join(DATASET_DIR, f"{loc}.pkl")
            with open(pkl_path, 'rb') as f:
                loc_data = pickle.load(f)
            
            # data_mask shape: (n_years, n_epiweeks, 1)
            mask = loc_data['data_mask']
            target = loc_data['Culex.pipiens']
            
            # Dynamically read the max possible epiweeks from the data shape
            max_epiweeks = mask.shape[1]
            
            # Count how many epiweeks are available per year
            available_epiweeks = np.sum(mask, axis=1).squeeze()
            
            # Count how many of those available epiweeks actually caught > 0 mosquitoes
            nonzero_epiweeks = np.sum((mask == 1) & (target > 0), axis=1).squeeze()
            
            # Fill heatmap column
            heatmap_data[:, x_idx] = available_epiweeks
            heatmap_nonzero[:, x_idx] = nonzero_epiweeks
            
        # 4. Generate Plot
        # Scale figure width based on number of locations so it's not squished, times 2 for two plots
        fig_width = max(12, len(locs_in_split) * 0.8)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 6))
        
        fig.patch.set_facecolor(BG)
        ax1.set_facecolor(PANEL)
        ax2.set_facecolor(PANEL)
        
        # --- First Heatmap (Total Available) ---
        cax1 = ax1.imshow(heatmap_data, cmap='YlGnBu', aspect='auto', vmin=0, vmax=max_epiweeks)
        
        ax1.set_xticks(np.arange(len(locs_in_split)))
        ax1.set_xticklabels(locs_in_split, rotation=90, ha='center', fontsize=9, color=TEXT_C)
        ax1.set_yticks(np.arange(len(YEARS)))
        ax1.set_yticklabels(YEARS, fontsize=10, color=TEXT_C)
        
        for i in range(len(YEARS)):
            for j in range(len(locs_in_split)):
                val = int(heatmap_data[i, j])
                color = "white" if val > (max_epiweeks / 2) else "black"
                ax1.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)
                
        ax1.set_title(f'Valid Data Points', fontsize=14, pad=15, color=TEXT_C)
        ax1.set_ylabel('Year', fontsize=12, color=TEXT_C)
        ax1.set_xlabel('Station Name', fontsize=12, color=TEXT_C, labelpad=10)
        ax1.spines[:].set_color(GRID_C)
        
        # --- Second Heatmap (Non-Zero Only) ---
        cax2 = ax2.imshow(heatmap_nonzero, cmap='YlGnBu', aspect='auto', vmin=0, vmax=max_epiweeks)
        
        ax2.set_xticks(np.arange(len(locs_in_split)))
        ax2.set_xticklabels(locs_in_split, rotation=90, ha='center', fontsize=9, color=TEXT_C)
        ax2.set_yticks(np.arange(len(YEARS)))
        ax2.set_yticklabels(YEARS, fontsize=10, color=TEXT_C)
        
        for i in range(len(YEARS)):
            for j in range(len(locs_in_split)):
                val = int(heatmap_nonzero[i, j])
                color = "white" if val > (max_epiweeks / 2) else "black"
                ax2.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)
                
        ax2.set_title(f'Non-Zero Mosquito Epiweeks', fontsize=14, pad=15, color=TEXT_C)
        ax2.set_xlabel('Station Name', fontsize=12, color=TEXT_C, labelpad=10)
        ax2.spines[:].set_color(GRID_C)
        
        # Figure Title
        fig.suptitle(f'Data Availability: {split_name.capitalize()} Set', fontsize=16, color=TEXT_C, y=1.02)
        
        # Adjust layout manually so the suptitle doesn't overlap before adding colorbar
        plt.tight_layout()
        
        # Shared Colorbar attached to the rightmost axis
        cbar = fig.colorbar(cax2, ax=ax2, pad=0.02)
        cbar.set_label(f'Epiweek Count (Max {max_epiweeks})', color=TEXT_C)
        cbar.ax.yaxis.set_tick_params(color=TEXT_C)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_C)
        
        out_path = f'data_availability_{split_name}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"Saved -> {out_path}")
        plt.close()
        
        # ---------------------------------------------------------------------------
        # Optional: log to Weights & Biases
        # ---------------------------------------------------------------------------
        import wandb
        
        # Initialize a wandb run to log this specific plot
        run = wandb.init(project='Mosquito_Prediction',
                         name=f'data_availability_{split_name}',
                         job_type='data_visualization')
                         
        # Upload the PNG image to Weights & Biases
        wandb.log({f'data_availability_{split_name}': wandb.Image(out_path)})
        
        # Finish the run
        wandb.finish()
        print(f'Logged {split_name} plot to Weights & Biases.')

if __name__ == '__main__':
    main()
