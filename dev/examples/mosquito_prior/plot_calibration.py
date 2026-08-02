"""
plot_calibration.py
-------------------
Plots Culex pipiens true counts vs epiweek for a chosen year,
across all trap locations. Helps calibrate MAE intuition by showing
the actual variance of the target signal.

Usage:
    python plot_calibration.py --year 2022
    python plot_calibration.py --year 2023 --col Culex.pipiens
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')   # headless — saves PNG without needing a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str,
                    default=r'C:\Users\shaf0043\Desktop\Mosquito_reserve'
                             r'\Mosquito_Data_Merged_with_Cov_Data_clean.xlsx')
parser.add_argument('--year', type=int, default=2022,
                    help='Calendar year to highlight (default: 2022)')
parser.add_argument('--col', type=str, default='Culex.pipiens',
                    help='Target column to plot')
parser.add_argument('--out', type=str, default='calibration_plot.png',
                    help='Output PNG path')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print(f'Loading {args.data_file} ...')
df = pd.read_excel(args.data_file)

assert args.col in df.columns, f'Column "{args.col}" not found in data.'
assert args.year in df['Year'].unique(), \
    f'Year {args.year} not in data. Available: {sorted(df["Year"].unique().tolist())}'

epiweeks_all = sorted(df['Epiweek'].unique().tolist())

# ---------------------------------------------------------------------------
# Global stats (all years, all locations) — for context
# ---------------------------------------------------------------------------
all_vals = df[args.col].dropna()
global_mean = all_vals.mean()
global_std  = all_vals.std()
global_max  = all_vals.max()
global_min  = all_vals.min()

print(f'\nGlobal {args.col} stats (all years, all locations):')
print(f'  min={global_min:.1f}  max={global_max:.1f}  '
      f'mean={global_mean:.1f}  std={global_std:.1f}')

# ---------------------------------------------------------------------------
# Per-epiweek stats for selected year
# ---------------------------------------------------------------------------
yr_df = df[df['Year'] == args.year][['Epiweek', args.col]].dropna()
ew_groups = yr_df.groupby('Epiweek')[args.col]

ew_mean   = ew_groups.mean()
ew_median = ew_groups.median()
ew_q1     = ew_groups.quantile(0.25)
ew_q3     = ew_groups.quantile(0.75)
ew_min    = ew_groups.min()
ew_max    = ew_groups.max()

print(f'\nYear {args.year} epiweek range: '
      f'{yr_df["Epiweek"].min()} – {yr_df["Epiweek"].max()}')
print(f'Year {args.year} {args.col} stats:')
print(f'  min={yr_df[args.col].min():.1f}  max={yr_df[args.col].max():.1f}  '
      f'mean={yr_df[args.col].mean():.1f}  std={yr_df[args.col].std():.1f}')

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10),
                         gridspec_kw={'height_ratios': [3, 1]})
fig.patch.set_facecolor('#0f1117')

COLOR_BG    = '#0f1117'
COLOR_PANEL = '#1a1d2e'
COLOR_GRID  = '#2a2d3e'
COLOR_MAIN  = '#4fc3f7'
COLOR_IQR   = '#1565c0'
COLOR_RANGE = '#0d47a1'
COLOR_MEAN  = '#f06292'
COLOR_MAE   = '#ffd54f'
COLOR_TEXT  = '#e0e0e0'

ew_list = sorted(ew_groups.groups.keys())

# ---- Top panel: distribution per epiweek -----------------------------------
ax = axes[0]
ax.set_facecolor(COLOR_PANEL)

# Shaded range (min–max)
ax.fill_between(ew_list,
                [ew_min[ew] for ew in ew_list],
                [ew_max[ew] for ew in ew_list],
                color=COLOR_RANGE, alpha=0.3, label='Min–Max range')

# IQR band
ax.fill_between(ew_list,
                [ew_q1[ew] for ew in ew_list],
                [ew_q3[ew] for ew in ew_list],
                color=COLOR_IQR, alpha=0.5, label='IQR (25–75%)')

# Median line
ax.plot(ew_list, [ew_median[ew] for ew in ew_list],
        color=COLOR_MAIN, linewidth=2.5, marker='o', markersize=5,
        label='Median across locations', zorder=5)

# Mean line
ax.plot(ew_list, [ew_mean[ew] for ew in ew_list],
        color=COLOR_MEAN, linewidth=1.5, linestyle='--',
        label='Mean across locations', zorder=5)

# Individual trap scatter
for ew, grp in ew_groups:
    jitter = np.random.uniform(-0.15, 0.15, size=len(grp))
    ax.scatter([ew + j for j in jitter], grp.values,
               color=COLOR_MAIN, alpha=0.25, s=18, zorder=3)

# MAE reference bands (±MAE from the mean line)
mae_value = 3.84   # from your last training run
for ew in ew_list:
    mu = ew_mean[ew]
    ax.plot([ew - 0.3, ew + 0.3], [mu + mae_value, mu + mae_value],
            color=COLOR_MAE, linewidth=1.2, alpha=0.8)
    ax.plot([ew - 0.3, ew + 0.3], [mu - mae_value, mu - mae_value],
            color=COLOR_MAE, linewidth=1.2, alpha=0.8)
ax.plot([], [], color=COLOR_MAE, linewidth=1.5,
        label=f'±MAE = ±{mae_value} (last run)')

ax.set_title(f'{args.col}  —  Year {args.year}  |  '
             f'True counts per epiweek across all trap locations',
             color=COLOR_TEXT, fontsize=13, pad=12)
ax.set_ylabel('Mosquito count', color=COLOR_TEXT, fontsize=11)
ax.tick_params(colors=COLOR_TEXT)
ax.spines[:].set_color(COLOR_GRID)
ax.grid(axis='y', color=COLOR_GRID, linewidth=0.8)
ax.set_xlim(min(ew_list) - 0.7, max(ew_list) + 0.7)
ax.set_xticks(ew_list)
ax.set_xticklabels([str(e) for e in ew_list], fontsize=8, color=COLOR_TEXT)
legend = ax.legend(facecolor=COLOR_PANEL, edgecolor=COLOR_GRID,
                   labelcolor=COLOR_TEXT, fontsize=9, loc='upper right')

# Global reference text
stats_txt = (f'Global (all yrs)  min={global_min:.0f}  max={global_max:.0f}  '
             f'mean={global_mean:.1f}  σ={global_std:.1f}')
ax.text(0.01, 0.98, stats_txt, transform=ax.transAxes,
        color='#90caf9', fontsize=8.5, va='top',
        bbox=dict(facecolor=COLOR_PANEL, edgecolor='none', alpha=0.7))

# ---- Bottom panel: count of traps reporting per epiweek -------------------
ax2 = axes[1]
ax2.set_facecolor(COLOR_PANEL)
n_reporting = [ew_groups.count()[ew] for ew in ew_list]
ax2.bar(ew_list, n_reporting, color=COLOR_MAIN, alpha=0.7, width=0.7)
ax2.set_ylabel('# locations\nreporting', color=COLOR_TEXT, fontsize=10)
ax2.set_xlabel('Epiweek', color=COLOR_TEXT, fontsize=11)
ax2.tick_params(colors=COLOR_TEXT)
ax2.spines[:].set_color(COLOR_GRID)
ax2.grid(axis='y', color=COLOR_GRID, linewidth=0.8)
ax2.set_xlim(min(ew_list) - 0.7, max(ew_list) + 0.7)
ax2.set_xticks(ew_list)
ax2.set_xticklabels([str(e) for e in ew_list], fontsize=8, color=COLOR_TEXT)

plt.tight_layout(pad=1.5)
plt.savefig(args.out, dpi=150, bbox_inches='tight',
            facecolor=COLOR_BG)
print(f'\nSaved -> {os.path.abspath(args.out)}')
