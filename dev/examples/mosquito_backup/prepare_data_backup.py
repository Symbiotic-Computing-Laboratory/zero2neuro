'''
prepare_data.py
---------------
Mosquito dataset preparation for the Zero2Neuro RNN pipeline.

Reads the raw Excel source and writes per-location and per-fold pickle files
that implement "Scenario A":

  * Each location contributes N_YEARS independent training examples —
    one example per calendar year.
  * Each example is a fixed-length sequence of N_EPIWEEKS timesteps.
  * Each timestep carries a feature vector of length n_features.

Pickle key layout (per key, per location file):
  shape  (N_YEARS, N_EPIWEEKS, 1)  — one scalar value per (year, epiweek) cell.

After zero2neuro's  np.concatenate(axis=-1)  across all input keys the
framework receives input tensors of shape:
  (N_YEARS, N_EPIWEEKS, n_input_features)

which it treats as N_YEARS independent examples, each a sequence of
N_EPIWEEKS timesteps with n_input_features features —— exactly the
(batch, timesteps, features) contract the Keras RNN layers expect.

Missing cells (no observation for that year/epiweek at a location) are
zero-padded; the accompanying  data_mask  key encodes which cells are real.

Usage
-----
  python prepare_data.py --data_file Mosquito_Data_Merged_with_Cov_Data_clean.xlsx

Run from the mosquito example directory so that the default --dataset_directory
resolves to  ./dataset/.

Author: generated for the Zero2Neuro project
'''

import os
import pickle
import sys

import numpy as np
import pandas as pd

# Import the shared Zero2Neuro argument parser so we reuse all standard
# argument definitions (--data_file, --dataset_directory, --data_n_folds,
# --data_seed, --verbose, ...) without duplicating them here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from parser import create_parser as _z2n_create_parser




# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Columns in the Excel file that do NOT carry feature signal
META_COLS = ['Unnamed: 0', 'Tag', 'Epiweek', 'Location.Name', 'Year']



# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_location_grid(loc_df, feature_cols, years, epiweeks):
    '''
    Build the (n_years × n_epiweeks × 1) numpy grid for every feature at
    a single trap location.

    Parameters
    ----------
    loc_df       : pd.DataFrame
        Rows belonging to exactly one Location.Name (pre-filtered by caller).
    feature_cols : list[str]
        All biological/environmental column names to extract (inputs + output).
    years        : list[int]
        The fully contiguous list of years discovered in the dataset.
    epiweeks     : list[int]
        The fully contiguous list of epiweeks discovered in the dataset.

    Returns
    -------
    grids : dict[str -> np.ndarray]
        Keys  = feature names.
        Values = float32 arrays of shape (n_years, n_epiweeks, 1).
        Cells with no observation remain 0.0.
    '''
    n_years = len(years)
    n_epiweeks = len(epiweeks)

    # Allocate zero-filled grids — zero is the padding value for absent cells.
    # Missing (year, epiweek) cells remain 0.0 (simple zero-padding, no mask).
    grids = {
        col: np.zeros((n_years, n_epiweeks, 1), dtype=np.float32)
        for col in feature_cols
    }
    # Binary mask: 1.0 where a real observation exists, 0.0 where zero-padded.
    grids['data_mask'] = np.zeros((n_years, n_epiweeks, 1), dtype=np.float32)
    # Note: year and epiweek are already encoded as axis-0 and axis-1 of every
    # array — they do NOT need to be repeated as extra feature-value keys.

    # Build a (Year, Epiweek) multi-index for O(1) cell lookup
    indexed = loc_df.set_index(['Year', 'Epiweek'])

    for yi, year in enumerate(years):
        for ei, epiweek in enumerate(epiweeks):

            # Skip cells for which no observation exists at this location
            if (year, epiweek) not in indexed.index:
                continue   # leave value at 0.0 (zero-padding)

            row = indexed.loc[(year, epiweek)]

            # When a location has multiple records for the same (year, epiweek)
            # (data quality issue) take the first row to stay deterministic
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            # Write observed feature values into the grid
            for col in feature_cols:
                grids[col][yi, ei, 0] = float(row[col])
            grids['data_mask'][yi, ei, 0] = 1.0   # mark cell as real

    return grids


# ---------------------------------------------------------------------------
# Pickle I/O
# ---------------------------------------------------------------------------

def save_pkl(data_dict, path):
    '''
    Serialise a feature-grid dictionary to a pickle file at the given path.

    Parameters
    ----------
    data_dict : dict[str -> np.ndarray]
    path      : str   — full file path including filename and .pkl extension
    '''
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as fp:
        pickle.dump(data_dict, fp)


def build_fold_dict(location_grids, all_pkl_keys):
    '''
    Assemble one fold's pickle dictionary by stacking all location grids
    along axis 0.

    Each per-location key has shape (N_YEARS, N_EPIWEEKS, 1).
    The resulting fold key has shape (n_locs * N_YEARS, N_EPIWEEKS, 1),
    which the Zero2Neuro framework interprets as  n_locs * N_YEARS  independent
    examples, each a sequence of N_EPIWEEKS timesteps.

    Parameters
    ----------
    location_grids : list[dict]
        One grid dict per location assigned to this fold.
    all_pkl_keys   : list[str]
        Ordered list of keys present in every grid dict.

    Returns
    -------
    fold : dict[str -> np.ndarray]
    '''
    fold = {}
    for key in all_pkl_keys:
        # Stack along axis=0: (N_YEARS, N_EW, 1) * n_locs -> (n_locs*N_YEARS, N_EW, 1)
        fold[key] = np.concatenate([lg[key] for lg in location_grids], axis=0)
    return fold


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_fold_summary(fold_idx, fold_dict, n_locations, n_years, output_col, verbose):
    '''Print a one-line summary for a completed fold pkl.'''
    n_examples = fold_dict[output_col].shape[0]   # n_locs * n_years
    key_shape  = fold_dict[output_col].shape
    if verbose >= 1:
        print(f'  fold_{fold_idx}.pkl  |  {n_locations} locations '
              f'× {n_years} years = {n_examples} examples  |  '
              f'key shape: {key_shape}')
    else:
        print(f'  fold_{fold_idx}.pkl  ->  {n_examples} examples, shape {key_shape}')



# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = _z2n_create_parser(description='Mosquito dataset preparation')
    args   = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load and validate the raw Excel file
    # -----------------------------------------------------------------------
    print(f'\n[1] Loading: {args.data_file}')
    df = pd.read_excel(args.data_file)
    print(f'    Rows: {len(df):,}  |  Columns: {len(df.columns)}')

    # Verify expected columns exist before anything else runs
    required_meta = ['Year', 'Epiweek', 'Location.Name']
    missing = set(required_meta) - set(df.columns)
    assert not missing, f'Required columns missing from Excel: {missing}'
    assert args.data_outputs, "You must specify at least one --data_outputs column."
    for out_col in args.data_outputs:
        assert out_col in df.columns, f'Output column "{out_col}" not found in the data file.'

    # -----------------------------------------------------------------------
    # 2. Dynamically extract the time grid boundaries from the data
    # -----------------------------------------------------------------------
    # Using range(min, max+1) ensures the 3D tensor grid is perfectly
    # sequential without any missing weeks/years skipped in the structure.
    years    = list(range(int(df['Year'].min()), int(df['Year'].max()) + 1))
    epiweeks = list(range(int(df['Epiweek'].min()), int(df['Epiweek'].max()) + 1))
    n_years    = len(years)
    n_epiweeks = len(epiweeks)
    
    print(f'    Time grid: {n_years} years ({years[0]}..{years[-1]}) × '
          f'{n_epiweeks} epiweeks ({epiweeks[0]}..{epiweeks[-1]})')

    # -----------------------------------------------------------------------
    # 3. Derive feature column lists
    # -----------------------------------------------------------------------
    # All signal columns (excludes row-index metadata)
    feature_cols = [c for c in df.columns if c not in META_COLS]

    # Input features = every feature column except the prediction target(s)
    input_cols   = [c for c in feature_cols if c not in args.data_outputs]

    # data_mask: binary array (1.0 = real observation, 0.0 = zero-padded cell).
    # Listed last so it is appended after all bio/env feature keys.
    aux_keys = ['data_mask']

    # Complete ordered list of keys present in every pkl dict
    all_pkl_keys = feature_cols + aux_keys

    print(f'    Input features: {len(input_cols)}  '
          f'|  Aux keys: {len(aux_keys)}  '
          f'|  Outputs: {len(args.data_outputs)}')

    # -----------------------------------------------------------------------
    # 4. Enumerate locations and assign to folds
    # -----------------------------------------------------------------------
    locations   = sorted(df['Location.Name'].unique())
    n_locations = len(locations)
    print(f'\n[2] Locations found: {n_locations}')
    print(f'    Folds: {args.data_n_folds}  |  Seed: {args.data_seed}')

    # Reproducible shuffle -> round-robin fold assignment
    rng      = np.random.default_rng(args.data_seed)
    shuffled = locations.copy()
    rng.shuffle(shuffled)

    # Map location name -> fold index (0 .. n_folds-1)
    loc_to_fold = {loc: i % args.data_n_folds for i, loc in enumerate(shuffled)}

    # -----------------------------------------------------------------------
    # 5. Build per-location grids and save individual pkl files
    # -----------------------------------------------------------------------
    os.makedirs(args.dataset_directory, exist_ok=True)
    print(f'\n[3] Writing per-location pkl files -> {args.dataset_directory}/')

    fold_grids = {i: [] for i in range(args.data_n_folds)}  # accumulate by fold

    for loc in locations:
        loc_df = df[df['Location.Name'] == loc]
        grids  = build_location_grid(loc_df, feature_cols, years, epiweeks)

        # Individual location pkl — useful for inference on a new trap location
        loc_path = os.path.join(args.dataset_directory, f'{loc}.pkl')
        save_pkl(grids, loc_path)

        fold_idx     = loc_to_fold[loc]
        n_real_cells = int(sum(
            (grids[k] != 0).any(axis=-1).sum()
            for k in [list(feature_cols)[0]]   # proxy: count non-zero rows in first key
        ))

        if args.verbose >= 1:
            print(f'  {loc:<20}  fold={fold_idx}')

        fold_grids[fold_idx].append(grids)

    n_per_loc_pkls = n_locations
    print(f'    {n_per_loc_pkls} per-location pkl files written.')

    # -----------------------------------------------------------------------
    # 6. Assemble and save fold pkl files
    # -----------------------------------------------------------------------
    print(f'\n[4] Writing fold pkl files ({args.data_n_folds} folds) -> '
          f'{args.dataset_directory}/')

    for fold_idx in range(args.data_n_folds):
        location_grids = fold_grids[fold_idx]
        n_locs_in_fold = len(location_grids)

        assert n_locs_in_fold > 0, (
            f'Fold {fold_idx} received no locations. '
            f'Increase --data_n_folds or check the data.'
        )

        fold_dict = build_fold_dict(location_grids, all_pkl_keys)
        fold_path = os.path.join(args.dataset_directory, f'fold_{fold_idx}.pkl')
        save_pkl(fold_dict, fold_path)

        print_fold_summary(fold_idx, fold_dict, n_locs_in_fold, n_years,
                           args.data_outputs[0], args.verbose)

    print('\nDone.\n')


if __name__ == '__main__':
    main()
