'''
prepare_data_prior_v2.py
---------------
Mosquito dataset preparation for the Zero2Neuro RNN pipeline.

Reads the raw Excel source and writes per-location and per-fold pickle files.
Includes the generation of an autoregressive prior feature (target shifted by t+1)
that properly respects the Year boundaries to prevent winter gaps from bleeding
into the Spring predictions.

 Pickle key layout (per key, per location file):
   shape  (N_YEARS, N_EPIWEEKS, 1)  — one scalar value per (year, epiweek) cell.
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

def build_location_grid(loc_df, feature_cols, years, epiweeks, data_outputs, min_valid_epiweeks):
    '''
    Build the numpy grid for every feature at a single trap location,
    dropping any years that don't meet the min_valid_epiweeks threshold.
    '''
    n_years = len(years)
    n_epiweeks = len(epiweeks)

    # Allocate zero-filled grids — zero is the padding value for absent cells.
    grids = {
        col: np.zeros((n_years, n_epiweeks, 1), dtype=np.float32)
        for col in feature_cols
    }
    # Binary mask: 1.0 where a real observation exists, 0.0 where zero-padded.
    grids['data_mask'] = np.zeros((n_years, n_epiweeks, 1), dtype=np.float32)

    # Build a (Year, Epiweek) multi-index for O(1) cell lookup
    indexed = loc_df.set_index(['Year', 'Epiweek'])
    
    valid_years_indices = []

    for yi, year in enumerate(years):
        valid_count = 0
        for ei, epiweek in enumerate(epiweeks):
            # Skip cells for which no observation exists at this location
            if (year, epiweek) not in indexed.index:
                continue   # leave feature values at 0.0 (zero-padding)

            valid_count += 1
            row = indexed.loc[(year, epiweek)]

            # When a location has multiple records for the same (year, epiweek)
            # (data quality issue) take the first row to stay deterministic
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            # Write observed feature values into the grid
            for col in feature_cols:
                grids[col][yi, ei, 0] = float(row[col])
            grids['data_mask'][yi, ei, 0] = 1.0   # mark cell as real
            
        if valid_count >= min_valid_epiweeks:
            valid_years_indices.append(yi)


    # Generate an autoregressive prior (t-1) for each output column
    # by rolling ONLY along the Epiweek axis (axis=1) and zeroing out the first week
    # of every year to prevent the winter gap discontinuity.
    for out_col in data_outputs:
        prior_col = f"{out_col}.prior"
        original = grids[out_col] # Shape is (n_years, n_epiweeks, 1)
        
        # Roll along the epiweek axis (axis=1)
        # This shifts week 19 to 20, 20 to 21, etc.
        prior_grid = np.roll(original, shift=1, axis=1)
        
        # We must overwrite the first epiweek (index 0) of EVERY year to 0.0
        # so that week 48 of the previous year doesn't wrap around and act as 
        # the prior for week 19.
        if prior_grid.shape[1] > 0:
            prior_grid[:, 0, :] = 0.0
            
        grids[prior_col] = prior_grid

    # Generate an autoregressive prior (t-1) for data_mask
    original_mask = grids['data_mask']
    prior_mask = np.roll(original_mask, shift=1, axis=1)
    
    # Zero out slot 0 of every year just like other priors
    if prior_mask.shape[1] > 0:
        prior_mask[:, 0, :] = 0.0
        
    grids['data_mask.prior'] = prior_mask

    # -----------------------------------------------------------------------
    # Filter out years that didn't meet the threshold
    # Instead of deleting the rows (which breaks modulo 7 math in plotting),
    # we mathematically "remove" the year by forcing its data_mask to 0.0.
    # The neural network uses data_mask as weights, so a 0.0 mask means the
    # network completely ignores the year during training!
    # -----------------------------------------------------------------------
    for yi in range(n_years):
        if yi not in valid_years_indices:
            grids['data_mask'][yi, :, :] = 0.0
            grids['data_mask.prior'][yi, :, :] = 0.0

    return grids


# ---------------------------------------------------------------------------
# Pickle I/O
# ---------------------------------------------------------------------------

def save_pkl(data_dict, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as fp:
        pickle.dump(data_dict, fp)


def build_fold_dict(location_grids, all_pkl_keys):
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
    parser.add_argument('--data_min_valid_epiweeks', type=int, default=15, 
                        help='Minimum valid epiweeks required to keep a year for a location')
    args   = parser.parse_args()

    # -----------------------------------------------------------------------
    # 1. Load and validate the raw Excel file
    # -----------------------------------------------------------------------
    print(f'\n[1] Loading: {args.data_file}')
    df = pd.read_excel(args.data_file)
    print(f'    Rows: {len(df):,}  |  Columns: {len(df.columns)}')

    required_meta = ['Year', 'Epiweek', 'Location.Name']
    missing = set(required_meta) - set(df.columns)
    assert not missing, f'Required columns missing from Excel: {missing}'
    assert args.data_outputs, "You must specify at least one --data_outputs column."
    for out_col in args.data_outputs:
        assert out_col in df.columns, f'Output column "{out_col}" not found in the data file.'

    # -----------------------------------------------------------------------
    # 2. Dynamically extract the time grid boundaries from the data
    # -----------------------------------------------------------------------
    years      = list(range(int(df['Year'].min()), int(df['Year'].max()) + 1))
    epiweeks   = list(range(int(df['Epiweek'].min()), int(df['Epiweek'].max()) + 1))
    n_years    = len(years)
    n_epiweeks = len(epiweeks)

    print(f'    Time grid: {n_years} years ({years[0]}..{years[-1]}) × '
          f'{n_epiweeks} epiweeks ({epiweeks[0]}..{epiweeks[-1]})')

    # -----------------------------------------------------------------------
    # 3. Derive feature column lists
    # -----------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c not in META_COLS]
    input_cols   = [c for c in feature_cols if c not in args.data_outputs]
    aux_keys = ['data_mask', 'data_mask.prior']

    # Include the .prior generated features into the keys to be pickled
    all_pkl_keys = list(feature_cols)
    for out_col in args.data_outputs:
        all_pkl_keys.append(f"{out_col}.prior")
        input_cols.append(f"{out_col}.prior") # Update for reporting
    all_pkl_keys.extend(aux_keys)

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

    rng      = np.random.default_rng(args.data_seed)
    shuffled = locations.copy()
    rng.shuffle(shuffled)

    loc_to_fold = {loc: i % args.data_n_folds for i, loc in enumerate(shuffled)}

    # -----------------------------------------------------------------------
    # 5. Build per-location grids and save individual pkl files
    # -----------------------------------------------------------------------
    os.makedirs(args.dataset_directory, exist_ok=True)
    print(f'\n[3] Writing per-location pkl files -> {args.dataset_directory}/')

    fold_grids = {i: [] for i in range(args.data_n_folds)}  # accumulate by fold

    for loc in locations:
        loc_df = df[df['Location.Name'] == loc]
        grids = build_location_grid(loc_df, feature_cols, years, epiweeks, args.data_outputs, args.data_min_valid_epiweeks)

        # Save the full (unfiltered) grids to the per-location file
        loc_path = os.path.join(args.dataset_directory, f'{loc}.pkl')
        save_pkl(grids, loc_path)

        fold_idx     = loc_to_fold[loc]
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
