'''
prepare_data_prior_selected.py
---------------
Mosquito dataset preparation for the Zero2Neuro RNN pipeline.

Filters the dataset to 2022-2024 and FORCES a manual fold selection:
- Train: ['R117-C385', 'R118-C380', 'R132-C372', 'R132-C381', 'R135-C383']
- Validation: ['R140-C378']
- Test: The original set of test stations that would have been in fold 4.
All other stations are DROPPED.
'''

import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from parser import create_parser as _z2n_create_parser


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

META_COLS = ['Unnamed: 0', 'Tag', 'Epiweek', 'Location.Name', 'Year']


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_location_grid(loc_df, feature_cols, years, epiweeks, data_outputs):
    n_years = len(years)
    n_epiweeks = len(epiweeks)

    grids = {
        col: np.zeros((n_years, n_epiweeks, 1), dtype=np.float32)
        for col in feature_cols
    }
    grids['data_mask'] = np.zeros((n_years, n_epiweeks, 1), dtype=np.float32)

    indexed = loc_df.set_index(['Year', 'Epiweek'])

    for yi, year in enumerate(years):
        for ei, epiweek in enumerate(epiweeks):
            if (year, epiweek) not in indexed.index:
                continue   

            row = indexed.loc[(year, epiweek)]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            for col in feature_cols:
                grids[col][yi, ei, 0] = float(row[col])
            grids['data_mask'][yi, ei, 0] = 1.0   
    
    for out_col in data_outputs:
        prior_col = f"{out_col}.prior"
        original = grids[out_col] 
        prior_grid = np.roll(original, shift=1, axis=1)
        if prior_grid.shape[1] > 0:
            prior_grid[:, 0, :] = 0.0
        grids[prior_col] = prior_grid

    original_mask = grids['data_mask']
    prior_mask = np.roll(original_mask, shift=1, axis=1)
    if prior_mask.shape[1] > 0:
        prior_mask[:, 0, :] = 0.0
        
    # ---------------------------------------------------------
    # HOW TO CONTROL data_mask.prior:
    # If you want to forcefully ignore the mask prior (set all to 0),
    # simply uncomment the following line:
    # prior_mask = np.zeros_like(prior_mask)
    #
    # If you want to forcefully assume all priors are valid (set all to 1),
    # simply uncomment the following line:
    # prior_mask = np.ones_like(prior_mask)
    # ---------------------------------------------------------
        
    grids['data_mask.prior'] = prior_mask

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
        fold[key] = np.concatenate([lg[key] for lg in location_grids], axis=0)
    return fold


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def print_fold_summary(fold_idx, fold_dict, n_locations, n_years, output_col, verbose):
    n_examples = fold_dict[output_col].shape[0]  
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
    years      = [2022, 2023, 2024]
    epiweeks   = list(range(int(df['Epiweek'].min()), int(df['Epiweek'].max()) + 1))
    n_years    = len(years)
    n_epiweeks = len(epiweeks)

    print(f'    Time grid filtered: {n_years} years ({years[0]}..{years[-1]}) × '
          f'{n_epiweeks} epiweeks ({epiweeks[0]}..{epiweeks[-1]})')

    # -----------------------------------------------------------------------
    # 3. Derive feature column lists
    # -----------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c not in META_COLS]
    input_cols   = [c for c in feature_cols if c not in args.data_outputs]
    aux_keys = ['data_mask', 'data_mask.prior']

    all_pkl_keys = list(feature_cols)
    for out_col in args.data_outputs:
        all_pkl_keys.append(f"{out_col}.prior")
        input_cols.append(f"{out_col}.prior") 
    all_pkl_keys.extend(aux_keys)

    # -----------------------------------------------------------------------
    # 4. Filter and assign locations to explicit folds
    # -----------------------------------------------------------------------
    all_locations = sorted(df['Location.Name'].unique())
    # 1. Define the hardcoded Train set (all 6 stations)
    custom_train = ['R117-C385', 'R118-C380', 'R132-C372', 'R132-C381', 'R135-C383', 'R140-C378']
    
    print(f'\n[2] Applying Manual Location Filters:')
    print(f'    Training Stations (ALL) : {len(custom_train)}')
    print(f'    Validation Stations     : 0 (Disabled)')
    print(f'    Testing Stations        : 0 (Disabled)')

    # Combine them to define our new universe of locations
    active_locations = sorted(custom_train)
    print(f'    Total Active Stations: {len(active_locations)} (Dropped {len(all_locations) - len(active_locations)})')

    loc_to_fold = {}
    
    # We must distribute custom_train across folds 0, 1, 2 to avoid "empty fold" crashes in zero2neuro
    # (Because zero2neuro expects 5 folds to exist, and uses 0,1,2 for training)
    for i, loc in enumerate(custom_train):
        loc_to_fold[loc] = 0  # ALL stations go into fold 0

    # -----------------------------------------------------------------------
    # 5. Build per-location grids and save individual pkl files
    # -----------------------------------------------------------------------
    os.makedirs(args.dataset_directory, exist_ok=True)
    print(f'\n[3] Writing per-location pkl files -> {args.dataset_directory}/')

    fold_grids = {i: [] for i in range(args.data_n_folds)} 

    for loc in active_locations:
        loc_df = df[df['Location.Name'] == loc]
        grids  = build_location_grid(loc_df, feature_cols, years, epiweeks, args.data_outputs)

        loc_path = os.path.join(args.dataset_directory, f'{loc}.pkl')
        save_pkl(grids, loc_path)

        fold_idx = loc_to_fold[loc]
        if args.verbose >= 1:
            print(f'  {loc:<20}  fold={fold_idx}')

        fold_grids[fold_idx].append(grids)

    print(f'    {len(active_locations)} per-location pkl files written.')

    # -----------------------------------------------------------------------
    # 6. Assemble and save fold pkl files
    # -----------------------------------------------------------------------
    print(f'\n[4] Writing fold pkl files ({args.data_n_folds} folds) -> {args.dataset_directory}/')

    for fold_idx in range(args.data_n_folds):
        location_grids = fold_grids[fold_idx]
        n_locs_in_fold = len(location_grids)

        # Handle empty folds safely instead of asserting. If we manually assign stations, 
        # it is possible a fold (e.g. fold 1 or 2) ends up completely empty.
        if n_locs_in_fold == 0:
            print(f'  fold_{fold_idx}.pkl  ->  EMPTY FOLD (Manually bypassed)')
            empty_dict = {key: np.zeros((1, n_epiweeks, 1), dtype=np.float32) for key in all_pkl_keys}
            fold_path = os.path.join(args.dataset_directory, f'fold_{fold_idx}.pkl')
            save_pkl(empty_dict, fold_path)
            continue

        fold_dict = build_fold_dict(location_grids, all_pkl_keys)
        fold_path = os.path.join(args.dataset_directory, f'fold_{fold_idx}.pkl')
        save_pkl(fold_dict, fold_path)

        print_fold_summary(fold_idx, fold_dict, n_locs_in_fold, n_years, args.data_outputs[0], args.verbose)

    print('\nDone.\n')


if __name__ == '__main__':
    main()
