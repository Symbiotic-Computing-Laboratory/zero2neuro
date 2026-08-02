'''
prepare_data_prior_v3.py
---------------
Mosquito dataset preparation for the Zero2Neuro RNN pipeline.

Reads the raw Excel source and writes per-location and per-fold pickle files.
Includes the generation of an autoregressive prior feature (target shifted by t+1)
that properly respects the Year boundaries to prevent winter gaps from bleeding
into the Spring predictions.

 Pickle key layout (per key, per location file):
   shape  (N_YEARS, N_EPIWEEKS, 1)  — one scalar value per (year, epiweek) cell.

 v3 change vs v2
 ---------------
 The epiweek grid size (n_epiweeks) is now an explicit CLI parameter
 (--n_epiweeks, default 30) rather than being inferred as
 max_epiweek - min_epiweek + 1 from the data.

 The grid is always anchored at the minimum epiweek present in the data
 (or overridden by --min_epiweek) and spans exactly --n_epiweeks consecutive
 slots.  Any observation whose epiweek label falls outside that window is
 silently ignored.

 Example
 -------
 Data has epiweeks 19-48.  With --n_epiweeks 30 (default) the grid covers
 slots [19, 20, ..., 48] — identical to v2.
 With --n_epiweeks 20 the grid covers only [19, 20, ..., 38]; weeks 39-48
 are dropped.
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

def build_location_grid(loc_df, feature_cols, years, epiweeks, data_outputs):
    '''
    Build the (n_years × n_epiweeks × 1) numpy grid for every feature at
    a single trap location.

    Parameters
    ----------
    loc_df       : DataFrame filtered to this location
    feature_cols : list of column names to include as features
    years        : list of calendar years (grid rows)
    epiweeks     : list of epiweek labels (grid columns) — length == n_epiweeks
    data_outputs : list of output column names (used to generate .prior keys)
    '''
    n_years    = len(years)
    n_epiweeks = len(epiweeks)

    # Allocate zero-filled grids — zero is the padding value for absent cells.
    grids = {
        col: np.zeros((n_years, n_epiweeks, 1), dtype=np.float32)
        for col in feature_cols
    }
    # Binary mask: 1.0 where a real observation exists, 0.0 where zero-padded.
    grids['data_mask'] = np.zeros((n_years, n_epiweeks, 1), dtype=np.float32)

    # Build a (Year, Epiweek) multi-index for O(1) cell lookup.
    # Only keep rows whose epiweek falls inside the grid window.
    valid_ew = set(epiweeks)
    filtered = loc_df[loc_df['Epiweek'].isin(valid_ew)]
    indexed  = filtered.set_index(['Year', 'Epiweek'])

    # Build a fast epiweek-label → grid-column-index map
    ew_to_idx = {ew: ei for ei, ew in enumerate(epiweeks)}

    for yi, year in enumerate(years):
        for epiweek in epiweeks:
            ei = ew_to_idx[epiweek]

            # Skip cells for which no observation exists at this location
            if (year, epiweek) not in indexed.index:
                continue   # leave feature values at 0.0 (zero-padding)

            row = indexed.loc[(year, epiweek)]

            # When a location has multiple records for the same (year, epiweek)
            # (data quality issue) take the first row to stay deterministic
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            # Write observed feature values into the grid
            for col in feature_cols:
                grids[col][yi, ei, 0] = float(row[col])
            grids['data_mask'][yi, ei, 0] = 1.0   # mark cell as real

    # Generate an autoregressive prior (t-1) for each output column
    # by rolling ONLY along the Epiweek axis (axis=1) and zeroing out the first
    # slot of every year to prevent the winter-gap discontinuity.
    for out_col in data_outputs:
        prior_col = f"{out_col}.prior"
        original  = grids[out_col]  # shape: (n_years, n_epiweeks, 1)

        # Roll along the epiweek axis (axis=1): week[i] gets value from week[i-1]
        prior_grid = np.roll(original, shift=1, axis=1)

        # Zero out slot 0 of every year so that the last week of the previous
        # year does not wrap around as the prior for the first week.
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
    parser = _z2n_create_parser(description='Mosquito dataset preparation (v3)')

    # -----------------------------------------------------------------------
    # Extra arguments specific to this preparation script
    # -----------------------------------------------------------------------
    parser.add_argument(
        '--n_epiweeks',
        type=int,
        default=30,
        metavar='N',
        help=(
            'Fixed number of epiweek slots in the output grid (default: 30). '
            'The grid is anchored at the minimum epiweek found in the data '
            '(or --min_epiweek if supplied) and spans exactly N consecutive '
            'epiweek labels.  Any observation outside this window is ignored.'
        )
    )
    parser.add_argument(
        '--min_epiweek',
        type=int,
        default=None,
        metavar='EW',
        help=(
            'Override the starting epiweek of the grid (default: auto-detected '
            'as the minimum epiweek present in the data). '
            'E.g. --min_epiweek 19 forces the grid to start at week 19.'
        )
    )
    parser.add_argument(
        '--drop_sparse_years',
        type=float,
        default=None,
        metavar='THRESHOLD',
        help=(
            'Drop any calendar year that covers fewer than THRESHOLD fraction '
            'of the --n_epiweeks grid slots (based on unique epiweeks present '
            'in that year, ignoring location count). '
            'Value must be in (0, 1]. E.g. 0.6 drops years that have data for '
            'fewer than 60%% of the expected epiweeks.'
        )
    )

    args = parser.parse_args()

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
    # 2. Build the fixed epiweek grid
    # -----------------------------------------------------------------------
    min_ew_data = int(df['Epiweek'].min())
    max_ew_data = int(df['Epiweek'].max())

    # Anchor: use --min_epiweek if supplied, otherwise auto-detect
    min_ew = args.min_epiweek if args.min_epiweek is not None else min_ew_data
    n_epiweeks = args.n_epiweeks
    max_ew = min_ew + n_epiweeks - 1   # last epiweek label that fits in the grid

    epiweeks = list(range(min_ew, max_ew + 1))  # length == n_epiweeks exactly

    print(f'\n    Epiweek grid  : {n_epiweeks} slots  '
          f'(labels {min_ew} – {max_ew})')
    print(f'    Data range    : epiweeks {min_ew_data} – {max_ew_data}')

    # Warn if any data falls outside the grid window
    out_of_window = df[(df['Epiweek'] < min_ew) | (df['Epiweek'] > max_ew)]
    if len(out_of_window) > 0:
        extra_ews = sorted(out_of_window['Epiweek'].unique().tolist())
        print(f'    WARNING: {len(out_of_window):,} rows have epiweek labels '
              f'outside [{min_ew}, {max_ew}] and will be ignored: {extra_ews}')

    # -----------------------------------------------------------------------
    # 2b. Year list and optional sparse-year filter
    # -----------------------------------------------------------------------
    all_years = list(range(int(df['Year'].min()), int(df['Year'].max()) + 1))

    print(f'\n    Years in data : {len(all_years)}  '
          f'({all_years[0]} – {all_years[-1]})')

    if args.drop_sparse_years is not None:
        threshold = args.drop_sparse_years
        assert 0 < threshold <= 1.0, (
            f'--drop_sparse_years must be in (0, 1], got {threshold}'
        )

        dropped_years = []
        kept_years    = []

        print(f'\n[1b] Filtering sparse years  '
              f'(threshold: epiweek coverage < {threshold*100:.0f}%  '
              f'of {n_epiweeks} slots)')
        print(f'     {"Year":>6}  {"Unique EWs":>10}  {"Coverage%":>10}  {"Action":>8}')
        print(f'     {"-"*6}  {"-"*10}  {"-"*10}  {"-"*8}')

        for yr in all_years:
            yr_ews    = df[df['Year'] == yr]['Epiweek']
            # Count only epiweeks that fall inside the grid window
            yr_ews_in = yr_ews[(yr_ews >= min_ew) & (yr_ews <= max_ew)]
            unique_ew = yr_ews_in.nunique()
            coverage  = unique_ew / n_epiweeks

            if coverage < threshold:
                action = 'DROP'
                dropped_years.append(yr)
            else:
                action = 'KEEP'
                kept_years.append(yr)

            print(f'     {yr:>6}  {unique_ew:>10}  {coverage*100:>9.1f}%  {action:>8}')

        print()
        if dropped_years:
            print(f'  Removed {len(dropped_years)} year(s): {dropped_years}')
        else:
            print('  No years removed (all years meet the coverage threshold).')
        print(f'  Remaining {len(kept_years)} year(s) for training: {kept_years}')

        if not kept_years:
            raise ValueError(
                'All years were dropped! '
                'Lower --drop_sparse_years threshold or check your data.'
            )

        df    = df[df['Year'].isin(kept_years)].copy()
        years = kept_years
    else:
        years = all_years

    n_years = len(years)
    print(f'\n    Final grid: {n_years} year(s) ({years[0]}–{years[-1]}) '
          f'× {n_epiweeks} epiweeks ({epiweeks[0]}–{epiweeks[-1]})')

    # -----------------------------------------------------------------------
    # 3. Derive feature column lists
    # -----------------------------------------------------------------------
    feature_cols = [c for c in df.columns if c not in META_COLS]
    input_cols   = [c for c in feature_cols if c not in args.data_outputs]
    aux_keys     = ['data_mask', 'data_mask.prior']

    # Include the .prior generated features into the keys to be pickled
    all_pkl_keys = list(feature_cols)
    for out_col in args.data_outputs:
        all_pkl_keys.append(f"{out_col}.prior")
        input_cols.append(f"{out_col}.prior")  # Update for reporting
    all_pkl_keys.extend(aux_keys)

    print(f'    Input features : {len(input_cols)}  '
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
        grids  = build_location_grid(loc_df, feature_cols, years, epiweeks, args.data_outputs)

        loc_path = os.path.join(args.dataset_directory, f'{loc}.pkl')
        save_pkl(grids, loc_path)

        fold_idx = loc_to_fold[loc]
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
