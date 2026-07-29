'''
Generate a small synthetic NetCDF file for testing the netcdf_loader plugin.

Problem: 4-bit parity — same as the pickle_test, but stored as a NetCDF file.
  - 16 examples (all 4-bit binary combinations)
  - 4 input features (b0, b1, b2, b3)
  - 1 output (parity = sum(bits) % 2)

Run this script once from the netcdf_test directory:
    python generate_data.py
'''

import numpy as np
import xarray as xr
from itertools import product

# --- Generate all 16 4-bit binary combinations ---
combinations = list(product([0, 1], repeat=4))
ins  = np.array(combinations, dtype=np.float32)   # shape (16, 4)
outs = (np.sum(ins, axis=1) % 2).astype(np.float32)  # shape (16,)

# --- Pack into an xarray Dataset (one variable per feature/target) ---
ds = xr.Dataset({
    'b0':     xr.DataArray(ins[:, 0], dims='sample'),
    'b1':     xr.DataArray(ins[:, 1], dims='sample'),
    'b2':     xr.DataArray(ins[:, 2], dims='sample'),
    'b3':     xr.DataArray(ins[:, 3], dims='sample'),
    'parity': xr.DataArray(outs,      dims='sample'),
})

ds.to_netcdf('parity.nc')
print(f'Written parity.nc  ({len(ds.sample)} examples, variables: {list(ds.data_vars)})')
