'''
NetCDF Data Loader Plugin for Zero2Neuro
========================================

This plugin seamlessly reads NetCDF (.nc, .nc4, .h5) files and returns structured numpy arrays 
(inputs, outputs, weights, groups, and tags) ready for the Zero2Neuro training pipeline.

### Core Features:
- **Auto-Detection**: Intelligently selects the best xarray engine based on the file extension.
- **Multidimensional Flattening**: Automatically flattens multi-dimensional variables (e.g., time x lat x lon) 
  so they always return as standard (Examples, Features) matrices.
- **Categorical Support**: Works with built-in tabular categorical conversions out of the box.

### Configuration Example:
```text
--data_format plugin
--data_representation numpy
--plugin_list
netcdf_loader-dataloader
engine=netcdf4

--data_files /path/to/data.nc
--data_inputs feature1 feature2
--data_outputs label
```
'''

import os
import numpy as np
import xarray as xr

from plugin_base import GenericPlugin, require
from zero2neuro_debug import handle_error, print_debug

# Maps file extensions to their optimal xarray backend engine
_ENGINE_BY_EXT = {
    '.nc': 'netcdf4', '.nc4': 'netcdf4',
    '.h5': 'h5netcdf', '.hdf5': 'h5netcdf',
    '.gz': 'scipy'
}

def _infer_engine(path: str) -> str:
    '''Guesses the best xarray backend from the file extension (defaults to netcdf4).'''
    _, ext = os.path.splitext(path.lower())
    return _ENGINE_BY_EXT.get(ext, 'netcdf4')


class netcdf_loader(GenericPlugin):
    '''
    Loads a single NetCDF file and extracts features, targets, and metadata based on CLI flags.
    '''

    def __init__(self):
        super().__init__()
        self.role = 'dataloader'
        self.parser.description = 'Load NetCDF files into structured numpy arrays.'

        self.parser.add_argument(
            '--engine', type=str, default=None,
            choices=['netcdf4', 'scipy', 'h5netcdf', 'store', 'auto'],
            help='xarray backend engine. Leave unset to auto-detect from extension.'
        )
        self.parser.add_argument(
            '--group', type=str, default=None,
            help='NetCDF group path inside the file (default: root group).'
        )
        self.parser.add_argument(
            '--decode_times', action='store_true',
            help='Decode CF-convention time coordinates using xarray.'
        )
        self.parser.add_argument(
            '--sample_dim', type=str, default=None,
            help='Dimension name representing individual examples (e.g., "time").'
        )
        self.parser.add_argument(
            '--preserve_dims', action='store_true',
            help='Do not flatten spatial dimensions (useful for CNN/Conv3D models).'
        )

    @staticmethod
    def _full_path(dataset_directory: str | None, file_path: str) -> str:
        '''Resolves file path against the dataset directory if it is relative.'''
        if os.path.isabs(file_path):
            return file_path
        if dataset_directory:
            return os.path.join(dataset_directory, file_path)
        return file_path

    @staticmethod
    def _extract_var(ds: 'xr.Dataset', name: str, path: str, sample_dim: str | None = None, preserve_dims: bool = False) -> np.ndarray:
        '''
        Pulls a variable from the dataset and reshapes it to (N_examples, N_features).
        If `preserve_dims` is True, keeps spatial dimensions intact and appends a channel axis.
        If `sample_dim` is provided, that dimension becomes the examples axis and the rest are flattened.
        '''
        if name not in ds:
            raise ValueError(f"Variable '{name}' not found in '{path}'. Available: {list(ds.data_vars)}")

        var = ds[name]

        if var.ndim == 0:
            return var.values.reshape(1, 1)
        if var.ndim == 1:
            return var.values[:, np.newaxis]

        if sample_dim is not None:
            if sample_dim not in var.dims:
                raise ValueError(f"sample_dim '{sample_dim}' not found in variable '{name}' in '{path}'.")
                
            other_dims = [d for d in var.dims if d != sample_dim]
            arr = var.transpose(sample_dim, *other_dims).values
            
            if preserve_dims:
                return np.expand_dims(arr, axis=-1)
            else:
                return arr.reshape(arr.shape[0], -1)

        # Default fallback: First dim is examples, last dim is features
        arr = var.values
        if preserve_dims:
            return arr
        return arr if arr.ndim == 2 else arr.reshape(-1, arr.shape[-1])

    @staticmethod
    def _stack_vars(ds: 'xr.Dataset', names: list[str], path: str, label: str, sample_dim: str | None = None, preserve_dims: bool = False) -> np.ndarray | None:
        '''Extracts and horizontally stacks multiple variables.'''
        if not names:
            return None

        arrays = [netcdf_loader._extract_var(ds, n, path, sample_dim, preserve_dims) for n in names]
        n_examples = arrays[0].shape[0]

        for arr, name in zip(arrays[1:], names[1:]):
            if arr.shape[0] != n_examples:
                raise ValueError(
                    f"Shape mismatch in {label}: '{name}' has {arr.shape[0]} examples, "
                    f"but '{names[0]}' has {n_examples}."
                )

        return np.concatenate(arrays, axis=-1)

    @staticmethod
    def _apply_categorical(ds: 'xr.Dataset', categorical_translation: list | None, path: str, verbose: int) -> 'xr.Dataset':
        '''Applies dictionary-based string-to-integer mappings for categorical variables.'''
        if not categorical_translation:
            return ds

        for var, tr_dict in categorical_translation:
            if var not in ds:
                continue

            arr = ds[var].values
            mapped = np.vectorize(lambda x: tr_dict.get(str(x), -999))(arr)

            if np.any(mapped == -999):
                failed = np.unique(arr[mapped == -999]).tolist()
                handle_error(f"Categorical mapping failed for '{var}' in '{path}'. Unmapped values: {failed}", verbose)

            ds[var] = xr.DataArray(mapped, dims=ds[var].dims, coords=ds[var].coords, attrs=ds[var].attrs)

        return ds

    def call(self, **kwargs) -> dict:
        '''
        Main execution step. Opens the NetCDF file(s), maps data, and packages it for the engine.
        '''
        args = require(kwargs, 'args')
        
        dataset_directory = kwargs.get('dataset_directory')
        debug_level = kwargs.get('debug_level', 0)
        categorical_translation = kwargs.get('categorical_translation')

        data = []

        for file_path in getattr(args, 'data_files', []):
            path = self._full_path(dataset_directory, file_path)
            if not os.path.exists(path):
                handle_error(f"netcdf_loader: file not found: '{path}'", args.verbose)

            # Detect and set engine
            engine = self.args.engine
            if engine in (None, 'auto'):
                engine = _infer_engine(path)

            print_debug(f"netcdf_loader: opening '{path}' (engine={engine})", 1, debug_level)

            # Open the file
            open_kw = {'engine': engine, 'decode_times': self.args.decode_times}
            if self.args.group: 
                open_kw['group'] = self.args.group
                
            ds = xr.open_dataset(path, **open_kw)

            # Process data
            ds = self._apply_categorical(ds, categorical_translation, path, args.verbose)
            dim = self.args.sample_dim
            p_dims = self.args.preserve_dims

            ins = self._stack_vars(ds, getattr(args, 'data_inputs', []) or [], path, 'inputs', dim, p_dims)
            outs = self._stack_vars(ds, getattr(args, 'data_outputs', []) or [], path, 'outputs', dim, p_dims)

            weights_var = getattr(args, 'data_weights', None)
            weights = self._extract_var(ds, weights_var, path, dim, p_dims).squeeze() if weights_var else None

            groups_var = getattr(args, 'data_groups', None)
            groups = self._extract_var(ds, groups_var, path, dim, p_dims).squeeze().astype(int) if groups_var else None

            tag_names = getattr(args, 'data_tag_examples', None) or []
            tags = {name: self._extract_var(ds, name, path, dim, p_dims).squeeze() for name in tag_names} if tag_names else None

            stratify_var = getattr(args, 'data_stratify', None)
            stratify = self._extract_var(ds, stratify_var, path, dim, p_dims).squeeze() if stratify_var else None

            ds.close()

            print_debug(
                f"netcdf_loader: ins={getattr(ins,'shape',None)} outs={getattr(outs,'shape',None)} "
                f"weights={getattr(weights,'shape',None)} tags={list(tags.keys()) if tags else None} "
                f"groups={getattr(groups,'shape',None)} stratify={getattr(stratify,'shape',None)}",
                1, debug_level
            )
            
            if ins is None:
                handle_error(f"netcdf_loader returned no 'ins' array for file '{file_path}'.", args.verbose)

            data.append((ins, outs, weights, tags, groups, stratify))

        return {
            'data': data
        }
