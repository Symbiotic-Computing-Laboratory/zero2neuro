# Normal NetCDF Loader vs. Preserve Dimensions (`preserve_dims`)

When the `netcdf_loader` plugin reads data from a multi-dimensional `.nc` file (like `climate_4d.nc`), the shape of the data often contains complex spatial and temporal dimensions. 

Here is how the plugin handles this data with and without the `preserve_dims` flag:

## The Normal Way (No Flag)
By default, `netcdf_loader` flattens all dimensions except for the primary `sample_dim`.

- **Input Shape in NetCDF:** `[samples=10, time=24, lat=5, lon=5, features=2]`
- **What Keras Receives:** `[10, 1200]` *(where $24 \times 5 \times 5 \times 2 = 1200$)*
- **Why it does this:** Standard Multi-Layer Perceptrons (Fully Connected networks) can only process 1-dimensional vectors. Flattening ensures the data can instantly be passed into Dense layers without crashing.
- **The downside:** You destroy all spatial and temporal relationships. The network has no idea that hour 1 is next to hour 2, or that lat 3 is next to lat 4.

## The `preserve_dims` Way
When you add the `preserve_dims` flag to the plugin arguments (e.g., `engine=netcdf4,sample_dim=sample,preserve_dims`), the loader skips the flattening step entirely.

- **Input Shape in NetCDF:** `[samples=10, time=24, lat=5, lon=5, features=2]`
- **What Keras Receives:** `[10, 24, 5, 5, 2]`
- **Why it does this:** Modern Deep Learning architectures like Convolutional Neural Networks (CNNs), U-Nets, and LSTMs *require* data to maintain its spatial and temporal structure to extract patterns. 
- **The benefit:** By preserving the native `(time, lat, lon)` shape, you can pass this data directly into spatial layers (like `Conv2D` or `Conv3D`), allowing the network to understand the meteorological topology of the dataset.

## The Math Behind Flattened Shapes (1200 and 600)

When you do not use `preserve_dims` and you configure a native `--network_type=fully_connected` architecture, you must manually specify the 1-dimensional sizes of the inputs and outputs in `network_config.txt`. Here is exactly where those numbers come from:

### Input Shape: 1200
Your raw NetCDF `climate_4d.nc` file contains two variables (Temperature and Humidity) mapped across a grid:
- **Time Steps:** 24
- **Latitude:** 5
- **Longitude:** 5
- **Variables:** 2 

When flattened, the total number of input variables sent to the Fully Connected network is:
`24 × 5 × 5 × 2 = 1,200` inputs per sample. 

### Output Shape: 600
Your target output (`mosquito_count`) is mapped across the same grid, but only has one variable:
- **Time Steps:** 24
- **Latitude:** 5
- **Longitude:** 5
- **Variables:** 1 

When flattened, the total number of output values the network needs to predict is:
`24 × 5 × 5 × 1 = 600` outputs per sample.

This is why your `network_config.txt` must explicitly use `--input_shape 1200` and `--output_shape 600` when the data is flattened!
