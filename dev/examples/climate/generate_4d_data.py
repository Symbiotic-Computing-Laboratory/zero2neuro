import numpy as np
import xarray as xr

# Let's create 10 samples (e.g., 10 different weather stations/locations)
# Each sample has 24 hours (time), a 5x5 grid (lat/lon)
samples = 10
hours = 24
lats = 5
lons = 5

# Generate random synthetic data for our 4D arrays
temperature = np.random.normal(loc=25, scale=5, size=(samples, hours, lats, lons)).astype(np.float32)
humidity = np.random.uniform(low=40, high=90, size=(samples, hours, lats, lons)).astype(np.float32)

# Our output variable (e.g. mosquito population) is a function of temp + humidity
mosquito_count = ((temperature * 2) + (humidity * 0.5) + np.random.normal(0, 1, size=(samples, hours, lats, lons))).astype(np.float32)

# Pack into a 4D xarray Dataset
ds = xr.Dataset(
    {
        'temperature': (['sample', 'time', 'lat', 'lon'], temperature),
        'humidity': (['sample', 'time', 'lat', 'lon'], humidity),
        'mosquito_count': (['sample', 'time', 'lat', 'lon'], mosquito_count),
    },
    coords={
        'sample': np.arange(samples),
        'time': np.arange(hours),
        'lat': np.linspace(-90, 90, lats),
        'lon': np.linspace(-180, 180, lons),
    }
)

ds.to_netcdf('climate_4d.nc')
print("Generated climate_4d.nc!")
print(ds)
