import sys
import os

# add src to path
sys.path.append('../../src')

from plugin_manager import PluginManager

pm = PluginManager(['plugins'])
print(f"plugin_paths: {pm.plugin_paths}")
try:
    pm.load_plugins(['netcdf_loader-data-loader', 'engine=netcdf4'], 3)
    print("Plugins loaded!")
except Exception as e:
    import traceback
    traceback.print_exc()

