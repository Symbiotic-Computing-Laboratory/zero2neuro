"""
Deeper inspection of the results pkl dataset key and check for saved model.
"""
import pickle
import numpy as np
import os

# Check results dir for any model files
print("=== Files in results/ ===")
for f in os.listdir('results'):
    size = os.path.getsize(f'results/{f}')
    print(f'  {f}  ({size:,} bytes)')

print()

with open(r'results/mosquito_tensor_R00_results.pkl', 'rb') as f:
    res = pickle.load(f)

# Inspect 'dataset' key deeply
print("=== dataset key ===")
ds = res['dataset']
for k, v in ds.items():
    if hasattr(v, 'shape'):
        print(f'  [{k}]  shape={v.shape}  dtype={v.dtype}')
    elif isinstance(v, dict):
        print(f'  [{k}]  dict with keys: {list(v.keys())[:10]}')
    elif isinstance(v, (list, np.ndarray)) and len(v) > 0:
        print(f'  [{k}]  len={len(v)}')
    else:
        print(f'  [{k}]  = {repr(v)[:80]}')
