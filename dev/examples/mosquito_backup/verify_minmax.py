import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def main():
    print("Loading raw pickle file from fold 0...")
    with open('dataset/fold_0.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Let's inspect a specific feature
    feature_name = 'Culex.restuans'
    feature_array = data[feature_name]  # Shape: (70, 30, 1)
    
    print(f"\n--- BEFORE NORMALIZATION ---")
    print(f"Raw '{feature_name}' Max: {np.max(feature_array)}")
    print(f"Raw '{feature_name}' Min: {np.min(feature_array)}")
    
    print("\nApplying MinMaxScaler (exactly what the Zero2Neuro plugin does)...")
    scaler = MinMaxScaler()
    
    # MinMaxScaler expects 2D data, so we flatten it temporarily
    original_shape = feature_array.shape
    array_2d = feature_array.reshape(-1, original_shape[-1])
    
    norm_2d = scaler.fit_transform(array_2d)
    norm_array = norm_2d.reshape(original_shape)
    
    print(f"\n--- AFTER NORMALIZATION ---")
    print(f"Normalized '{feature_name}' Max: {np.max(norm_array)}")
    print(f"Normalized '{feature_name}' Min: {np.min(norm_array)}")

if __name__ == '__main__':
    main()
