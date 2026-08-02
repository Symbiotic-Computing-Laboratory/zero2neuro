import pandas as pd
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Feature Statistics Inspector")
    parser.add_argument('--data_file', type=str, 
                        default='Mosquito_Data_Merged_with_Cov_Data_clean.xlsx',
                        help='Path to the Excel dataset file')
    args = parser.parse_args()

    print(f"Loading dataset: {args.data_file}...")
    try:
        df = pd.read_excel(args.data_file)
    except FileNotFoundError:
        print(f"Error: Could not find file {args.data_file}")
        return

    # Columns to ignore from statistics (metadata)
    meta_cols = ['Unnamed: 0', 'Tag', 'Epiweek', 'Location.Name', 'Year']
    
    # Filter only the actual feature columns
    feature_cols = [col for col in df.columns if col not in meta_cols]

    print(f"\nFound {len(feature_cols)} features in the dataset.\n")
    print("-" * 110)
    print(f"{'Feature Name':<32} | {'Total':<7} | {'Zero=0':<7} | {'Non-Zero':<8} | {'Min':<8} | {'Max':<8} | {'Mean ± Std'}")
    print("-" * 110)

    for col in feature_cols:
        # Extract the column and drop NaNs for accurate statistics
        data = df[col].dropna()
        
        total = len(data)
        if total == 0:
            print(f"{col:<32} | {'0':<7} | {'0':<7} | {'0':<8} | {'N/A':<8} | {'N/A':<8} | N/A")
            continue

        zeros = (data == 0).sum()
        non_zeros = total - zeros
        
        c_min = data.min()
        c_max = data.max()
        c_mean = data.mean()
        c_std = data.std()

        # Format the output beautifully
        print(f"{col:<32} | {total:<7} | {zeros:<7} | {non_zeros:<8} | {c_min:<8.2f} | {c_max:<8.2f} | {c_mean:.2f} ± {c_std:.2f}")

    print("-" * 110)

if __name__ == '__main__':
    main()
