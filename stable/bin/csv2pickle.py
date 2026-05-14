'''
Perform a simple translation of a csv file to a pickle file

Andrew H. Fagg 2025-07-09 Original


Usage:
python csv2pickle.py --infile FNAME --outfile OFNAME

'''

import argparse
import pickle
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser('CSV to Pickle Converter')
parser.add_argument('--infile', '-i', type=str, default=None, help='Input file')
parser.add_argument('--outfile', '-o', type=str, default=None, help='Output file')

args = parser.parse_args()

df = pd.read_csv(args.infile)

# Convert each column to a Nx1 numpy array; store in a dict
d = {col: np.expand_dims(df[col].to_numpy(), axis=-1) for col in df.columns}

# Write out
with open(args.outfile, "wb") as fp:
    pickle.dump(d, fp)

