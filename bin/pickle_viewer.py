'''
Summarize the contents of a Zero2Neuro pickle-formatted file.

Andrew H. Fagg 2025-06-27 Original


Usage:
python pickle_viewer.py --file FNAME

'''

import argparse
import pickle

parser = argparse.ArgumentParser('Pickle Viewer')
parser.add_argument('--file', type=str, default=None, help='File to summarize')

args = parser.parse_args()

# File name specified?
if args.file is None:
    print('Must specify a file name')

else:
    # Read the dict from the file
    with open(args.file, 'rb') as fp:
        d = pickle.load(fp)

        # Loop over all key, value pairs and report
        for k,v in d.items():
            print(k, v.shape)

