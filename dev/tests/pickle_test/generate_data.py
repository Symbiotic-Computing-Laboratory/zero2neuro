'''
Generate data for testing a multiple pickle files as input.

We are setting up a 4-bit parity function

Andrew H. Fagg 2025-06-27 Original


'''

import numpy as np
from itertools import product
import pickle

# Generate all 4-bit binary combinations
combinations = list(product([0, 1], repeat=4))

# Convert to  numpy array
ins = np.array(combinations, dtype=np.uint8)

# Compute parity
outs = np.sum(ins, axis=1) % 2

# Split into some smaller datasets
ins0 = ins[:8,:]
outs0 = outs[:8]

ins1 = ins[8:12,:]
outs1 = outs[8:12]

ins2 = ins[12:15,:]
outs2 = outs[12:15]

ins3 = ins[15:,:]
outs3 = outs[15:]

###
# Write out

# Full data set
with open('parity.pkl', 'wb') as fp:
    d = {'ins': ins,
         'outs': outs,
         }
    pickle.dump(d, fp)

# First 8
with open('parity0.pkl', 'wb') as fp:
    d = {'ins': ins0,
         'outs': outs0,
         }
    pickle.dump(d, fp)

# Next 4
with open('parity1.pkl', 'wb') as fp:
    d = {'ins': ins1,
         'outs': outs1,
         }
    pickle.dump(d, fp)

# Next 3
with open('parity2.pkl', 'wb') as fp:
    d = {'ins': ins2,
         'outs': outs2,
         }
    pickle.dump(d, fp)

# Next 1
with open('parity3.pkl', 'wb') as fp:
    d = {'ins': ins3,
         'outs': outs3,
         }
    pickle.dump(d, fp)

#####
# Split ins into two pieces
# Full data set
with open('parityB.pkl', 'wb') as fp:
    d = {'ins0': ins[:,:3],
         'ins1': ins[:,3:],         
         'outs': outs,
         }
    pickle.dump(d, fp)

# First 8
with open('parityB0.pkl', 'wb') as fp:
    d = {'ins0': ins0[:,:3],
         'ins1': ins0[:,3:],         
         'outs': outs0,
         }
    pickle.dump(d, fp)

# Next 4
with open('parityB1.pkl', 'wb') as fp:
    d = {'ins0': ins1[:,:3],
         'ins1': ins1[:,3:],         
         'outs': outs1,
         }
    pickle.dump(d, fp)

# Next 3
with open('parityB2.pkl', 'wb') as fp:
    d = {'ins0': ins2[:,:3],
         'ins1': ins2[:,3:],         
         'outs': outs2,
         }
    pickle.dump(d, fp)

# Next 1
with open('parityB3.pkl', 'wb') as fp:
    d = {'ins0': ins3[:,:3],
         'ins1': ins3[:,3:],         
         'outs': outs3,
         }
    pickle.dump(d, fp)
    
