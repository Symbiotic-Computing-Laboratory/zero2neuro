# Performing Many Experiments

There are many situations where we want to perform not a single ML
experiment, but a series of experments.  For example:

- All N rotations of a N-Fold cross-validation training/evaluation run
- All N rotations of a N-Fold cross-validation training/evaluation run
__x__ a range of different dropout parameter values
   - Here 'x' refers to the _Cartesian product_ of the set of N
rotations and the set of K dropout parameter values.  In other words,
we want to express all possible combinations of the two

The process for using this approach with Zero2Neuro is:
1. Define the arguments and their respective sets.  This implicitly
defines all possible combinations of argument values.  These possible
combinations have an explicit order (indexed by 0, 1, ...)

2. Declare which experimental index is currently being
trained/evaluated

## Example 1: Rotations

1. Typically defined in a separate configuration file (e.g.,
_cartesian.txt_):

```text
	--cartesian_arguments
	data_rotation:0,1,2,3,4
```

equivalently:
```text
	--cartesian_arguments
	data_rotation:range(5)
```

2. Declare which combination to execute.  This is typically given at
the command line:
```text
	--cartesian_selection_index 2
```

where the valid index values are 0,1,2,3,4 and map trivially to
data_rotation 0,1,2,3,4, respectively.

## Example 2: Rotations x Dropout

1. _cartesian.txt_:

```text
	--cartesian_arguments
	data_rotation:0,1,2,3,4
	dropout:None,.1,.2
```

2. cartesian_selection_index can then be in the range of 0..14,
specifying the argument combinations in the following order:

```
	0:	{'data_rotation': 0, 'dropout': None}
	1:	{'data_rotation': 0, 'dropout': 0.1}
	2:	{'data_rotation': 0, 'dropout': 0.2}
	3:	{'data_rotation': 1, 'dropout': None}
	4:	{'data_rotation': 1, 'dropout': 0.1}
	5:	{'data_rotation': 1, 'dropout': 0.2}
	6:	{'data_rotation': 2, 'dropout': None}
	7:	{'data_rotation': 2, 'dropout': 0.1}
	8:	{'data_rotation': 2, 'dropout': 0.2}
	9:	{'data_rotation': 3, 'dropout': None}
	10:	{'data_rotation': 3, 'dropout': 0.1}
	11:	{'data_rotation': 3, 'dropout': 0.2}
	12:	{'data_rotation': 4, 'dropout': None}
	13:	{'data_rotation': 4, 'dropout': 0.1}
	14:	{'data_rotation': 4, 'dropout': 0.2}
```

___
## Executing multiple training/evaluation experiments

There are multiple ways to perform the full set of experiments once
one has defined the Cartesian product of argument choices.

1. By hand at the command line or in a Jupyter cell

	Explicitly cycle through the different possible values of
_cartesian_selection_index_, starting Zero2Neuro for each possible value


2. Batch file in SLURM (on the OU Supercomputer)

	Add the following to your batch file:
```
	#SBATCH --array 0-k
```
   - where k = the maximum number of combinations minus 1.  When you schedule your experiment, this will tell SLURM to perform a
total of k+1 experiments, numbered 0, 1, 2, ... k

      Then, add to the command line that executes your experiment:
```
	--cartesian_selection_index $SLURM_ARRAY_TASK_ID
```

So, your command line will look something like this:

```
	python $ZERO2NEURO_PATH/zero2neuro.py @data.txt @experiment.txt @network.txt -vvv --cartesian_selection_index $SLURM_ARRAY_TASK_ID
```

3. Coming features:
   - Execute all k+1 experiments at the command line or
in a Jupyter cell.

___
## Details

Individual lines in --cartesian_arguments can take on multiple forms

### List

```
	ARG:VAL0, VAL1, VAL2, ... VALj-1
```

- ARG must be a command line argument
- VALs can be lists of ints, floats, or strings.  These types cannot
be mixed.  However, one may use the _None_
keyword in any of these lists (a typical default value that usually
means that nothing has been specified)

### Functions

1. START, END, SKIP are ints (END is exclusive):
```
	ARG:range(END)                          -> 0,1,2,3, ..., END-1
	ARG:range(START,END)                    -> START, START+1, START+2, ..., END-1
	ARG:range(START,END,SKIP)               -> START, START+SKIP, START+2*SKIP, ..., END-1
```

2. START, END, SKIP are floats (END is exclusive):
```
	ARG:arange(START, END, SKIP)            -> START, START+SKIP, START+2*SKIP, ..., END-epsilon
```
    
3. START, END are floats, NUM, BASE are ints (END is inclusive):
```
	ARG:logspace(START, END, NUM[, BASE=10])   -> NUM items in the range START^BASE ... END^BASE arranged exponentially
	ARG:exp_range(START, END, NUM[, BASE=10])  -> NUM items in the range START ... BASE arranged exponentially
```
          - BASE is an optional parameter; the default value is base 10

### Example Functions

```
	range(20)                 -> [0, 1, 2, 3, ..., 19]
	range(0,20,2)             -> [0, 2, 4, 6, ..., 18]
	arange(0, .5, .1)         -> [0, .1, .2, .3, .4]
	arange(0, .51, .1)        -> [0, .1, .2, .3, .4, .5]
	logspace(-5, 0, 6)        -> [.00001, .0001, .001, .01, .1, 1.0]
	exp_range(1, 100000, 6)   -> [1, 10, 100, 1000, 10000, 100000]
	exp_range(.00001, 1.0, 6) -> [.00001, .0001, .001, .01, .1, 1.0]
```

### Specifying multiple experiments in your batch file

Range of experiments:
```
	#SBATCH --array 0-k
```

Comma separated list of experiments:
```
	#SBATCH --array 0,1,2,20,21,22
```

A mix:
```
	#SBATCH --array 0-2,20-22
```


___
## Constraints
- All argument (ARG) names must correspond to command-line arguments
- The type of the specified value must match the type of the command
line argument
- Output file names: all arguments that are being varied must occur in
the output file names.  For example:
```
	--output_file_base={args.experiment_name}_{args.network_type}_R{args.data_rotation:02d}_dropout{args.dropout}
```
___
