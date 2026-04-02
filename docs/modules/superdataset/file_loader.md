---
title: File Loader
nav_order: 10
parent: Data Files to Data Sets
---
# File Loader

## Data Format 
Data can be loaded from a set of different files.  However, all files
in the set must be of the same [Data File Format](data_files.md).
The format is specified using the following argument:

```
--data_format=XXX
```

where XXX is one of tabular (csv or xlsx), tabular-indirect, (csv or
xlsx), pickle, tf-dataset


## Data File location

Location of the set of data files:
```
--data_set_directory=PATH
```
where PATH is a relative or absolute path to the set of data files
(default is the local directory).

## Data Files

File name(s) can be specified in one of two ways:

### Single File
```
--data_file=YYY
```
where YYY is the name of a file relative to the data_set_directory

### Multiple Files

```
--data_files
YYY0
YYY1
YYY2
 :
```
where YYY0, YYY1, ... is a list of file names.

### Multiple Files with Fold Assignment

```
--data_files
YYY0,F0
YYY1,F1
YYY2,F2
 :
```
where YYY0, YYY1, ... is a list of file names, and F0, F1, ... are the
integer fold numbers (0, 1, ...) that the contents of each file may
be assigned to (see [Data Set Generator](data_folds_to_sets.md) for
more detail).

## Selecting Fields from Files

For tabular, tabular-indirect, and pickle formatted files, one must
specify which of the fields within the file to use for different
purposes.  For tabular files, the fields are different columns.

### Data Inputs

The list of fields from the file to include as inputs to the model
```
--data_inputs
in0
in1
 :
```

Notes:
- tabular, tabular-indirect: specifies the columns in the table to
include
- pickle: specifies the keys from the dictionary to include

### Data Outputs
The list of fields from the file to include as the desired outputs
values from the model

```
--data_outputs
out0
out1
  :
```

Notes:
- tabular, tabular-indirect: columns to include
- pickle: keys from the dictionary to include

### Data Weights (Advanced; optional)
Each example in a dataset can be weighted for the purposes of
computing the loss function.  In unbalanced data 
sets, the common cases can dominate the loss, resulting in a model
that essentially ignores the rare cases.  When using weights, the rare
cases  should be assigned a weight of 1, and the common cases should be
assigned a low weight (e.g., the ratio of rare to common).  All
weights should fall within the range 0...1.
```
--data_weights=FIELD
```
where FIELD is the string name of the field that contains the weights.


### Data Groups (Advanced; optional)
The field to be used to possibly determine the fold that each example
belongs to.

```
--data_weights=FIELD
```
The values in FIELD are natural numbers (integers: 0, 1, ....).  See
[Data Set Generator](data_folds_to_sets.md) for more detail. 

___

## Additional Tabular Arguments (Advanced)
These arguments can be used for both csv and xlsx formatted files

### Header Row
By default, the header row is assumed to be the first row.  This
argument can be used to set a different row

```
--tabular_header_row=ROW
```
where ROW is the row index (counting from 1)

### Header Names
Some tabular files do not contain a header row.  Use this argument to
force the names of the columns
```
--tabular_header_names
COL1
COL2
 :
```
where COLX is the string name for column X.

### Tabular Columns
In some cases, one must ignore a subset of the tabular columns.  The
next two arguments can be used to specify the set of columns to
use for the full set of fields.

```
--tabular_column_range
START
END
```
where START and END are integers specifying the column indices to
include (columns START ... END, including END).

TODO: check inclusive of END

```
--tabular_column_list
COL1
COL2
 :
```
where COLX is an integer column index (counting from 1)

### Text Encoding

The name of the byte encoding for the tabular files.  The default is
ASCII (utf-8).

```
--tabular_encoding=ENCODING
```

TODO: define ENCODING options
