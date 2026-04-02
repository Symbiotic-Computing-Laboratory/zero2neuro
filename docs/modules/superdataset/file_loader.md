---
title: File Loader
nav_order: 10
parent: Data Files to Data Sets
---
# File Loader

## Key arguments

### Data Format
The data format specifies the (type of the data
files)[./data_files.md] to load.  

```
--data_format=XXX
```

- Can be one of tabular (csv or xlsx), tabular-indirect, (csv or
xlsx), pickle, tf-dataset


### Data File List
There must be at least one data file; there are two possible ways to
specify the list:

```
--data_files
file0.format
file1.format
  :

```
where fileX.format is the name of the file to load

or

```
--data_files
file0.format,F0
file1.format,F1
  :

```
where FX is a natural number (0, 1, ...) that potentially specifies
the fold to assign the file to.

### Data Inputs
The list of fields from the file to include as example inputs
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
The field to be used as the individual example weights.  These
weights are typically in the range: 0 < w <= 1, and specify how
important the example is in the training process


```
--data_weights=field
```

### Data Groups (Advanced; optional)
The field to be used to possibly determine the fold that the example
belongs to.

```
--data_weights=field
```

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
include.

```
--tabular_column_range
START
END
```
where START and END are integers specifying the column indices to
include (columns START ... END, including END).

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
