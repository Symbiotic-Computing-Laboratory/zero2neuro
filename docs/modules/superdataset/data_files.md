---
title: Data File Formats
nav_order: 10
parent: SuperDataSet
---
# Data File Formats

Zero2Neuro supports the import of a range of different data formats. 
One typically specifies the format within the data configuration text
file using the argument:

```
--data_format=XXX
```
where XXX is one of the following:
  
## Tabular (Beginner)
```
--data_format=tabular
```

The tabular data format "spreadsheet" style formatting of the data
into rows and columns.  Zero2Neuro supports the import of both .csv
and .xlsx files, which can be produced by a range of application
programs (including Excel, Google Sheets, and Libreoffice). 

Within the tabular file, individual data examples are rows.  Columns
contain different input features, desired outputs, and other
example-specific information.  By default, the first row of the data
table is the __header__, which contains the names of each of the
columns.  These names are used to specify which columns will be used
as input features, and which will be used as desired outputs.

## Tabular-Indirect (Beginner)
```
--data_format=tabular-indirect
```

The tabular indirect data format is the same as tabular format, with
the exception that one column can be used to specify the path to a
unique file (one for each example) that contains a 2D image.  Zero2Neuro
supports a range of image formats.


## Pickle (Intermediate)
```
--data_format=pickle
```

Pickle files contain serialized Python objects that can
later be unpacked into data examples.  For the Zero2Neuro pickle data
format, the file contains a single dictionary object:
- The dictionary keys are named data fields 
   - these named fields play the same role as the column names in the
tabular files.   
- Each value associated with a key is a numpy
array of shape (N, d0, ...):
   - N is the number of examples (same for all values in the
dictionary),
   - d0, ..., dk-1 is some k dimensions that make up a single example
for that field
- Examples:
   - For color images, the shape will be (N, R, C, 3) (image rows,
columns, and color)
   - For individual examples that are vectors, the shape will be (N,
d0)
   - For timeseries data, the shape will be (N, T, d), where T is the
number of time steps, and d is the vector for each time step.

Generating pickle-formatted files requires a degree of Python
programming.


## TF-Dataset (Advanced)
```
--data_format=tf-dataset
```

TF-Datasets are a way to handle data sets that are too large to be
stored in memory and/or that require a lot of effort to fetch
from disk (e.g., very large images).  TF-Datasets allow the model to
be trained on a subset of the data (a batch) while the next batch 
fetched from disk.  TF-Datasets also support caching fetched data to
high-speed local storage (RAM or high-speed disk) the first time that
it is loaded; the next 
time the data batch is needed for training, it is automatically pulled
from the high-speed storage instead of having to wait for the disk to
load it again.

Generating TF-dataset-formatted files requires a high degree of Python
programming.

## Pickle-array
(future)

## NetCDF
(future)
  
