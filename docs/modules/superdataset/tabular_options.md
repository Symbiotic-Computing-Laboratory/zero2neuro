---
title: Tabular File Options
nav_order: 10
parent: File Loader
has_children: False
---

# Tabular File Options (Advanced)
These arguments can be used for both csv and xlsx formatted files

## Header Row
By default, the header row is assumed to be the first row.  This
argument can be used to set a different row

```
--tabular_header_row=ROW
```
where ROW is the row index (counting from 1)

## Header Names
Some tabular files do not contain a header row.  Use this argument to
force the names of the columns
```
--tabular_header_names
COL1
COL2
 :
```
where COLX is the string name for column X.

## Tabular Columns
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

```
--tabular_column_list
COL1
COL2
 :
```
where COLX is an integer column index (counting from 1)

## Text Encoding

The name of the byte encoding for the tabular files.  

```
--tabular_encoding=ENCODING
```

The default value of _tabular_encoding_ is None (a reasonable choice).  In this case, SuperDataSet uses the content of the file to guess the character encoding.  If the format cannot be guessed, then SuperDataSet will use utf-8.

One can also specify the character encoding explicitly: [see the Python Standard Encodings Table](https://docs.python.org/3/library/codecs.html#standard-encodings)
