---
title: Data Translation
nav_order: 10
parent: File Loader
has_children: False
---

# Data Translation

Data Translation allows for the automatic translation of input/output
columns into a ML-ready representation.  These 
translation methods are only supported for __tabular files__ and __pickle files__.  

We currently support two forms of __Data Translation__:
- Enumerated data types: automatic translation from a string or other value to an integer
- Integer mapping: translation of an integer value into arbitrary feature value.  This can be used to implement, among other things, one-hot encoding

## Categorical Variables

A categorical variable typically takes on a value of one of several
strings.  Because deep networks work exclusively with numerical data,
string values must be translated to numerical values.  SuperDataSet
provides two ways to translate sets of strings into non-negative
integers.

### Option 1: Provide a list of strings

```
--data_columns_categorical_to_int
VAR0:STRA0,STRA1,STRA2, ...
VAR1:STRB0,STRB1,STRB2, ...
```

Here, columns VAR0 and VAR1 in the input data table is declared as
categorical variables.  VAR0 can take on one of several different
string values: STRA0, STRA1, STRA2, etc.  These string values are
mapped in order: STRA0 is mapped to integer 0 and STRA1 is mapped to
integer 1.  Likewise, VAR1 refers to another column in the data table
and has its own set of strings.

#### Example:

```
--data_columns_categorical_to_int
color:red,purple,green,pink,orange
number:one,two,three,four
```

Column __color__ in the data table is translated accordingly: red to
0, purple to 1, green to 2, pink to 3, and orange to 4.  Likewise,
column __number__ is translated accordingly: one to 0, two to 1, three
to 2, and four to 3.

### Option 2: Provide a list of strings and mapped integers

```
--data_columns_categorical_to_int_direct
VAR0:STRA0->INTA0,STRA1->INTA1,STRA2->INTA2, ...
VAR1:STRB0->INTB0,STRB1->INTB1,STRB2->INTB2, ...
```

Columns VAR0 and VAR1 in the input data table is declared as
categorical variables.  VAR0 can take on one of several different
string values: STRA0, STRA1, STRA2, etc.  These string values are
mapped to INTA0, INTA1, INTA2, respectively.  Likewise, VAR1 refers to
another column in the data table and has its own set of strings.

### Example:

```
--data_columns_categorical_to_int_direct
color:red->1,purple->3,green->0,pink->2,orange->1
number:one->1,two->2,three->0,four->3
```

Column __color__ in the data table is translated accordingly: red to
1, purple to 3, green to 0, pink to 2, and orange to 1.  Note that
multiple strings can be mapped to the same integer.  Likewise,
column __number__ is translated accordingly: one to 1, two to 2, three
to 0, and four to 3.

### Notes
- The translated value replaces the original value in the table (e.g., strings are changed to their corresponding integers)
- Column names and string values will have white space (e.g., spaces)
removed before the translation is performed.  However, internal white
space is kept for the translation process.
- All strings are case sensitive.
- When using this feature with pickle files, the specified column (field) must refer to a vector and not a matrix or other tensor.

### Examples
- [Iris Example](../../../examples/iris/README.md): Translate the specified output data column from a string to an integer, which is then used as a sparse categorical variable for training purposes

___

## Integer Mapping

Integer mapping is used to translate an integer value (typically a natural number from the Categorical Variable translation) into an arbitrary real value.  When one specifies a list of these translations, one can create a mapping from an integer value to a vector of features.

The general format is:
```
--data_columns_categorical_to_float_direct
VAR0:0->VALA0, 1->VALA1, 2->VALA2, ...
VAR0:0->VALB0, 1->VALB1, 2->VALB2, ...
    :
```

Column VAR0 from the data table is used to create a number of different features; here, we explicitly show two features; the ':' implies that there can be more.  The first row specifies the mapping from integer value to specific real numbers; an integer value of 0 is mapped to VALA0, 1 is mapped to VALA1, etc.  The second row specifies the mapping from the same column to a different set of real numbers.

### Example: Creating a One-Hot Encoding
A one-hot encoding maps an integer value to a vector containing all zeros (0s) except for a single one (1); the location of the 1 is determined by the value of the integer.

In this example, the original tabular column contains a string that can be one of several different values: foo, bar, or baz.  The first argument translates these to integers 0, 1, or 2, respectively.  The second argument (using all three rows) translates these integers into vectors.  

```
--data_columns_categorical_to_int
input_type:foo,bar,baz
--data_columns_categorical_to_float_direct
input_type:0->1.0, 1->0.0, 2->0.0
input_type:0->0.0, 1->1.0, 2->0.0
input_type:0->0.0, 1->0.0, 2->1.0
```

The mapping performed by these two steps is:
- __foo__ -> [1.0, 0.0, 0.0],
- __bar__ -> [0.0, 1.0, 0.0],
- __baz__ -> [0.0, 0.0, 1.0],

### Example: Creating Other Encodings
If the different string values have some real or ordinal relationship, then one can use these tools to create mappings that acknowledge these relationships.  For example:

```
--data_columns_categorical_to_int
value:one,two,three,four
--data_columns_categorical_to_float_direct
value:0->1.0,  1->0.25, 2->0.0,  3->0.0
value:0->0.0,  1->1.0,  2->0.25, 3->0.0
value:0->0.0,  1->0.0,  2->1.0,  3->0.25
value:0->0.0,  1->0.0,  2->0.0,  3->1.0
```
The mapping performed by these two steps is:
- __one__ ->   [1.0,  0.0,  0.0,  0.0],
- __two__ ->   [0.25, 1.0,  0.0,  0.0],
- __three__ -> [0.0,  0.25, 1.0,  0.0],
- __four__ ->  [0.0,  0.0,  0.25, 1.0],

The asymmetry of the mapping makes it clear how the strings are mapped to feature vectors

### Notes
- It is possible to translate multiple columns in this fashion
- In effect, the new feature vectors are added as a set of columns into the table, while the original column is deleted from the table
