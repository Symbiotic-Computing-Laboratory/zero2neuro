[Base Index](../../index.md)  
[Previous Index](index.md)  
# Data Translation

## Categorical Variables

A categorical variable typically takes on a value of one of several
strings.  Because deep networks work exclusively with numerical data,
string values must be translated to numerical values.  SuperDataSet
provides two ways to translate sets of strings into non-negative
integers.

__Option 1__: Provide a list of strings

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

Example:

```
--data_columns_categorical_to_int
color:red,purple,green,pink,orange
number:one,two,three,four
```

Column __color__ in the data table is translated accordingly: red to
0, purple to 1, green to 2, pink to 3, and orange to 4.  Likewise,
column __number__ is translated accordingly: one to 0, two to 1, three
to 2, and four to 3.

__Option 2__: Provide a list of strings and mapped integers

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

Example:

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

__Notes:__
- Column names and string values will have white space (e.g., spaces)
removed before the translation is performed.  However, internal white
space are kept for the translation process.
- All strings are case sensitive.


## One-Hot Encoding
TODO
