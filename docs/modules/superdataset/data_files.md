# Data File Formats
  
## Tabular  
Tabular data format refers to a spreadsheet formatting of the data with rows and columns. Two popular types of tabular data files are .csv files and .xlsx files. When using tabular data you take the columns as the data's features and the rows as a specific point in the data. The individual cells inside of a row and column contain the data that is used by the model during experiments.
  
## Tabular-Indirect
Tabular indirect is the same as tabular data, with .csv and .xlsx but instead of having to specify specific values in the cells it can instead specify file paths. This is useful for datasets with large individual data point sizes like with images, as it allows referring to a location rather than the data itself. It also helps contain multiple datasets in one file limiting the amount of clutter in configuration files.  

## Pickle  
Pickle files are files containing serialized python objects that can later be deserialized. A common practice is storing dictionaries in pickle files in which there is a key and value(s) behind that key. This is very similair to the columns in a tabular file and can be used the same way for providing data to the model.  

## Pickle-array
(future)

## NetCDF
(future)
  
## TF-Dataset  
TF-Datasets are a way to process data that is too large to be stored in memory. It allows things such as prefetching, which allows a preparation of data during the experiment minimizing time between training and grabbing data. It supports several data formats like csv files and numpy arrays and has a lot of customizability for how data is managed. It also allows caching which stored preprocessed data in a file or in ram.  
