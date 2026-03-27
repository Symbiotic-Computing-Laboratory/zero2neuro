# Example: Detecting Malignant or Benign Breast Tumors

## Data
- File: [wdbc.csv](wdbc.csv)

## Network
- Binary Classification 
- Twenty-nine inputs (Details on tumor)
- Two Hidden Layers
- One output (M or B, Binary)

## Details
This model takes in the measurements and details of a tumor and predicts if it is benign or malignant. It uses binary classification to give a prediction of confidence between the tumor being malignant or benign. The output function is a sigmoid function which is contained within a 0-1 range which acts as a probability function of 0-100%.

## Experiment Suggestion
Try to limit what measurements are being fed and see if you can find if there are some that are more important than others. 
