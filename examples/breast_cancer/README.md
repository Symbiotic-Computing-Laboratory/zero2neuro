# Example: Detecting Malignant or Benign Breast Tumors

## Data
- File: [wdbc.csv](wdbc.csv)

## Network
- Sparse Categorical
- Twenty-nine inputs (Details on tumor)
- Two Hidden Layers
- Two outputs (M or B)

## Details
This model takes in the measurements and details of a tumor and predicts if it is benign or malignant. It uses sparse categorical so it gives percentages of how confident it is.

## Experiment Suggestion
Try to limit what measurements are being fed and see if you can find if there are some that are more important than others. 
