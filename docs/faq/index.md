---
title: Frequently Asked Questions
nav_order: 60
parent: Zero2Neuro
---

# Frequently Asked Questions

## Constructing Data Sets

- __How do I split my one data file into 80% training, 10% validation, and 10% testing?__ 
The simple way to implement this is to:
   1. Split the data randomly into 10 folds
   2. Use Holistic N-Fold Cross-Validation using the default training set size (N-2=8 folds)



