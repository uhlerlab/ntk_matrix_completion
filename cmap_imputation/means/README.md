This folder provides scripts for running averaging-based methods to impute CMAP data. It includes a Command-Line-Interface for interacting with the scripts. There are choices for selecting a scale-factor, which computes the average over the training set as so, for each (cell, drug) combination in the test set:

<p align="center">
  prediction = cell_mean * scale_factor + drug_mean * (1 - scale_factor) 
  </p>

The available scripts in this directory are:
- [means.py](means.py)
