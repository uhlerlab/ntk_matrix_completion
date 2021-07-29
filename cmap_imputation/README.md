This repository provides various imputation methods for completing matrices of the form (cell type, drug type) x gene id. Enter one of the folders for additional documentation, or run `run_trials.sh` for a demo (the full demo can take many hours to complete). To see more information about a script, simply execute the following in a shell:
```bash
python script_name.py -h
```

For most default results, the cell type 'MCF7' was included only in the training set (not in the testing set), and the cell types {'SNU1040', 'HEK293T', 'HS27A'} were excluded from both training and testing sets, as they did not have at least 10 drugs in the dataset (which was the selected default number of folds in the K-Fold Cross Validation analysis).
