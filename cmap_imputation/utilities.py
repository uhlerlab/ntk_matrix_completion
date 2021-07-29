import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from scipy.spatial.distance import cosine


def train_test_w_controls(allData, drugs_in_train, seed):
    np.random.seed(seed)
    allCells = allData.index.get_level_values('unit').drop_duplicates()

    train = None
    test = None

    for cell_type in allCells:
        cell_subset = allData[allData.index.get_level_values('unit') == cell_type]
        drug_set = cell_subset.index.get_level_values('intervention').drop_duplicates()
        train_drugs = np.random.choice(drug_set, size=drugs_in_train, replace=False)

        if train is None:
            train = cell_subset[cell_subset.index.get_level_values('intervention').isin(train_drugs)]
        else:
            train = pd.concat([train, cell_subset[cell_subset.index.get_level_values('intervention').isin(train_drugs)]])

        if test is None:
            test = cell_subset[~cell_subset.index.get_level_values('intervention').isin(train_drugs)]
        else:
            test = pd.concat([test, cell_subset[~cell_subset.index.get_level_values('intervention').isin(train_drugs)]])

    return train, test


def get_cosims(true, pred):
    cosims = []

    for row_id in range(true.shape[0]):
        i = true[row_id]
        j = pred[row_id]

        if not(np.abs(np.sum(i)) <= 1e-8 or np.abs(np.sum(j)) <= 1e-8):
            cosims.append(1 - cosine(i, j))

    return cosims


def get_splits_in_cell_type(allData, k=10, seed=5):
    cell_types = list(allData.index.get_level_values("unit").drop_duplicates())

    fold_iterators = [KFold(n_splits=k, shuffle=True, random_state=seed) for cell_type in cell_types]

    for idx, cell_type in enumerate(cell_types):
        dataSlice = allData[allData.index.get_level_values("unit") == cell_type]
        fold_iterators[idx] = fold_iterators[idx].split(dataSlice)

    for fold in range(k):
        train = None
        test = None

        for idx, cell_type in enumerate(cell_types):
            train_idx, test_idx = next(fold_iterators[idx])
            dataSlice = allData[allData.index.get_level_values("unit") == cell_type]

            if train is None:
                train = dataSlice.iloc[train_idx]
                test = dataSlice.iloc[test_idx]
            else:
                train = train.append(dataSlice.iloc[train_idx])
                test = test.append(dataSlice.iloc[test_idx])

        yield train, test
