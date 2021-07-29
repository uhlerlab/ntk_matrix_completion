import sys
import pathlib
import os
from prior import make_prior
from cli import validate_inputs

import numpy as np
import pandas as pd
from auto_tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utilities import train_test_w_controls, get_cosims, get_splits_in_cell_type

sys.path.insert(1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical"))
from singular_graphics import plot_graphics


PI = np.pi


def kappa(x):
    return (x * (PI - np.arccos(x)) + np.sqrt(1 - np.square(x))) / PI +\
           (x * (PI - np.arccos(x))) / PI


def predict_space_opt_CMAP_data(all_data, mask, num_test_rows, X):
    """
    Space optimized version for CMap Data
    """
    all_data = all_data.T
    mask = mask.T

    num_observed = int(np.sum(mask[0:1, :]))
    num_missing = mask[0:1, :].shape[-1] - num_observed

    K_matrix = np.zeros((num_observed, num_observed))
    k_matrix = np.zeros((num_observed, num_missing))

    observed_data = all_data[:, :num_observed]
    X_cross_terms = kappa(np.clip(X @ X.T, -1, 1))

    K_matrix[:, :] = X_cross_terms[:num_observed, :num_observed]
    k_matrix[:, :] = X_cross_terms[:num_observed, num_observed: num_observed + num_missing]

    results = np.linalg.solve(K_matrix, observed_data.T).T @ k_matrix

    assert results.shape == (all_data.shape[0], num_test_rows), "Results malformed"

    return results.T


if __name__ == '__main__':
    print("Modify make_prior in prior.py to add a custom prior! There are a few choices to start.")
    allData, only_train, method, SEED, path_prefix, plot, prior = validate_inputs()
    path_prefix += f"{prior}Prior"

    if method[0] == 'kfold':
        iterator = tqdm(get_splits_in_cell_type(allData, k=method[1], seed=SEED), total=method[1])
    elif method[0] == 'sparse':
        iterator = tqdm([train_test_w_controls(allData, drugs_in_train=method[1], seed=SEED)], total=1)
    else:
        raise AssertionError("Unknown method")

    all_true = None  # Ground truths for all fold(s)
    all_metrics_ntk = None  # Metrics (R^2, Cosine Similarity, Pearson R) for each entry
    ntk_predictions = None  # Predictions for all fold(s)
    splits = None  # Per-fold metrics

    # Iterate over all predictions and populate metrics matrices
    for train, test in iterator:
        X = make_prior(train, only_train, test, prior)

        all_data = pd.concat([train, test]).to_numpy()
        ##### SAFETY
        all_data[train.shape[0]:, :] = 0
        ##### SAFETY
        mask = np.ones_like(all_data)
        mask[len(train):, :] = 0

        results_ntk = predict_space_opt_CMAP_data(all_data, mask, len(test), X=X)

        prediction_ntk = pd.DataFrame(data=results_ntk,
                                      index=test.index,
                                      columns=test.columns)
        ntk_predictions = prediction_ntk if ntk_predictions is None\
                            else pd.concat([ntk_predictions, prediction_ntk])

        true = test.to_numpy()
        temp_df_ntk = pd.DataFrame(data=np.column_stack([r2_score(true.T, results_ntk.T, multioutput='raw_values'),
                                                            get_cosims(true, results_ntk)]), index=test.index)
        temp_df_ntk.columns = ['r_squared', 'cosine_similarity']
        print(temp_df_ntk)
        print(train.shape, test.shape)
        new_splits = pd.DataFrame(data=np.column_stack([temp_df_ntk['r_squared'].mean(),
                                                        temp_df_ntk['cosine_similarity'].mean(),
                                                        len(train),
                                                        len(test)]))
        new_splits.columns = ['r_squared', 'cosine_similarity', 'train_size', 'test_size']

        splits = new_splits if splits is None else pd.concat([splits, new_splits], ignore_index=True)
        all_metrics_ntk = temp_df_ntk if all_metrics_ntk is None else pd.concat([all_metrics_ntk, temp_df_ntk])
        all_true = pd.concat([all_true, test]) if all_true is not None else test

    r_ntk, p_ntk = pearsonr(all_true.to_numpy().ravel(), ntk_predictions.to_numpy().ravel())
    all_metrics_ntk['pearson_r'] = r_ntk
    all_metrics_ntk['pearson_r_p_value'] = p_ntk

    pd.to_pickle(all_true, path_prefix + "GroundTruth.pkl")
    pd.to_pickle(ntk_predictions, path_prefix + "Predictions.pkl")
    pd.to_pickle(all_metrics_ntk, path_prefix + "AllMetrics.pkl")
    pd.to_pickle(splits, path_prefix + "SplitMetrics.pkl")

    if plot:
        plot_graphics(ntk_predictions, all_true, all_metrics_ntk, path_prefix, True)
