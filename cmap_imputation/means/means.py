import sys
import pathlib
import os

import numpy as np
import pandas as pd
from cli import validate_inputs
from auto_tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utilities import train_test_w_controls, get_cosims, get_splits_in_cell_type

sys.path.insert(1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical"))
from singular_graphics import plot_graphics


def compute_means(train_data, test_data, scale_factor):
    """ Fits means model on train data, and evaluates it on test data, saving results to predictions folder
    """
    predicted_results = np.zeros_like(test_data)

    # Precompute train cell and drug means
    train_cell_means = train_data.groupby(level='unit').mean()
    train_drug_means = train_data.groupby(level='intervention').mean()

    # Compute the mean of means as well
    overall_cell_avg = train_cell_means.mean().to_numpy()
    overall_drug_avg = train_drug_means.mean().to_numpy()

    # For each row in test data, populate the prediction as the linear
    # combination of the means of its cell and drug type in the train set
    test_idx = 0
    for index, _ in tqdm(test_data.iterrows(), total=len(test_data)):
        Cell_id = str(index[0])
        Drug_id = str(index[1])

        # The cell id exists in the train
        if Cell_id in train_cell_means.index:
            cell_component = train_cell_means.loc[Cell_id].to_numpy()
        # Cell id does not exist in train, simply use the mean of means
        else:
            cell_component = overall_cell_avg

        if Drug_id in train_drug_means.index:
            drug_component = train_drug_means.loc[Drug_id].to_numpy()
        else:
            drug_component = overall_drug_avg

        # Linear combination of cell and drug components as prediction
        predicted_results[test_idx] = cell_component * scale_factor + \
                                      drug_component * (1 - scale_factor)
        test_idx += 1

    return predicted_results


if __name__ == '__main__':
    # Get inputs from CLI
    allData, only_train, method, SEED, path_prefix, scale_factor, plot = validate_inputs()

    # Select correct iterator based on desired method
    if method[0] == 'kfold':
        iterator = tqdm(get_splits_in_cell_type(allData, k=method[1], seed=SEED), total=method[1])
    elif method[0] == 'sparse':
        iterator = tqdm([train_test_w_controls(allData, drugs_in_train=method[1], seed=SEED)], total=1)
    else:
        raise AssertionError("Unknown method")

    all_true = None  # Ground truths for all fold(s)
    all_metrics_means = None  # Metrics (R^2, Cosine Similarity, Pearson R) for each entry
    means_predictions = None  # Predictions for all fold(s)
    splits = None  # Per-fold metrics

    for train, test in iterator:
        train_modified = pd.concat([only_train, train])
        true = test.to_numpy()

        # SAFETY - do not allow predictor to see test data
        test_copy = test.copy()
        for col in test_copy.columns:
            test_copy[col].values[:] = 0

        # Obtain predictions for means
        results_means = compute_means(train_modified, test_copy, scale_factor)
        prediction_means = pd.DataFrame(data=results_means,
                                        index=test.index,
                                        columns=test.columns)
        means_predictions = prediction_means if means_predictions is None\
                            else pd.concat([means_predictions, prediction_means])

        # Populate new metrics
        temp_df_means = pd.DataFrame(data=np.column_stack([r2_score(true.T, results_means.T, multioutput='raw_values'), get_cosims(true, results_means)]), index=test.index)
        temp_df_means.columns = ['r_squared', 'cosine_similarity']
        new_splits = pd.DataFrame(data=np.column_stack([temp_df_means['r_squared'].mean(),
                                                        temp_df_means['cosine_similarity'].mean(),
                                                        len(train),
                                                        len(test)]))
        new_splits.columns = ['r_squared', 'cosine_similarity', 'train_size', 'test_size']

        splits = new_splits if splits is None else pd.concat([splits, new_splits], ignore_index=True)
        all_metrics_means = temp_df_means if all_metrics_means is None else pd.concat([all_metrics_means, temp_df_means])

        all_true = pd.concat([all_true, test]) if all_true is not None else test

    # Compute pearson R and save all metrics
    r_means, p_means = pearsonr(all_true.to_numpy().ravel(), means_predictions.to_numpy().ravel())
    all_metrics_means['pearson_r'] = r_means
    all_metrics_means['pearson_r_p_value'] = p_means

    pd.to_pickle(all_true, path_prefix + "GroundTruth.pkl")
    pd.to_pickle(means_predictions, path_prefix + "Predictions.pkl")
    pd.to_pickle(all_metrics_means, path_prefix + "AllMetrics.pkl")
    pd.to_pickle(splits, path_prefix + "SplitMetrics.pkl")

    if plot:
        plot_graphics(means_predictions, all_true, all_metrics_means, path_prefix, True)
