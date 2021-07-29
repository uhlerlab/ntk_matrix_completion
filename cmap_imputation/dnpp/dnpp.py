import heapq
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


def get_all_correlations(trainData):
    """ Gets all correlations between drug types for a single cell in trainData
    """
    cellCorrelations = {cell: None for cell in trainData.index.get_level_values("unit").drop_duplicates()}

    for cell_type in cellCorrelations:
        cellSubset = trainData[trainData.index.get_level_values("unit") == cell_type]
        labels = cellSubset.index.get_level_values("intervention")
        corrMatrix = np.corrcoef(cellSubset.to_numpy(), rowvar=True)
        cellCorrelations[cell_type] = pd.DataFrame(data=corrMatrix, index=labels, columns=labels)

    return cellCorrelations


def dnpp_fit_evaluate(train_data, test_data, k):
    """ Fits DNPP model on train data, and evaluates it on test data, saving results to predictions folder
    """
    predicted_results = np.zeros_like(test_data)
    train_cell_means = train_data.groupby(level='unit').mean()

    correlationMatrices = get_all_correlations(train_data)
    cached_sim = {}

    def similarity(drug_1, drug_2):
        """ Returns the similarity between drugs 1 and 2 relative to the train set
        """
        drug_combo = frozenset({drug_1, drug_2})

        if drug_combo in cached_sim:
            return cached_sim[drug_combo]
        else:
            cell_set = set(train_data[train_data.index.get_level_values("intervention") == drug_1].index.get_level_values("unit")).intersection(
                set(train_data[train_data.index.get_level_values("intervention") == drug_2].index.get_level_values("unit"))
            )

            if len(cell_set) == 0:
                sim = None
            else:
                sim = sum(correlationMatrices[cell].loc[(drug_1, drug_2)] for cell in cell_set) / len(cell_set)

            cached_sim[drug_combo] = sim

        return cached_sim[drug_combo]

    test_idx = 0
    num_filled_1d = 0

    for index, _ in tqdm(test_data.iterrows(), total=len(test_data)):
        Cell_id = str(index[0])
        Drug_id = str(index[1])

        neighbor_candidates = train_data[(train_data.index.get_level_values("unit") == Cell_id) &
                                         (train_data.index.get_level_values("intervention") != Drug_id)]\
                                             .index.get_level_values("intervention")

        min_heap = []

        # Iterate through neighbor drugs and push maxima to the heap
        for neighbor_drug in tqdm(neighbor_candidates):
            cor = similarity(Drug_id, neighbor_drug)

            if cor is None:
                continue
            elif len(min_heap) < k:
                heapq.heappush(min_heap, (cor, neighbor_drug))
            elif cor > min_heap[0][0]:
                heapq.heappushpop(min_heap, (cor, neighbor_drug))

        k_neighbors = np.array([train_data.loc[(Cell_id, drug)].to_numpy() for _, drug in min_heap])
        weights = np.array([wt for wt, _ in min_heap]).ravel()
        weights = np.clip(weights, 0, 1)

        # If none of the samples have positive correlation, predict mean over actions
        if np.sum(weights) == 0:
            num_filled_1d += 1
            pred = train_cell_means.loc[Cell_id].to_numpy()
            predicted_results[test_idx] = pred
        else:
            weights /= np.sum(weights)
            pred = np.dot(weights, k_neighbors)
            predicted_results[test_idx] = pred

        test_idx += 1

    return predicted_results, num_filled_1d


if __name__ == '__main__':
    allData, only_train, method, SEED, path_prefix, neighbors, plot = validate_inputs()

    if method[0] == 'kfold':
        iterator = tqdm(get_splits_in_cell_type(allData, k=method[1], seed=SEED), total=method[1])
    elif method[0] == 'sparse':
        iterator = tqdm([train_test_w_controls(allData, drugs_in_train=method[1], seed=SEED)], total=1)
    else:
        raise AssertionError("Unknown method")

    all_true = None  # Ground truths for all fold(s)
    all_metrics_dnpp = None  # Metrics (R^2, Cosine Similarity, Pearson R) for each entry
    dnpp_predictions = None  # Predictions for all fold(s)
    splits = None  # Per-fold metrics

    # Iterate over all predictions and populate metrics matrices
    for train, test in iterator:
        train_modified = pd.concat([only_train, train])
        true = test.to_numpy()

        # SAFETY - do not allow predictor to see test data
        test_copy = test.copy()
        for col in test_copy.columns:
            test_copy[col].values[:] = 0

        results_dnpp, num1d = dnpp_fit_evaluate(train_modified, test_copy, neighbors)
        prediction_dnpp = pd.DataFrame(data=results_dnpp,
                                       index=test.index,
                                       columns=test.columns)
        dnpp_predictions = prediction_dnpp if dnpp_predictions is None\
                            else pd.concat([dnpp_predictions, prediction_dnpp])
        print(f"\nNumber Filled with Mean over Actions {num1d} out of {len(test)}")

        temp_df_dnpp = pd.DataFrame(data=np.column_stack([r2_score(true.T, results_dnpp.T, multioutput='raw_values'),
                                                            get_cosims(true, results_dnpp)]), index=test.index)
        temp_df_dnpp.columns = ['r_squared', 'cosine_similarity']
        new_splits = pd.DataFrame(data=np.column_stack([temp_df_dnpp['r_squared'].mean(),
                                                        temp_df_dnpp['cosine_similarity'].mean(),
                                                        len(train),
                                                        len(test)]))
        new_splits.columns = ['r_squared', 'cosine_similarity', 'train_size', 'test_size']

        splits = new_splits if splits is None else pd.concat([splits, new_splits], ignore_index=True)
        all_metrics_dnpp = temp_df_dnpp if all_metrics_dnpp is None else pd.concat([all_metrics_dnpp, temp_df_dnpp])

        all_true = pd.concat([all_true, test]) if all_true is not None else test

    r_dnpp, p_dnpp = pearsonr(all_true.to_numpy().ravel(), dnpp_predictions.to_numpy().ravel())
    all_metrics_dnpp['pearson_r'] = r_dnpp
    all_metrics_dnpp['pearson_r_p_value'] = p_dnpp

    pd.to_pickle(all_true, path_prefix + "GroundTruth.pkl")
    pd.to_pickle(dnpp_predictions, path_prefix + "Predictions.pkl")
    pd.to_pickle(all_metrics_dnpp, path_prefix + "AllMetrics.pkl")
    pd.to_pickle(splits, path_prefix + "SplitMetrics.pkl")

    if plot:
        plot_graphics(dnpp_predictions, all_true, all_metrics_dnpp, path_prefix, True)
