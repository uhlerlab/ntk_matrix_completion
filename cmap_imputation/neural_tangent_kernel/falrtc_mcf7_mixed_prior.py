import sys
import pathlib
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize
from auto_tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from ntk import predict_space_opt_CMAP_data
from cli import validate_inputs

sys.path.insert(1, str(pathlib.Path(__file__).parent.absolute().parent))
from utilities import train_test_w_controls, get_cosims, get_splits_in_cell_type

sys.path.insert(1, os.path.join(str(pathlib.Path(__file__).parent.absolute().parent), "graphical"))
from singular_graphics import plot_graphics


if __name__ == '__main__':
    technique = 'FALRTC'
    allData, only_train, method, SEED, path_prefix, plot, threshold = validate_inputs(technique=technique,
                                                                                      mixed_prior=True)

    # Delete 'FALRTC'
    ind = path_prefix.rfind(technique)
    save_prefix = path_prefix[:ind] + "MixedFALRTCNTK" + path_prefix[ind + len(technique):]

    only_train_indexed = only_train.reset_index()
    dim = allData.shape[1]
    encoding = only_train.to_numpy()
    encoding = np.vstack([encoding, np.average(encoding, axis=0)])

    if method[0] == 'kfold':
        iterator = tqdm(get_splits_in_cell_type(allData, k=method[1], seed=SEED), total=method[1])
    else:
        raise AssertionError("Only works with KFold CV")

    all_true = None  # Ground truths for all fold(s)
    all_metrics_ntk = None  # Metrics (R^2, Cosine Similarity, Pearson R) for each entry
    ntk_predictions = None  # Predictions for all fold(s)
    splits = None  # Per-fold metrics

    threshold = threshold
    cell_importance = 1.25
    cell_prior = 'train_mean'  # Alternatively, 'oneHot'
    normalization_factor = 1.5

    # Iterate over all predictions and populate metrics matrices
    for train, test in iterator:
        # Complex prior involving FaLRTC and Only Train Combination
        test_all = test.sort_values(by='unit',
                                    axis=0,
                                    ascending=False,
                                    key=lambda x: train.index.get_level_values("unit").value_counts()[x])
        train_drug_counts = train.index.get_level_values("unit").value_counts()
        test_cell_indices = test_all.index.get_level_values("unit")

        # Split data into two sections: low and high based on how many drugs the cell has in the train set
        test_low_data = test_all[test_cell_indices.isin(train_drug_counts[train_drug_counts < threshold].index)]
        test_high_data = test_all[test_cell_indices.isin(train_drug_counts[train_drug_counts >= threshold].index)]

        all_data_low = pd.concat([train, test_low_data]).to_numpy()
        all_data_high = pd.concat([train, test_high_data]).to_numpy()

        ##### SAFETY
        all_data_low[train.shape[0]:, :] = 0
        all_data_high[train.shape[0]:, :] = 0
        ##### SAFETY

        mask_low = np.ones_like(all_data_low)
        mask_low[len(train):, :] = 0
        mask_high = np.ones_like(all_data_high)
        mask_high[len(train):, :] = 0

        if cell_prior == 'oneHot':
            cell_encoding = {}

            for idx, unique_cell in enumerate(set(train.index.get_level_values('unit'))
                                        .union(set(test_all.index.get_level_values('unit')))):
                cell_encoding[unique_cell] = idx

            embedding_low = np.zeros((all_data_low.shape[0], len(cell_encoding) + dim))
            embedding_high = np.zeros((all_data_high.shape[0], dim))

        elif cell_prior == 'train_mean':
            embedding_low = np.zeros((all_data_low.shape[0], dim + dim))
            embedding_high = np.zeros((all_data_high.shape[0], dim))

        else:
            raise AssertionError("Please pick one of oneHot or train_mean as cell_prior")

        x_row_index = 0
        failed = 0

        for index, row in train.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            try:
                drug_idx = only_train_indexed[only_train_indexed.intervention == Drug_id].index[0]
            except IndexError:
                failed += 1
                drug_idx = -1

            if (cell_prior == 'oneHot') and (Cell_id in cell_encoding):
                embedding_low[x_row_index, cell_encoding[Cell_id]] = cell_importance * np.linalg.norm(encoding[drug_idx, :])
                embedding_low[x_row_index, len(cell_encoding):] = encoding[drug_idx, :]

            elif cell_prior == 'train_mean':
                cell_repr = train[train.index.get_level_values("unit") == Cell_id].mean().to_numpy()
                cell_repr /= np.linalg.norm(cell_repr)
                embedding_low[x_row_index, :dim] = cell_importance * np.linalg.norm(encoding[drug_idx, :]) * cell_repr
                embedding_low[x_row_index, dim:] = encoding[drug_idx, :]

            x_row_index += 1

        for index, row in test_low_data.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])
            try:
                drug_idx = only_train_indexed[only_train_indexed.intervention == Drug_id].index[0]
            except IndexError:
                failed += 1
                drug_idx = -1

            if (cell_prior == 'oneHot') and (Cell_id in cell_encoding):
                embedding_low[x_row_index, cell_encoding[Cell_id]] = cell_importance * np.linalg.norm(encoding[drug_idx, :])
                embedding_low[x_row_index, len(cell_encoding):] = encoding[drug_idx, :]

            elif cell_prior == 'train_mean':
                cell_repr = train[train.index.get_level_values("unit") == Cell_id].mean().to_numpy()
                cell_repr /= np.linalg.norm(cell_repr)
                embedding_low[x_row_index, :dim] = cell_importance * np.linalg.norm(encoding[drug_idx, :]) * cell_repr
                embedding_low[x_row_index, dim:] = encoding[drug_idx, :]

            x_row_index += 1

        try:
            falrtc_encoding = pd.read_pickle(path_prefix + "Predictions.pkl")

        except FileNotFoundError:
            raise AssertionError(f"Please train the FaLRTC Model at path {path_prefix + 'Predictions.pkl'}")

        falrtc_encoding = falrtc_encoding[falrtc_encoding.index.isin(test_high_data.index)][test_high_data.columns].reindex(test_high_data.index)

        ##### Encoding for examples with high number of elements in train
        embedding_high[:train.shape[0], :] = train.to_numpy()
        embedding_high[train.shape[0]:, :] = falrtc_encoding[:len(test_high_data)]

        ##### NTK
        print("\nFinished Encoding")
        print(f"{failed} Failed Encodings out of {x_row_index}")

        results_ntk_low = None
        results_ntk_high = None

        if len(test_low_data) != 0:
            embedding_low = np.hstack([embedding_low, normalization_factor * np.eye(all_data_low.shape[0])])
            normalize(embedding_low, axis=1, copy=False)
            results_ntk_low = predict_space_opt_CMAP_data(all_data_low, mask_low, len(test_low_data), X=embedding_low)

        if len(test_high_data) != 0:
            embedding_high = np.hstack([embedding_high, normalization_factor * np.eye(all_data_high.shape[0])])
            normalize(embedding_high, axis=1, copy=False)
            results_ntk_high = predict_space_opt_CMAP_data(all_data_high, mask_high, len(test_high_data), X=embedding_high)

        if results_ntk_high is not None and results_ntk_low is not None:
            results_ntk = np.concatenate([results_ntk_high, results_ntk_low])
        elif results_ntk_high is not None:
            results_ntk = results_ntk_high
        else:
            results_ntk = results_ntk_low

        prediction_ntk = pd.DataFrame(data=results_ntk,
                                      index=test_all.index,
                                      columns=test_all.columns)
        ntk_predictions = prediction_ntk if ntk_predictions is None\
                            else pd.concat([ntk_predictions, prediction_ntk])

        true = test_all.to_numpy()
        temp_df_ntk = pd.DataFrame(data=np.column_stack([r2_score(true.T, results_ntk.T, multioutput='raw_values'),
                                                            get_cosims(true, results_ntk)]), index=test_all.index)
        temp_df_ntk.columns = ['r_squared', 'cosine_similarity']

        new_splits = pd.DataFrame(data=np.column_stack([temp_df_ntk['r_squared'].mean(),
                                                        temp_df_ntk['cosine_similarity'].mean(),
                                                        len(train),
                                                        len(test)]))
        new_splits.columns = ['r_squared', 'cosine_similarity', 'train_size', 'test_size']

        splits = new_splits if splits is None else pd.concat([splits, new_splits], ignore_index=True)

        all_metrics_ntk = temp_df_ntk if all_metrics_ntk is None else pd.concat([all_metrics_ntk, temp_df_ntk])

        all_true = pd.concat([all_true, test_all]) if all_true is not None else test_all

    r_ntk, p_ntk = pearsonr(all_true.to_numpy().ravel(), ntk_predictions.to_numpy().ravel())
    all_metrics_ntk['pearson_r'] = r_ntk
    all_metrics_ntk['pearson_r_p_value'] = p_ntk

    pd.to_pickle(all_true, save_prefix + "GroundTruth.pkl")
    pd.to_pickle(ntk_predictions, save_prefix + "Predictions.pkl")
    pd.to_pickle(all_metrics_ntk, save_prefix + "AllMetrics.pkl")
    pd.to_pickle(splits, save_prefix + "SplitMetrics.pkl")

    if plot:
       plot_graphics(ntk_predictions, all_true, all_metrics_ntk, save_prefix, True)
