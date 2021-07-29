import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize, OneHotEncoder


VALID_METHODS = {'identity', 'OneHotDrug', 'OneHotCell', 'OneHotCombo',
                 'custom', 'random', 'only_train_cell_oneHot', 'only_train_cell_average'}


def make_prior(train, only_train, test, method='identity'):
    assert method in VALID_METHODS, f"Invalid method used, pick one of {VALID_METHODS}"
    all_data = np.vstack((train.to_numpy(), test.to_numpy()))

    normalization_factor = 1.5
    prior = None

    if method == 'identity':
        prior = np.eye(all_data.shape[0])
        return prior

    elif method == 'OneHotDrug':
        all_data_df = pd.concat([train, test])
        encoder = OneHotEncoder()
        prior = encoder.fit_transform(all_data_df.reset_index().intervention.to_numpy().reshape(-1, 1)).toarray()

    elif method == 'OneHotCell':
        all_data_df = pd.concat([train, test])
        encoder = OneHotEncoder()
        prior = encoder.fit_transform(all_data_df.reset_index().unit.to_numpy().reshape(-1, 1)).toarray()

    elif method == 'OneHotCombo':
        drug_scale_factor = 0.75
        cell_encoding = {}
        drug_encoding = {}

        for idx, unique_cell in enumerate(set(train.index.get_level_values('unit'))
                                    .union(set(test.index.get_level_values('unit')))):
            cell_encoding[unique_cell] = idx

        for idx, unique_drug in enumerate(set(train.index.get_level_values('intervention'))
                                    .union(set(test.index.get_level_values('intervention')))):
            drug_encoding[unique_drug] = idx

        prior = np.zeros((all_data.shape[0], len(cell_encoding) + len(drug_encoding)))

        x_row_index = 0
        for index, row in train.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            prior[x_row_index, cell_encoding[Cell_id]] = 1
            prior[x_row_index, len(cell_encoding) + drug_encoding[Drug_id]] = drug_scale_factor
            x_row_index += 1

        for index, row in test.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            prior[x_row_index, cell_encoding[Cell_id]] = 1
            prior[x_row_index, len(cell_encoding) + drug_encoding[Drug_id]] = drug_scale_factor
            x_row_index += 1

    elif method == 'random':
        dim = 100
        prior = np.random.rand(all_data.shape[0], dim)

    elif method == 'only_train_cell_oneHot':
        cell_scale_factor = 0.75

        cell_encoding = {}
        only_train = only_train.reset_index().groupby("intervention").agg('mean')
        dim = all_data.shape[1]

        encoding = only_train.to_numpy()
        encoding = np.vstack([encoding, np.average(encoding, axis=0)])
        only_train.reset_index(inplace=True)

        for idx, unique_cell in enumerate(set(train.index.get_level_values('unit'))
                                    .union(set(test.index.get_level_values('unit')))):
            cell_encoding[unique_cell] = idx

        prior = np.zeros((all_data.shape[0], len(cell_encoding) + dim))

        x_row_index = 0
        failed = 0
        for index, row in train.iterrows():
            Cell_id = str(index[0]) if str(index[0]) in cell_encoding else None
            Drug_id = str(index[1])

            try:
                drug_idx = only_train[only_train.intervention == Drug_id].index[0]
            except IndexError:
                failed += 1
                drug_idx = -1

            if Cell_id is not None:
                prior[x_row_index, cell_encoding[Cell_id]] = cell_scale_factor * np.linalg.norm(encoding[drug_idx, :])
            prior[x_row_index, len(cell_encoding):] = encoding[drug_idx, :]

            x_row_index += 1

        for index, row in test.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            try:
                drug_idx = only_train[only_train.intervention == Drug_id].index[0]
            except IndexError:
                failed += 1
                drug_idx = -1

            if Cell_id is not None:
                prior[x_row_index, cell_encoding[Cell_id]] = cell_scale_factor * np.linalg.norm(encoding[drug_idx, :])
            prior[x_row_index, len(cell_encoding):] = encoding[drug_idx, :]

            x_row_index += 1

    elif method == 'only_train_cell_average':
        cell_scale_factor = 0.75

        only_train = only_train.reset_index().groupby("intervention").agg('mean')
        dim = all_data.shape[1]

        encoding = only_train.to_numpy()
        encoding = np.vstack([encoding, np.average(encoding, axis=0)])
        only_train.reset_index(inplace=True)

        prior = np.zeros((all_data.shape[0], dim + dim))

        x_row_index = 0
        failed = 0
        for index, row in train.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            try:
                drug_idx = only_train[only_train.intervention == Drug_id].index[0]
            except IndexError:
                failed += 1
                drug_idx = -1

            cell_repr = train[train.index.get_level_values("unit") == Cell_id].mean().to_numpy()
            cell_repr /= np.linalg.norm(cell_repr)
            prior[x_row_index, :dim] = cell_scale_factor * np.linalg.norm(encoding[drug_idx, :]) * cell_repr
            prior[x_row_index, dim:] = encoding[drug_idx, :]

            x_row_index += 1

        for index, row in test.iterrows():
            Cell_id = str(index[0])
            Drug_id = str(index[1])

            try:
                drug_idx = only_train[only_train.intervention == Drug_id].index[0]
            except IndexError:
                failed += 1
                drug_idx = -1

            cell_repr = train[train.index.get_level_values("unit") == Cell_id].mean().to_numpy()
            cell_repr /= np.linalg.norm(cell_repr)
            prior[x_row_index, :dim] = cell_scale_factor * np.linalg.norm(encoding[drug_idx, :]) * cell_repr
            prior[x_row_index, dim:] = encoding[drug_idx, :]

            x_row_index += 1

    elif method == 'custom':
        raise NotImplementedError("Custom prior not implemented")

    prior = np.hstack([prior, normalization_factor * np.eye(all_data.shape[0])])
    normalize(prior, axis=1, copy=False)
    return prior
