import argparse
import os
import pathlib
import pandas as pd


def str2bool(v):
    ''' Credits:
    https://stackoverflow.com/a/43357954
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clipped_int(v):
    try:
        if int(v) > 0:
            return int(v)
        else:
            raise argparse.ArgumentTypeError(f'Need integer value > 0, got {int(v)}')
    except ValueError:
        raise argparse.ArgumentTypeError(f'Need integer value > 0, got {v}')


def cli():
    curr_path = pathlib.Path(__file__).parent.absolute()

    options = argparse.ArgumentParser()
    options.add_argument('-i', dest='original_mat', default=os.path.join(curr_path.parent, "data", "largeTensor.pkl"),
                         type=str, help=f'Path to initial matrix, .pkl DF - Default {os.path.join(curr_path.parent, "data", "largeTensor.pkl")}')
    options.add_argument('-p', dest='preds_path', default=os.path.join(curr_path.parent, "predictions"),
                         type=str, help=f'Path to save/load predictions - Default {os.path.join(curr_path.parent, "predictions")}')
    options.add_argument('-n', dest='nearest_neighbors', default=10, type=clipped_int,
                         help='Number of nearest neighbors - Default 10')
    options.add_argument('-s', dest='seed', default=5, type=int,
                         help='Seed of random generators - Default 5')
    options.add_argument('-k', dest='kfold', default=10, type=int,
                         help='Number of folds for kFold CV, or <= 0 for Sparse Subset - Default 10')
    options.add_argument('-b', dest='sparse_subset', default=0, type=int,
                         help='Number of drugs in train, per-cell, for Sparse Subset, or <= 0 for KFold - Default 0')
    options.add_argument('-r', dest='remove_cells', default=['SNU1040', 'HEK293T', 'HS27A'], nargs='*',
                         help="Cells to remove from train and test - Default 'SNU1040' 'HEK293T' 'HS27A'")
    options.add_argument('-o', dest='only_train', default=['MCF7'], nargs='*',
                         help="Cells to include only in training, not in test splits - Default 'MCF7'")
    options.add_argument('-l', dest='plot', default=True, type=str2bool,
                         help="Whether or not to plot the results - Default: True")
    return options.parse_args()


def validate_inputs():
    args = cli()

    # Original Matrix
    if not os.path.exists(args.original_mat):
        raise AssertionError(f"Path to matrix, {args.original_mat} does not exist")
    allData = pd.read_pickle(args.original_mat)

    # Results
    if not os.path.exists(args.preds_path):
        os.mkdir(args.preds_path)

    # KFold vs Sparse Subset
    if (args.kfold <= 0 and args.sparse_subset <= 0) or \
        (args.kfold > 0 and args.sparse_subset > 0):
        raise AssertionError(f"Only one of kfold and sparse subset can be positive")

    if args.kfold > 0:
        method = ('kfold', args.kfold)
        path_prefix = f"DNPPKFold{args.kfold}Folds"
    else:
        method = ('sparse', args.sparse_subset)
        path_prefix = f"DNPPSparseSubset{args.sparse_subset}DrugsInTrain"

    allData = allData[~allData.index.get_level_values("unit").isin(args.remove_cells)]

    for cell in args.only_train:
        assert cell in allData.index.get_level_values("unit"), f"Cell {cell} not in input data"

    only_train = allData[allData.index.get_level_values("unit").isin(args.only_train)]
    allData = allData[~allData.index.get_level_values("unit").isin(args.only_train)]

    path_prefix += f"{args.seed}Seed{os.path.splitext(os.path.basename(args.original_mat))[0]}Source"

    return allData, only_train, method, args.seed, os.path.join(args.preds_path, path_prefix), args.nearest_neighbors, args.plot
