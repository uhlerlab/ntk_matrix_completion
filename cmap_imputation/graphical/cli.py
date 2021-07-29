import argparse
import os
import pathlib
import pandas as pd


def cli():
    curr_path = pathlib.Path(__file__).parent.absolute()

    options = argparse.ArgumentParser()
    options.add_argument('-p', dest='preds_path', default=os.path.join(curr_path.parent, "predictions"),
                         type=str, help=f'Path to save/load predictions - Default {os.path.join(curr_path.parent, "predictions")}')
    options.add_argument('-n', dest='ntk_path', required=True,
                         type=str, help='Path to NTK All Metrics Dataset to compare to')
    options.add_argument('-d', dest='dnpp_path', required=True,
                         type=str, help='Path to DNPP All Metrics Dataset to compare to')
    options.add_argument('-s', dest='seed', default=5, type=int,
                         help='Seed of random generators - Default 5')
    options.add_argument('-m', dest='metric', default='both', type=str,
                         help='Statistic to plot (cosine_similarity, r_squared, both) - Default both')

    return options.parse_args()


def validate_inputs():
    args = cli()

    # Results
    if not os.path.exists(args.preds_path):
        os.mkdir(args.preds_path)

    # NTK and DNPP results
    if not (os.path.exists(args.ntk_path) and os.path.exists(args.dnpp_path)):
        raise AssertionError(f"Path {args.ntk_path} or {args.dnpp_path} does not exist")

    path_prefix = f"DNPPNTKComparison{os.path.basename(args.ntk_path).split('.')[0]}and{os.path.basename(args.dnpp_path).split('.')[0]}"

    return os.path.join(args.preds_path, path_prefix),\
            args.metric, pd.read_pickle(args.ntk_path), pd.read_pickle(args.dnpp_path)
