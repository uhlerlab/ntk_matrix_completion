import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import pearsonr


def plot_graphics(predictions, ground_truth, all_metrics, save_path,
                  incl_stats, figsize=(10, 10), dpi=600):
    """ Plot graphics
    """
    fig, ax = plt.subplots(figsize=figsize)
    all_true = ground_truth.to_numpy()
    all_preds = predictions.to_numpy()

    ax.scatter(all_true.ravel(), all_preds.ravel(), color='blue', s=5, alpha=0.1)

    ## DNPP GRAPHICAL
    lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ## Results processing DNPP
    r, p = pearsonr(all_true.ravel(), all_preds.ravel())
    r_2_vals = all_metrics['r_squared']
    cosim = all_metrics['cosine_similarity']

    if incl_stats:
        ax.annotate(f"Pearson R: {round(r, 5)}, p-value: {p}", (lims[0] + 0.1, lims[1] - 0.1))
        ax.annotate(f"R^2 Avg, Median, Std. Dev: {round(np.average(r_2_vals), 5)}, {round(np.median(r_2_vals), 5)}, {round(np.std(r_2_vals), 5)}", (lims[0] + 0.1, lims[1] - 0.15))
        ax.annotate(f"Cosine Sim Avg, Median, Std. Dev: {round(np.average(cosim), 5)}, {round(np.median(cosim), 5)}, {round(np.std(cosim), 5)}", (lims[0] + 0.1, lims[1] - 0.2))
    else:
        print(f"Pearson R: {round(r, 5)}; Pearson p-value: {p}")
        print(f"R^2 Avg, Median, Std. Dev: {round(np.average(r_2_vals), 5)}, {round(np.median(r_2_vals), 5)}, {round(np.std(r_2_vals), 5)}")
        print(f"Cosine Sim Avg, Median, Std. Dev: {round(np.average(cosim), 5)}, {round(np.median(cosim), 5)}, {round(np.std(cosim), 5)}")

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Ground Truth")
    ax.set_ylabel("Prediction")

    fig.tight_layout()
    fig.savefig(save_path + f"PredictionsGraph.png", dpi=dpi)
