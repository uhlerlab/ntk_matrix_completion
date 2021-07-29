import matplotlib
matplotlib.use('Agg')

import copy

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from cli import validate_inputs


def generate_heatmap(all_metrics_dnpp, all_metrics_ntk, stat,
                     save_path, incl_stats, figsize=(10, 10), dpi=600):

    all_metrics_dnpp.sort_index(inplace=True)
    all_metrics_ntk = all_metrics_ntk.sort_index()[all_metrics_dnpp.columns]

    assert (all_metrics_dnpp.index == all_metrics_ntk.index).all(), "Indices do not match"

    df = ((all_metrics_ntk > all_metrics_dnpp).astype(int) * 2 - 1)
    ytick_space = 50

    if stat == 'cosine_similarity':
        stats = ['cosine_similarity']
    elif stat == 'r_squared':
        stats = ['r_squared']
    elif stat == 'both':
        stats = ['cosine_similarity', 'r_squared']
    else:
        raise AssertionError("Pick from the valid metrics: r_squared, cosine_similarity, both")

    for metric in stats:
        counts = df.groupby(['intervention', 'unit'])[metric].last()
        sorted_perturbations = counts.groupby('intervention').size().sort_values(ascending=True)
        sorted_celltypes = counts.groupby('unit').size().sort_values(ascending=False)
        count_matrix = np.full((len(sorted_perturbations), len(sorted_celltypes)), float("inf"))
        pert2ix = {pert: ix for ix, pert in enumerate(sorted_perturbations.index)}
        cell2ix = {cell: ix for ix, cell in enumerate(sorted_celltypes.index)}
        pert_ixs = counts.index.get_level_values('intervention').map(pert2ix)
        cell_ixs = counts.index.get_level_values('unit').map(cell2ix)
        count_matrix[pert_ixs, cell_ixs] = counts.values
        fig, ax = plt.subplots(figsize=figsize)

        if incl_stats:
            ax.set_xlabel("Cell Types")
            ax.set_ylabel("Perturbation IDs")
            ax.set_title(f"Heat Map for {metric}, NTK vs DNPP")
        ax.set_xticks(list(range(len(sorted_celltypes))))
        ax.set_yticks(list(reversed(range(len(sorted_perturbations), 0, -ytick_space))))
        ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False)
        ax.tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False, labelsize='x-small')
        ax.set_xticklabels([str(ix+1)+":"+ct for ix, ct in enumerate(sorted_celltypes.index)], ha='right', rotation=70)
        plt.tight_layout()

        cmap = copy.copy(mpl.cm.get_cmap("RdYlGn"))
        cmap.set_over("white")
        plt.imshow(count_matrix, aspect='auto', vmin=-1, vmax=1, interpolation='none', cmap=cmap)
        custom_lines = [mpl.lines.Line2D([0], [0], color='red', lw=4),
                        mpl.lines.Line2D([0], [0], color='green', lw=4)]
        ax.legend(custom_lines, ['DNPP Performs Better', 'NTK Performs Better'])

        plt.savefig(save_path + f"{metric}Metric.png", dpi=dpi)


if __name__ == '__main__':
    path_prefix, metric, all_metrics_ntk, all_metrics_dnpp = validate_inputs()
    generate_heatmap(all_metrics_dnpp, all_metrics_ntk, metric, path_prefix, True)
