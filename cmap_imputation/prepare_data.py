import scipy.io
import pathlib
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # Get train dataset, as referenced in Hodos et al., 2018
    dir_main = pathlib.Path(__file__).parent.absolute()
    try:
        allData = scipy.io.loadmat(os.path.join(dir_main, "data", "large.mat"))
    except FileNotFoundError:
        raise FileNotFoundError("Expected file 'large.mat' in directory ./data/\n" +
               "Please see the dataset linked in the README here " +
               "https://github.com/clinicalml/dgc_predict/tree/b8bff6d757fc757aadf39034b9972db37c6da983\n" +
               "Download the raw data and put 'large.mat' in a directory called 'data'" +
               f"in this folder, like so:\n{os.path.join(dir_main, 'data', 'large.mat')}.")

    # Label drug, gene, and cell fields in tensor 
    pertIds = [elt.astype(str)[0] for elt in allData['pertIds'].ravel()]
    geneIds = [elt.astype(str)[0] for elt in allData['geneIds'].ravel()]
    cellIds = [elt.astype(str)[0] for elt in allData['cellIds'].ravel()]
    tensor = allData['T'].astype(np.float64)

    df = pd.concat([pd.DataFrame(x) for x in tensor.transpose(2, 0, 1)],
                ignore_index=True)

    df['unit'] = [cell_id for cell_id in cellIds
                        for _ in range(len(pertIds))]

    df['intervention'] = [pert_id for _ in range(len(cellIds))
                                for pert_id in pertIds]

    # Set drug and cell indices accordingly, rename axis, drop NaNs, and save to pkl
    df.set_index(['unit', 'intervention'], inplace=True)
    df.columns = geneIds
    df.rename_axis('rid', axis=1, inplace=True)
    df.dropna(inplace=True)

    # Save the refactored tensor as a .pkl file
    savepath = os.path.join(dir_main, "data", "largeTensor.pkl")

    if os.path.exists(savepath):
        overwrite = input(f"A file already exists at path {savepath}, do you want to overwrite? (Y/N): ")

    if overwrite.lower() == 'y':
        df.to_pickle(savepath)
        print(f"File at {savepath} overwritten")
    else:
        print(f"Refactored tensor not saved")
