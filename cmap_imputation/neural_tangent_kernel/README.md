This folder provides scripts for running the Neural Tangent Kernel to impute CMAP data. It includes a Command-Line-Interface for interacting with the scripts. There are choices for using a few standard priors, as well as custom (user-written) and mixed priors (FaLRTC and DNPP). The mixed-prior scripts use two different priors, based on the quantity of drugs in the training set, for a given cell. See the `{dnpp/falrtc}_mcf7_mixed_prior.py` files for additional information.

The available scripts in this directory are:
- [dnpp_mcf7_mixed_prior.py](dnpp_mcf7_mixed_prior.py)
- [falrtc_mcf7_mixed_prior.py](falrtc_mcf7_mixed_prior.py)
- [ntk.py](ntk.py)

__Note:__
For the FaLRTC mixed prior, one must first use the Matlab script found [here](https://github.com/clinicalml/dgc_predict/blob/b8bff6d757fc757aadf39034b9972db37c6da983/matlab/thirdparty/visual/FaLRTC.m) for imputation, and then use those outputs as inputs to the NTK method. The linked code is used by [Hodos et al., 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5753597/pdf/nihms921438.pdf) in their analysis, though it references an earlier work by [Liu et al., 2012](https://www.cs.rochester.edu/u/jliu/paper/Ji-ICCV09.pdf).
