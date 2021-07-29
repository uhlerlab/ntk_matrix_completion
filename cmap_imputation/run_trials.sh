# Prepare Data
python prepare_data.py

# Trains DNPP on Sparse Subset (Seed 149)
python dnpp/dnpp.py -b 2 -k 0 -s 149

# Trains DNPP on 10-Fold for All Data (Seed 149)
python dnpp/dnpp.py -s 149

# Trains Means on 10-Fold for All Data (Seed 149)
python means/means.py -s 149

# Train NTK with MCF7 + Cell Averages for 10-Fold (Seed 149)
python neural_tangent_kernel/ntk.py -s 149 -x only_train_cell_average

# Train NTK with Combination DNPP and MCF7 Priors (Seed 149)
python neural_tangent_kernel/dnpp_mcf7_mixed_prior.py -s 149

# Train NTK with Combination FaLRTC and MCF7 Priors (Seed 149)
python neural_tangent_kernel/falrtc_mcf7_mixed_prior.py -s 149

# Generate Comparative Graphics for DNPP vs Combination (Seed 149)
python graphical/comparative_graphics.py -d ./predictions/DNPPKFold10Folds149SeedlargeTensorTrainSourceAllMetrics.pkl -n ./predictions/MixedDNPPNTKKFold10Folds149SeedlargeTensorTrainSourceAllMetrics.pkl -s 149
