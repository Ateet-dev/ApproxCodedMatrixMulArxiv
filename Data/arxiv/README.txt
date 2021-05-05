File name format:
m: number of splits of A or B
k: number of non straggling nodes. For matdot and chebyshev k=N_succ, for Opt code k=recovery threshold it is designed for.
r: number of failures (n-N_succ)
\gamma: Evaluation points are divided by this value.
seed: seed for optcode design.

Matdot:
MatDot_(m)_(k)_(r)_(\gamma).npy

Chebyshev:
Chebyshev_(m)_(k)_(r).npy

OptCode:
Opt_(m)_(k)_(r)_(seed).npy

Each file contains, alpha, beta, D, loss in order.