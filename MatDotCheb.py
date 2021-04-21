import numpy as np
import itertools
from scipy.special import comb as nCr
import pathlib


def getCode(m,N_succ,r,coding='MatDot',seed=None,write=False,div=1,DataDir=None):
    '''
        Generates MatDot and Chebyshev codes and optionally saves to file.

        Input:
            m: number of splits in matrices A or B.
            N_succ: number of nodes that do not straggle
            r: number of failures (n - N_succ)
            coding: 'MatDot or 'Chebyshev
            seed: currently unused
            write: writes to false
            div: evaluation points are divided by this value
            DataDir: The directory where the file is saved.

        Returns:
            alphas: Contains the encoding functions for A
            betas: Contains the encoding functions for B
            D: Contains the decoding functions
            losses: Contains losses for each failure pattern
            D_nz_locs: Failure pattern matrix, which represents all combinations of failure patterns.

        '''
    if seed != None:
        np.random.seed(seed)
    k = 2 * m - 1
    n = N_succ + r
    if coding == 'MatDot':
        # var_vals=(np.random.rand(n)-0.5)/div
        var_vals = np.array([np.cos(np.pi * (2 * l - 1) / (2 * n)) for l in range(1, n + 1)])/div
        V_Full = np.vander(var_vals,N=k,increasing=True).astype('float64')
        alphas = np.vander(var_vals,N=m,increasing=True).astype('float64')
        betas = np.vander(var_vals,N=m).astype('float64')
    elif coding == 'Chebyshev':
        rho = np.array([np.cos(np.pi * (2 * l - 1) / (2 * n)) for l in range(1, n + 1)])/div
        alphas = np.zeros((n, m))
        betas = np.zeros((n, m))

        for i in range(m):
            idx = np.eye(m)[i]
            cheb = np.polynomial.chebyshev.Chebyshev(idx)
            coef = np.flip(np.polynomial.chebyshev.cheb2poly(cheb.coef))
            alphas[:, i] = np.polyval(coef, rho)
            betas[:, i] = alphas[:, i]

        alphas[:, 0], betas[:, 0] = alphas[:, 0] / np.sqrt(2), betas[:, 0] / np.sqrt(2)
    else:
        return

    E = np.zeros((n,m**2))
    for i in range(E.shape[0]):
        E[i] = np.kron(alphas[i],betas[i])


    p = nCr(n,n-N_succ,exact=True)
    D_nz_locs = np.ones((p, n), bool)
    D_indices = np.arange(n)
    l = 0

    for comb in itertools.combinations(range(n), n-N_succ):
        for i in range(len(comb)):
            D_nz_locs[l, comb[i]] = False
        l = l+1

    D = np.zeros((p,n))
    flat_eye = np.eye(m).flatten()
    # condmax = 0
    for err_pat_index in range(D.shape[0]):
        if coding == 'MatDot':
            V = V_Full[D_indices[D_nz_locs[err_pat_index]],:]
            invV = np.linalg.pinv(V)
            D[err_pat_index,D_nz_locs[err_pat_index]] = invV[m-1]
        elif coding == 'Chebyshev':
            V = E[D_indices[D_nz_locs[err_pat_index]], :]
            D[err_pat_index, D_nz_locs[err_pat_index]] = flat_eye @ np.linalg.pinv(V)

        # condv = np.linalg.cond(V)
        # if (condv > condmax):
        #     condmax=condv
    Desired_matrix = np.tile(flat_eye,(p,1))
    losses = np.linalg.norm(Desired_matrix-D@E,axis=1)**2
    loss = np.sum(losses)
    if __name__ == '__main__':
        print(loss)

    if write:
        filename = DataDir+coding+'_'+str(m)+'_'+str(N_succ)+'_'+str(r)\
                   +('_'+str(seed) if seed!=None and coding == 'MatDot' else '')\
                   +('_'+str(div) if coding == 'MatDot' else '')+'.npy'
        with open(filename, 'wb') as f:
            np.save(f, alphas.T)
            np.save(f, betas.T)
            np.save(f, D.T)
            np.save(f, loss)
    return alphas,betas,D,losses,D_nz_locs


if __name__ == '__main__':
    m = 5
    N_succ = 5
    r = 2
    # MatDot, Chebyshev
    DataDir = 'Data/arxiv/'
    pathlib.Path(DataDir).mkdir(parents=True, exist_ok=True)
    getCode(m, N_succ, r, coding='MatDot', write=True, div=1,DataDir=DataDir)