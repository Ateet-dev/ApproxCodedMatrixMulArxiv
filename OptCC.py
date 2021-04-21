import numpy as np
import matplotlib.pyplot as plt
import itertools
import pathlib
from tqdm import tqdm
import time

#--- Auxiliary functions for measuring performance ---#

# function for computing objective value
def objective(A,B,D,Pk):
    m,n = A.shape
    k = len(Pk[1])
    Ga = A.transpose().dot(A)
    Gb = B.transpose().dot(B)
    Gx = Ga*Gb
    c = np.diag(A.transpose().dot(B))
    
    error = 0
    for ix in range(len(Pk)):
        s = Pk[ix] # current set
        d = D[s,ix]
        d1 = d.reshape(k,1)
        error += m - 2*sum(d*c[s]) +d1.transpose().dot(Gx[s,:][:,s]).dot(d1)
    
    return error[0,0]

# function for computing error distribution
def error_distribution(A,B,D,Pk):
    m,n = A.shape
    k = len(Pk[1])
    Ga = A.transpose().dot(A)
    Gb = B.transpose().dot(B)
    Gx = Ga*Gb
    c = np.diag(A.transpose().dot(B))
    
    error = np.zeros(len(Pk))
    for ix in range(len(Pk)):
        s = Pk[ix] # current set
        d = D[s,ix]
        d1 = d.reshape(k,1)
        error[ix] = np.abs(m - 2*sum(d*c[s]) +d1.transpose().dot(Gx[s,:][:,s]).dot(d1))
    
    return error

#--- Main iteration for solving the optimization ---#
def Qopt(m,k,n,seed=0, num_steps=2000, skip_count=0,write_steps=1000,plot=False,write=False,trial=0):
    '''
    Solve Frobenius norm minimization for finding coded computing coefficients. 
    Considers m chunks distributed to n workers.
    
    Input:
        A, B: m x n matrices that are the starting point of the minimization.
        k: Recovery threshold
        num_steps: Number of optimization steps (too many steps may produce numerical instability)
        plot: Plot optimization progress and error pattern distribution
        write_steps: Every write_steps steps objective error is computed and break condition is checked.
        write: If true, every write_steps steps, in addition to computing error, current state is also written to an npy file.
        
    Returns:
        A: alpha coefficients
        B: beta coefficients
        D: n x |Pk| matrix with recovery coefficients, where Pk is the set of all subsets of [n] of size k
        Pk: considered set of subsets that index columns of D
        error: final objective error      
    
    '''

    np.random.seed(seed)  # Always keep track of seed!

    # randomly initialize coefficient matrix for optimization
    A = np.random.randn(m, n)  # alpha coeff
    B = np.random.randn(m, n)  # beta coeff
    
    # TODO: raise ValueError if A and B are not the same shapes
    
    Pk = list(itertools.combinations(range(n),k)) # set of all subsets of [n] of size k
    ncomb = len(Pk)
    Pk = [np.array(s) for s in Pk] # deal with annoying indexing errors

    skip_err_pats = np.random.choice(range(ncomb), skip_count, replace=False)
    # print('Skipping err pats ' + str(skip_err_pats))
    err_pat_inds = list(range(ncomb))
    [err_pat_inds.remove(i) for i in skip_err_pats]
    Pk = [Pk[i] for i in err_pat_inds]
    ncomb -= skip_count

    # create matrix of reconstruction coefficients
    # TODO: if n is large and k is smaller, make this sparse
    D = np.zeros((n,ncomb))

    if plot or write:
        total_writes = np.ceil(num_steps/write_steps).astype('int')+1
        error_vec = np.zeros(total_writes)
        plot_steps = np.zeros(total_writes)
        filenum = 0

    if (write):
        pathname = 'Data/OptCode/DAB' + '_' + str(m) + '_' + str(k) + '_' + str(n)+'/Trial_' + str(trial)
        pathlib.Path(pathname).mkdir(parents=True, exist_ok=True)

    # tic = time.perf_counter_ns()
    Ga = A.transpose().dot(A)

    # for t in tqdm(range(num_steps)):
    for t in range(num_steps):

        Gb = B.transpose().dot(B)
        Gx = Ga*Gb
        c = np.diag(A.transpose().dot(B))

        # Step 1: minimize for d    
        for ix in range(len(Pk)):
            s = Pk[ix] # current set
            Gs = Gx[s,:][:,s]
            cs = c[s]

            # solve
            D[s,ix] = np.linalg.solve(Gs,cs)

        z = D.sum(axis=1)
        z = z.reshape(len(z),1)
        Z = D.dot(D.transpose())

        # Step 2: minimize for A
        Btilde = Z*Gb
        A = np.linalg.solve(Btilde,z*B.transpose()).transpose()

        # Step 3: minimize for B
        Ga = A.transpose().dot(A)
        Atilde = Z*Ga
        B = np.linalg.solve(Atilde,z*A.transpose()).transpose()

        # store current objective value
        if (t != 0 and (t%write_steps == 0 or t == num_steps-1)):
            error = objective(A, B, D, Pk)
            if __name__ == '__main__':
                print(t,error)
            if (plot or write):
                error_vec[filenum] = error
                plot_steps[filenum] = t
                filenum += 1
            if (write):
                writedatafun(D,A,B,error_vec[:filenum],plot_steps[:filenum],t,filenum-1,m,k,n,pathname)
            if (error < 1e-26):
                break

            
    # toc = time.perf_counter_ns()
    # print((toc-tic)/1e9)

    # make plots
    if plot:

        # compute error across different failure patterns
        error_dist = error_distribution(A,B,D,Pk)

        # plot optimization progress
        plt.plot(plot_steps,error_vec)
        plt.yscale('log')
        plt.xlabel('Number of iterations')
        plt.ylabel('Objective value')
        plt.title('Optimization Progress for n={}, m={}, k={}'.format(n,m,k) )
        plt.grid()
        plt.show()

        # plot error distribution across failure patterns
        plt.figure()
        plt.plot((1+np.arange(len(error_dist)))/len(error_dist),np.sort(error_dist),'o--')
        plt.yscale('log')
        plt.xlim(0,1.1)
        plt.xlabel('Fraction of failure patterns')
        plt.ylabel('Approximation error')
        plt.title('Inverted Cumulative Approx. Error for n={}, m={}, k={}'.format(n,m,k))
        plt.grid()
        plt.show()
    
    error = objective(A,B,D,Pk)
    # print('Final objective error {}'.format(error,"f"))
    
    return A,B,D,Pk,error,skip_err_pats

def writedatafun(D,A,B,error_vec,plot_steps,iter,filenum,m,k,n,pathname):
    with open(pathname + '/DAB' + '_' + str(m) + '_' + str(k) + '_' + str(n) + '.txt', 'a+') as f:
        f.write('Iteration - ' + str(iter) + '\n')
        f.write('File - ' + 'DAB' + '_' + str(m) + '_' + str(k) + '_' + str(n) + '_' + str(filenum) + '.npy' + '\n')
        f.write('Error - ' + str(error_vec[-1]) + '\n')
        f.write('================================================================\n')

    with open(pathname + '/DAB' +'_'+ str(m) + '_' +str(k)+'_' + str(n) +'_'+str(filenum)+ '.npy', 'wb') as f:
        np.save(f, A)
        np.save(f, B)
        np.save(f, D)
        np.save(f, k)
        np.save(f, iter)
        np.save(f, error_vec)
        np.save(f, plot_steps)

def AlternateMinMultiply(A,B,D,alpha,beta,m,n,err_pat_ind):
    ''' Performs A@B using coded matrix multiplication.

        Input:
            err_pat_ind: Indicates which failure pattern to be used.
    '''
    if (A.shape[1]%m!=0):
        padlen = ((A.shape[1] // m) + 1) * m - A.shape[1]
        A= np.hstack((A,np.zeros((A.shape[0],padlen))))
        B = np.vstack((B, np.zeros((padlen,B.shape[1]))))
    A_partioned = np.vstack(
        [A[np.newaxis, :, i:i + A.shape[1] // m] for i in range(0, A.shape[1] - A.shape[1] // m + 1, A.shape[1] // m)])
    B_partioned = np.vstack(
        [B[np.newaxis, i:i + B.shape[0] // m, :] for i in range(0, B.shape[0] - B.shape[0] // m + 1, B.shape[0] // m)])

    if np.isnan(err_pat_ind):
        ll=[A_partioned[i] @ B_partioned[i] for i in range(m-2)]
        C_decoded = np.sum(ll, axis=0)
        return C_decoded

    # Tensordot understanding: https://stackoverflow.com/questions/41870228/understanding-tensordot
    A_encoded = [np.tensordot(alpha[i], A_partioned, axes=((0), (0))) for i in range(n)]
    B_encoded = [np.tensordot(beta[i], B_partioned, axes=((0), (0))) for i in range(n)]

    C_encoded = np.vstack([(AA @ BB)[np.newaxis] for AA, BB in zip(A_encoded, B_encoded)])

    C_decoded = np.tensordot(D[err_pat_ind], C_encoded, axes=((0), (0)))[np.newaxis]
    # C_decoded = np.vstack([np.tensordot(D[i], C_encoded, axes=((0), (0)))[np.newaxis] for i in range(p)])

    return C_decoded[0]

def getD(A,B,Pk):
    '''Gets decoding matrix D for given failure pattern matrix Pk, using in RunPk'''
    n = A.shape[1]
    ncomb = len(Pk)
    D = np.zeros((n, ncomb))

    Ga = A.transpose().dot(A)
    Gb = B.transpose().dot(B)
    Gx = Ga * Gb
    c = np.diag(A.transpose().dot(B))
    for ix in range(len(Pk)):
        s = Pk[ix]  # current set
        Gs = Gx[s, :][:, s]
        cs = c[s]

        # solve
        D[s, ix] = np.linalg.solve(Gs, cs)
    return D

def RunPk(m,k,n,seed=1234,num_steps=1000):
    '''Generates losses and epsilons from N_succ=k to N_succ=n, with OptCode designed for recovery threshold k.'''
    A, B, _, _, _,_ = Qopt(m, k, n, seed=seed, num_steps=num_steps)
    ks = range(k,n+1)
    losses = np.zeros(len(ks))
    epsilons = np.zeros(len(ks))
    for ix in range(len(ks)):
        Pk = list(itertools.combinations(range(n), ks[ix]))
        ncomb = len(Pk)
        Pk = [np.array(s) for s in Pk]
        D = getD(A, B, Pk)
        E = np.zeros((m ** 2, n))
        for i in range(n):
            E[:, i] = np.kron(A[:, i], B[:, i])
        reconst = E @ D
        flat_eye = np.eye(m).flatten()
        Desired_Matrix = np.kron(np.ones((ncomb, 1)), flat_eye).T
        losses[ix] = np.linalg.norm(Desired_Matrix[:, :ncomb] - reconst) ** 2

        np.random.seed(2)
        AA = np.random.randn(100, 100)
        AA = AA / np.linalg.norm(AA)
        np.random.seed(3)
        BB = np.random.randn(100, 100)
        BB = BB / np.linalg.norm(BB)
        ind = np.argmax(np.linalg.norm(Desired_Matrix - reconst, axis=0) ** 2)
        a1 = AlternateMinMultiply(AA, BB, D.T, A.T,B.T, m, n, ind)
        a2 = AA @ BB
        epsilons[ix]=np.max(np.abs(a1 - a2))
    return losses, epsilons

if __name__ == '__main__':
    NUM_STEPS = 100000
    WRITE_STEPS = 1000
    SEED = 787
    A, B, D, Pk, error,skip_err_pats = Qopt(m=5, k=5, n=7, seed=SEED, num_steps=int(NUM_STEPS), write_steps=int(WRITE_STEPS), plot=False, write=False)
    # losses, epsilons = RunPk(m=3, k=3, n=6, seed=7, num_steps=int(1e6))
    # with open('Opt_3_3_6_7.npy', 'wb') as f:
    #     np.save(f, losses)
    #     np.save(f, epsilons)