import numpy as np
import itertools
from scipy.special import comb as nCr
import OptCC
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Code for plotting figure 2.

m = 3
k=2*m-1
n = 2*m # number of worker nodes

def RunPk(alphas,betas,ka):
    '''Generates loss and epsilon for N_succ=ka for matdot and chebyshev codes'''
    E = np.zeros((n, m ** 2))
    for i in range(E.shape[0]):
        E[i] = np.kron(alphas[i], betas[i])
    p = nCr(n, n - ka, exact=True)
    D_nz_locs = np.ones((p, n), bool)
    D_indices = np.arange(n)
    l = 0

    for comb in itertools.combinations(range(n), n - ka):
        for i in range(len(comb)):
            D_nz_locs[l, comb[i]] = False
        l = l + 1

    flat_eye = np.eye(m).flatten()
    D = np.zeros((p, n))
    condmax = 0
    for err_pat_index in range(D.shape[0]):
        V = E[D_indices[D_nz_locs[err_pat_index]], :]
        condv = np.linalg.cond(V)
        if (condv > condmax):
            condmax = condv
        D[err_pat_index, D_nz_locs[err_pat_index]] = flat_eye @ np.linalg.pinv(V)

    Desired_Matrix = np.tile(flat_eye, (p, 1))

    np.random.seed(2)
    A=np.random.randn(100,100)
    A=A/np.linalg.norm(A)
    np.random.seed(3)
    B=np.random.randn(100,100)
    B=B/np.linalg.norm(B)
    ind=np.argmax(np.linalg.norm(Desired_Matrix - D @ E, axis=1) ** 2)

    a1=OptCC.AlternateMinMultiply(A, B, D, alphas, betas, m, n, ind)
    a2=A@B
    epsilon=np.max(np.abs(a1 - a2))
    loss =  np.linalg.norm(Desired_Matrix - D @ E) ** 2

    res = np.array([loss,epsilon])
    return res

# Chebyshev
alphas = np.zeros((n,m))
betas = np.zeros((n,m))

rho = [np.cos(np.pi*(2*l-1)/(2*n)) for l in range(1, n+1)]
for i in range(m):
    idx = np.eye(m)[i]
    cheb = np.polynomial.chebyshev.Chebyshev(idx)
    coef = np.flip(np.polynomial.chebyshev.cheb2poly(cheb.coef))
    alphas[:,i] = np.polyval(coef, rho)
    betas[:,i] = alphas[:,i]

alphas[:,0], betas[:,0] = alphas[:,0]/np.sqrt(2),  betas[:,0]/np.sqrt(2)

chebres = np.zeros((2,n-m+1))
for ka in range(m,n+1):
    chebres[:,ka-m]=RunPk(alphas,betas,ka)

# Matdot
# np.random.seed(1)
var_vals=np.array([np.cos(np.pi * (2 * l - 1) / (2 * n)) for l in range(1, n + 1)])/1
V_Full = np.vander(var_vals,N=k,increasing=True).astype('float64')
alphas = np.vander(var_vals,N=m,increasing=True).astype('float64')
betas = np.vander(var_vals,N=m).astype('float64')

vanderres = np.zeros((2,n-m+1))
for ka in range(m,n+1):
    vanderres[:,ka-m]=RunPk(alphas,betas,ka)

# Approx Matdot
# np.random.seed(1)
# var_vals=(np.random.rand(n)-0.5)/70000
var_vals=np.array([np.cos(np.pi * (2 * l - 1) / (2 * n)) for l in range(1, n + 1)])/70000
V_Full = np.vander(var_vals,N=k,increasing=True).astype('float64')
alphas = np.vander(var_vals,N=m,increasing=True).astype('float64')
betas = np.vander(var_vals,N=m).astype('float64')

avanderres = np.zeros((2,n-m+1))
for ka in range(m,n+1):
    avanderres[:,ka-m]=RunPk(alphas,betas,ka)

# OptCode
# losses, epsilons = OptCC.RunPk(m=3, k=3, n=6, seed=7, num_steps=int(1e6))
with open('Data/arxiv/Opt_3_3_6_7.npy', 'rb') as f:
    losses=np.load(f)
    epsilons=np.load(f)
optres = np.vstack((losses,epsilons))

SMALL_SIZE = 10
MEDIUM_SIZE = 17
BIGGER_SIZE = 26

rcParams.update({'figure.autolayout': True})
rcParams.update({
    "text.usetex": True})
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all',gridspec_kw={'height_ratios': [4.5, 1]},figsize=(10, 7))
fig.subplots_adjust(hspace=0.02)  # adjust space between axes

# Fig 2a
ks=range(m,n+1)
ax1.plot(ks,chebres[0,:],label='chebyshev')
ax1.plot(ks,vanderres[0,:],label='MatDot')
ax1.plot(ks,avanderres[0,:],label='Approx MatDot')
ax1.plot(ks,optres[0,:],label='Opt Code(k=3)')
ax2.plot(ks,chebres[0,:],label='chebyshev')
ax2.plot(ks,vanderres[0,:],label='MatDot')
ax2.plot(ks,avanderres[0,:],label='Approx MatDot')
ax2.plot(ks,optres[0,:],label='Opt Code(k=3)')
ax1.set_yscale('log')
ax2.set_yscale('log')
plt.xlabel(r'$N_{succ}\rightarrow$')
ax1.set_ylabel(r'Loss $\rightarrow$')
# ax1.set_title('Loss vs $N_{succ}$ for $m=3,P=6$')
# ax1.legend(loc='upper left',bbox_to_anchor=(1.05, 1),fontsize='large')
ax1.legend(fontsize='x-large',framealpha=0.2)
# zoom-in / limit the view to different portions of the data
ax1.set_ylim(1e-13, 2e2)  # outliers only
ax2.set_ylim(1e-32, 1e-22)  # most of the data

# hide the spines between ax and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
# plt.savefig('../Data/Pics/jres1.png')
plt.show()

# Fig 2b
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all',gridspec_kw={'height_ratios': [3, 1]},figsize=(10, 7))
fig.subplots_adjust(hspace=0.03)  # adjust space between axes


ks=range(m,n+1)
ax1.plot(ks,chebres[1,:],label='chebyshev')
ax1.plot(ks,vanderres[1,:],label='MatDot')
ax1.plot(ks,avanderres[1,:],label='Approx MatDot')
ax1.plot(ks,optres[1,:],label='Opt Code(k=3)')
ax2.plot(ks,chebres[1,:],label='chebyshev')
ax2.plot(ks,vanderres[1,:],label='MatDot')
ax2.plot(ks,avanderres[1,:],label='Approx MatDot')
ax2.plot(ks,optres[1,:],label='Opt Code(k=3)')
ax1.set_yscale('log')
ax2.set_yscale('log')
plt.xlabel(r'$N_{succ}\rightarrow$')
ax1.set_ylabel(r'$\epsilon$ $\rightarrow$')
# ax1.set_title('$\epsilon$ vs $N_{succ}$ for $m=3,P=6$')
# ax1.legend(loc='upper left',bbox_to_anchor=(1.05, 1),fontsize='large')
ax1.legend(fontsize='x-large')

# zoom-in / limit the view to different portions of the data
ax1.set_ylim(1e-9, 1e-1)  # outliers only
ax2.set_ylim(1e-19, 1e-13)  # most of the data

# hide the spines between ax and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
# plt.savefig('../Data/Pics/jres2.png')
plt.show()

# Fig 2c
from MatDotCheb import getCode as mdc
A=np.random.randn(100,100)
A=A/np.linalg.norm(A)
B=np.random.randn(100,100)
B=B/np.linalg.norm(B)
epsilons=[]
losses=[]
Pks=[]
conds=[]
# set_trace()
divs=range(1,500000,1000)
for div in divs:
    epsilonk=np.zeros(n-m+1)
    lossesi=np.zeros(n-m+1)
    for k in range(m,n+1):
        alphas,betas,D,lossesk,Pk = mdc(m,k,n-k,coding='MatDot',seed=1,div=div)
        ind = np.argmax(lossesk)
        m1=OptCC.AlternateMinMultiply(A, B, D, alphas, betas, m, n, ind)
        m2=A@B
        epsilonk[k-m]=np.max(np.abs(m1 - m2))
        lossesi[k-m]=lossesk[ind]
    epsilons.append(epsilonk)
    losses.append(lossesi)
    Pks.append(Pk)
losses=np.array(losses)
epsilons=np.array(epsilons)
plt.figure(figsize=(9,7))
[plt.semilogy(divs,epsilons[:,i],label=str(i+3)) for i in range(losses.shape[1])]
plt.xlabel(r'div $\rightarrow$')
plt.ylabel(r'$\epsilon$ $\rightarrow$')
plt.legend(title='$N_{succ}$',fontsize='large',title_fontsize='xx-large')
# plt.savefig('../Data/Pics/jres3.png')
plt.show()

# Fig 2d
A=np.random.randn(100,100)
A=A/np.linalg.norm(A)
B=np.random.randn(100,100)
B=B/np.linalg.norm(B)
epsilons=[]
losses=[]
Pks=[]
conds=[]
# set_trace()
divs=[1,400,1000,5000,10000,70000]
for div in divs:
    epsilonk=np.zeros(n-m+1)
    lossesi=np.zeros(n-m+1)
    for k in range(m,n+1):
        alphas,betas,D,lossesk,Pk = mdc(m,k,n-k,coding='MatDot',seed=1,div=div)
        ind = np.argmax(lossesk)
        m1=OptCC.AlternateMinMultiply(A, B, D, alphas, betas, m, n, ind)
        m2=A@B
        epsilonk[k-m]=np.max(np.abs(m1 - m2))
        lossesi[k-m]=lossesk[ind]
    epsilons.append(epsilonk)
    losses.append(lossesi)
    Pks.append(Pk)
losses=np.array(losses)
epsilons=np.array(epsilons)

plt.figure(figsize=(9,7))
[plt.semilogy(range(m,n+1),epsilons[i],label=str(divs[i])) for i in range(len(losses))]
plt.xlabel(r'$N_{succ}$ $\rightarrow$')
plt.ylabel(r'$\epsilon$ $\rightarrow$')
# plt.legend(title="div",fontsize='large',loc='upper left',bbox_to_anchor=(1.05, 1))
plt.legend(title="div",fontsize='large',title_fontsize='xx-large')
# plt.savefig('../Data/Pics/jres4.png')
plt.show()