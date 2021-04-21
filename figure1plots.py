import re
from pathlib import Path
import numpy as np
import  matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import pandas as pd


# Figure 1 plotting codes, searches through OptCode output files to find losses and plot them.

outputs = {}
seeds = {}
vals = []
outputs2 = {}
outputs3 = {'m':np.array([]),'r':np.array([]),'loss':np.array([])}
# outputs3 = {'m':np.array([]),'loss':np.array([])}
for path in Path('../Data/Logs/arxiv/').rglob('*.txt'):
    trial_part = path.parts[3]
    mkr_match = re.search('m(\d+)_k(\d+)_r(\d+).txt', trial_part)
    if mkr_match is None:
        # print('Skipping '+trial_part)
        continue
    m = int(mkr_match.groups()[0])
    k = int(mkr_match.groups()[1])
    r = int(mkr_match.groups()[2])

    textfile = open(path, 'r')
    filetext = textfile.read()
    textfile.close()
    pat = re.compile(r"Iteration - (\d+)\nseed - .*\nSkipping err pats \[]\nloss - (\d+.?\d*(?:[Ee]-?\d+)?)")
    matches = re.findall(pat, filetext)

    arr = np.array(matches).astype(np.float)
    min_ind_trial = np.argmin(arr[:, 1])
    min_iter_trial = arr[min_ind_trial,0]
    min_val_trial = arr[min_ind_trial,1]
    strk = '2m-'+str(2*m-k)

    if (outputs.get(r) == None):
        outputs[r] = {}
        seeds[r] = {}
        outputs2[r] = {}
    if (outputs[r].get(strk) == None):
        outputs[r][strk] = {}
        seeds[r][strk] = {}
        outputs2[r][strk] = {}
    outputs[r][strk][m] = min_val_trial
    seeds[r][strk][m] = min_iter_trial
    outputs2[r][strk][m] = arr[:,1]

    if (strk=='2m-2'):
        lenl = arr[:,1].size
        outputs3['m'] = np.hstack((outputs3['m'],np.ones((lenl))*m))
        outputs3['r'] = np.hstack((outputs3['r'], np.ones((lenl)) * r))
        outputs3['loss'] = np.hstack((outputs3['loss'], arr[:,1]))

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

rcParams.update({'figure.autolayout': True})
rcParams.update({
    "text.usetex": True})
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Fig 1a
plt.figure(figsize=(10,7))
plt.plot(outputs[1]['2m-2'].keys(),outputs[1]['2m-2'].values(),label='P=k+1')
plt.plot(outputs[2]['2m-2'].keys(),outputs[2]['2m-2'].values(),label='P=k+2')
plt.plot(outputs[3]['2m-2'].keys(),outputs[3]['2m-2'].values(),label='P=k+3')
plt.yscale('log')
plt.xlabel(r'm$\rightarrow$')
plt.ylabel(r'Loss $\rightarrow$')
# plt.title('Loss vs m for k=2m-2')
# plt.legend(loc='upper left',bbox_to_anchor=(1.05, 1))
plt.legend(loc='upper left')
plt.savefig('../Data/Pics/res1.png')
plt.show()

klist = sorted(outputs[1].keys(), key=lambda x: x[-1])
klist = klist[::-1]
# klist = klist[:0:-1]

# Fig 1b
plt.figure(figsize=(10,7))
# plt.plot(klist,[outputs[1][i].get(2) for i in klist],label='m=2')
plt.plot(klist,[outputs[1][i].get(3) for i in klist],label='m=3')
plt.plot(klist,[outputs[1][i].get(4) for i in klist],label='m=4')
plt.plot(klist,[outputs[1][i].get(5) for i in klist],label='m=5')
plt.plot(klist,[outputs[1][i].get(6) for i in klist],label='m=6')
# plt.plot(klist,[outputs[1][i].get(7) for i in klist],label='m=7')
plt.yscale('log')
plt.xlabel(r'k$\rightarrow$')
plt.ylabel(r'Loss $\rightarrow$')
# plt.title('Loss vs k for P=k+1')
plt.legend(loc='upper left',bbox_to_anchor=(1.05, 1))
plt.savefig('../Data/Pics/res4.png')
plt.show()

# Fig 1c
plt.figure(figsize=(10,7))
df=pd.DataFrame(outputs3)
grid=sns.lineplot(data=df,x='m',y='loss',hue='r')
grid.set(yscale='log')
grid.legend(['P=k+1', 'P=k+2','P=k+3'])
plt.savefig('../Data/Pics/res7.png')
plt.show()
# ax=plt.axes()
# ax.yaxis.set_label_position("right")