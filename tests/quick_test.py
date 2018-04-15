## PNS toy example
# Example of horizontal analysis using PNS

## 1. Set up
# To make this example work, please first download 
# J. S. Marron's softwares AllMatlab7Combined.zip on 
# http://www.unc.edu/depts/stat-or/miscellaneous/marron/Matlab7Software/
# and unzip it. The softwares we need are 'General' and 'Smoothing'.
# Then, add 'General','Smoothing' and 'SRVF' to your search path. 
# One way to do it is:
# Click on 'Set Path' -> 'Add with Subfolders', select 'General', 'Smoothing'
# and 'SRVF', then all these three and their subfolders will be added to
# the search path.

## 2. Make data
import numpy as np
eps = np.finfo(np.float64).eps
from scipy.stats import beta
from matplotlib import pyplot as plt
from scipy.io.matlab import loadmat

input_data = loadmat("../Example/test_funcs.mat",squeeze_me=True)

d = input_data["d"]
n = input_data["n"]
sig = input_data["sig"]
m = input_data["m"]
warpRange = input_data["warpRange"]
b = input_data["b"]

gamO = input_data["gamO"]
a = input_data["a"]
t = input_data["t"]
m = input_data["m"] 
f = input_data["f"]
MaxItr=20
show_plot=True

fig0, ax0 = plt.subplots()
for i in range(n):
    ax0.plot(t, f[:,i],'.-')

## 3. Fisher-Rao curve registration
# Include code from time_warping_oneplot
#[fn,qn,q0,fmean,mqn,gam,psi,stats]=time_warping_oneplot(f,t',mycolor);

binsize = np.mean(np.diff(t))
M, N = f.shape
f0 = f

_, fy = np.gradient(f,binsize,binsize)

q = fy/np.sqrt(np.abs(fy)+eps)

mq = q[:,0]
mf = f[:,0]

test_srvfs = loadmat("../Example/test_q1_q2.mat", squeeze_me=True)

srvf1 = test_srvfs["srvf1"]
srvf2 = test_srvfs["srvf2"]
official_G = test_srvfs["G"]
official_T = test_srvfs["T"]

from dynamic_programming_q2 import dp
G,T = dp( srvf1, t, srvf2, t, t, t, 0.0)
plt.show()
