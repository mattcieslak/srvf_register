from __future__ import division
import numpy as np
eps = np.finfo(np.float64).eps
from sklearn.metrics.pairwise import pairwise_distances_argmin
from scipy.integrate import simps
def invertGamma(gam):

    N = len(gam)
    x = np.arange(N,dtype=np.float)/(N-1)
    gamI = np.interp(x, gam, x)

    if np.isnan(gamI[-1]):
        gamI[-1] = 1
    else:
        gamI = gamI / gamI[-1]

    return gamI

def SqrtMeanInverse(gam, maxiter=20):
 
    n,T  = gam.shape
    dT = 1./(T-1)
    psi = np.zeros((n,T-1))
    
    for i in range(n):
        psi[i] = np.sqrt(np.diff(gam[i])/dT+eps)

    ## Find direction
    mnpsi = psi.mean(0)
    closest_match = pairwise_distances_argmin(mnpsi[None,:], psi)[0]

    mu = psi[closest_match]
    t = 1
    lvm = np.zeros(maxiter)
    vec = np.zeros((n,T-1))
    for iternum in range(maxiter):
        for i in range(n):
            v = psi[i] - mu
            dot1 = simps(mu*psi[i],np.linspace(0,1,T-1))
            if dot1 > 1:
                dot_limited = 1
            elif dot1 < -1:
                dot_limited = -1
            else:
                dot_limited = dot1

            shooting_len = np.arccos(dot_limited)
            if shooting_len > 0.0001:
                vec[i,:] = (shooting_len/np.sin(shooting_len))*(psi[i,:] - np.cos(shooting_len)*mu)
            else:
                vec[i,:] = np.zeros((1,T-1))
            
        vm = vec.mean(0)
        lvm[iternum] = np.sqrt(np.sum(vm*vm)*dT)
        if lvm[iternum] < 1e-6 or iternum >= maxiter:
            break
        mu = np.cos(t*lvm[iternum])*mu + (np.sin(t*lvm[iternum])/lvm[iternum])*vm

    gam_mu = np.concatenate([[0], np.cumsum(mu*mu)]) / T
    gam_mu = (gam_mu-min(gam_mu))/(max(gam_mu)-min(gam_mu))
    return invertGamma(gam_mu)

