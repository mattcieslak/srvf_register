from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances_argmin
from dynamic_programming_q2 import dp
from SqrtMeanInverse import SqrtMeanInverse
from scipy.integrate import cumtrapz, simps

#function [fn,qn,q0,fmean,mqn,gam,psi,stats] = time_warping(f,t,colmap,lambda,option)

def group_warp_analysis(f,t,lam, MaxItr=30, show_plot=False):
    binsize = np.diff(t).mean()
    M, N = f.shape
    f0 = f.copy()
    def compose_temp(gam):
        return (t[-1]-t[0]) * gam + t[0]
    
    if show_plot:
        fig, axes = plt.subplots(nrows=2, ncols=3)
        for func in f0.T:
            axes[0,0].plot(t, func, '-')
        axes[0,0].set_title('Original data')
    
    # Compute the q-function of the plot
    fy, fx = np.gradient(f,binsize,binsize)
    q = fy / np.sqrt(np.abs(fy) + eps)
    
    # Set initial using the original f space
    mnq = q.mean(1)
    
    closest_match = pairwise_distances_argmin(mnq[None,:], q.T)[0]
    mq = q[:,closest_match]
    mf = f[:,closest_match]
    
    gam = np.zeros((N,q.shape[0]),dtype=np.float)
    
    normed_mean = np.ascontiguousarray(mq / np.linalg.norm(mq))
    for k in range(N):
        q_c = q[:,k] / np.linalg.norm(q[:,k])
        G,T = dp(normed_mean, t, q_c, t, t, t, lam)
        gam0 = np.interp(t, T, G)
        gam[k,:] = (gam0-gam0[0])/(gam0[-1] - gam0[0])  # slight change on scale
    
    # Apply the inverse to the mean
    gamI = SqrtMeanInverse(gam)
    gamI_dev = np.gradient(gamI, 1./(M-1))
    mf = np.interp( compose_temp(gamI), t, mf)
    mq = np.gradient(mf, binsize) / np.sqrt(np.abs(np.gradient(mf, binsize)) + eps)
    
    # Compute Mean
    print('Computing Karcher mean of %d functions in SRVF space...' % N)
    ds = np.zeros(MaxItr+1, dtype=np.float)
    ds[0] = np.inf
    qun = np.zeros(MaxItr, dtype=np.float)
    # Change mq to hold all the means over time
    mq = np.column_stack([mq]*MaxItr)
    mq[:,1:] = 0
    
    # These are dynamically grown in matlab. preallocate here
    fcollect = np.zeros((f.shape[0], f.shape[1], MaxItr),dtype=np.float)
    
    qcollect = np.zeros((q.shape[0], q.shape[1], MaxItr),dtype=np.float)
    fcollect[:,:,0] = f
    qcollect[:,:,0] = q
    qun = np.zeros(MaxItr,dtype=np.float)
    
    for r in range(MaxItr):
        print('updating step: r=%d' % r)
    
        if r == MaxItr:
            print('maximal number of iterations is reached.')
        
        f_temp = np.zeros_like(f0)
        q_temp = np.zeros_like(q)
        # use DP to find the optimal warping for each function w.r.t. the mean
        gam = np.zeros_like(gam)
        gam_dev = np.zeros((N,q.shape[0]))
    
        normed_mean = np.ascontiguousarray(mq[:,r] / np.linalg.norm(mq[:,r]))
        for k in range(N):
            q_c = q[:,k] / np.linalg.norm(q[:,k])
            G,T = dp(normed_mean, t, q_c, t, t, t, lam)
            gam0 = np.interp(t, T, G)
            gam[k,:] = (gam0-gam0[0])/(gam0[-1] - gam0[0])  # slight change on scale
            gam_dev[k,:] = np.gradient(gam[k,:], 1/(M-1))
            f_temp[:,k] = np.interp(compose_temp(gam[k,:]), t, f[:,k]);
            q_temp[:,k] = np.gradient(f_temp[:,k], binsize) / \
                          np.sqrt(np.abs(np.gradient(f_temp[:,k], binsize))+eps)
    
        qcollect[:,:,r+1] = q_temp
        fcollect[:,:,r+1] = f_temp
        
        ds[r+1] = np.sum(simps(
                (mq[:,r].reshape(-1,1) - qcollect[:,:,r+1])**2, t, axis=0)) +  \
            lam * np.sum(simps((1-np.sqrt(gam_dev.T))**2,t,axis=0))
        
        # Minimization Step
        # compute the mean of the matched function
        mq[:,r+1] = np.mean(qcollect[:,:,r+1], 1)
        
        qun[r] = np.linalg.norm(mq[:,r+1] - mq[:,r]) / np.linalg.norm(mq[:,r])
        if qun[r] < 1e-2 or r >= MaxItr:
            break
    
    
    r = r+1
    for k in range(N):
        q_c = q[:,k]
        mq_c = mq[:,r]
        G, T = dp(mq_c/np.linalg.norm(mq_c),t,q_c/np.linalg.norm(q_c),t,t,t, lam)
        gam0 = np.interp(t, T, G)
        gam[k,:] = (gam0-gam0[0])/(gam0[-1]-gam0[0])  # slight change on scale
        gam_dev[k,:] = np.gradient(gam[k,:], 1./(M-1.))
    
    
    gamI = SqrtMeanInverse(gam)
    gamI_dev = np.gradient(gamI, 1./(M-1.))
    mq[:,r+1] = np.interp((t[-1] - t[0])*gamI + t[0], t, mq[:,r]) * np.sqrt(gamI_dev)
    gamI_range = (t[-1]-t[0]) * gamI + t[0]
    for k in range(N):
        qcollect[:,k,r+1] = np.interp(gamI_range, t, qcollect[:,k,r]) * np.sqrt(gamI_dev)
        fcollect[:,k,r+1] = np.interp(gamI_range, t, fcollect[:,k,r])
        gam[k,:] = np.interp(gamI_range, t, gam[k,:])
    
    # Aligned data & stats
    fn = fcollect[:,:,r+1].copy()
    qn = qcollect[:,:,r+1].copy()
    q0 = q
    mean_f0 = f0.mean(1)
    std_f0 = f0.std(1)
    
    mean_fn = fn.mean(1)
    std_fn = fn.std(1)
    
    mqn = mq[:,r+1]
    fmean = np.concatenate([[f0[0].mean()], cumtrapz(mqn* np.abs(mqn), t)])
    
    fgam = np.zeros((M,N),dtype=np.float)
    for ii in range(N):
        fgam[:,ii] = np.interp( (t[-1]-t[0]) * gam[ii] + t[0], t, fmean)
    
    var_fgam = np.var(fgam,1)
    
    stats_orig_var = trapz(std_f0**2, t)
    stats_amp_var = trapz(std_fn**2, t)
    stats_phase_var = trapz(var_fgam, t)
    
    gam = gam.T
    fy, fx = np.gradient(gam, binsize, binsize)
    psi = np.sqrt(fy+eps)
    
    
    if show_plot:
        gam_range = np.arange(M,dtype=np.float)/(M-1)
        for _gam in gam.T:
            axes[0,1].plot(gam_range, _gam, "-")
        axes[0,1].set_title('Warping functions')
        
        for i in range(fn.shape[1]):
            axes[0,2].plot(t, fn[:,i], "-")
       
        axes[0,2].set_title(r'Warped data, $\lambda$ = %.3f' % lam )
        
        axes[1,0].plot(t, mean_f0, 'b-')
        axes[1,0].plot(t, mean_f0+std_f0, 'r-')
        axes[1,0].plot(t, mean_f0-std_f0, 'g-')
        axes[1,0].set_title(r'Original data: Mean $\pm$ STD')
        
        
        axes[1,1].plot(t, mean_fn, 'b-')
        axes[1,1].plot(t, mean_fn+std_fn, 'r-')
        axes[1,1].plot(t, mean_fn-std_fn, 'g-')
        axes[1,1].set_title(r'Warped data, $\lambda$ = %.3f: Mean $\pm$ STD'%lam)
        
        axes[1,2].plot(t, fmean, 'g-')
        axes[1,2].set_title(r'$f_{mean}$')        
        
        
        