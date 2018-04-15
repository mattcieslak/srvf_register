from __future__ import print_function, division
import numpy as np
eps = np.finfo(np.float64).eps
from sklearn.metrics.pairwise import pairwise_distances_argmin
from .dynamic_programming_q2 import dp, parallel_dp
from .SqrtMeanInverse import SqrtMeanInverse
from scipy.integrate import cumtrapz, simps, trapz

has_matplotlib = True
try:
    import matplotlib.pyplot as plt
except ImportError:
    has_matplotlib = False


class RegistrationProblem(object):
    def __init__(self, original_functions, sample_times=None,
                 max_karcher_iterations=15, lambda_value=0.0, update_min=1e-2):
        """
        Encapsulates the data of a group time series registration problem
        """
        # Store input data
        self.original_functions = original_functions
        self.n_samples, self.n_functions = self.original_functions.shape
        self.max_karcher_iterations = max_karcher_iterations
        self.lambda_value = lambda_value
        self.update_min = update_min

        if sample_times is None:
            print("No sample times provided, assuming dt=1")
            self.sample_times = np.arange(self.self.n_samples, dtype=np.float)
        else:
            self.sample_times = sample_times.squeeze()

        # Check inputs
        if len(sample_times.shape) > 1 or \
                    sample_times.shape[0] != self.original_functions.shape[0]:
            raise ValueError(
                "Mismatch between sample_times and original_functions")

        self.binsize = np.diff(self.sample_times).mean()
        self.registered = False



    def _compose_temp(self, gam):
        return (self.sample_times[-1]-self.sample_times[0]) * gam \
                    + self.sample_times[0]


    def plot_registration(self):
        if not has_matplotlib: return
        fig, axes = plt.subplots(nrows=2, ncols=3)
        for func in self.original_functions.T:
            axes[0,0].plot(self.sample_times, func, '-')
        axes[0,0].set_title('Original data')
        gam_range = np.arange(self.n_samples,dtype=np.float)/(self.n_samples-1)
        for _gam in self.mean_to_orig_warps.T:
            axes[0,1].plot(gam_range, _gam, "-")
        axes[0,1].set_title('Warping functions')

        for i in range(self.registered_functions.shape[1]):
            axes[0,2].plot(self.sample_times, self.registered_functions[:,i], "-")

        axes[0,2].set_title(r'Warped data, $\lambda$ = %.3f' % self.lambda_value )

        axes[1,0].plot(self.sample_times, self.unregistered_function_mean, 'b-')
        axes[1,0].plot(self.sample_times, self.unregistered_function_mean+self.unregistered_function_std, 'r-')
        axes[1,0].plot(self.sample_times, self.unregistered_function_mean-self.unregistered_function_std, 'g-')
        axes[1,0].set_title(r'Original data: Mean $\pm$ STD')


        axes[1,1].plot(self.sample_times, self.registered_function_mean, 'b-')
        axes[1,1].plot(self.sample_times, self.registered_function_mean+self.unregistered_function_std, 'r-')
        axes[1,1].plot(self.sample_times, self.registered_function_mean-self.unregistered_function_std, 'g-')
        axes[1,1].set_title(r'Warped data, $\lambda$ = %.3f: Mean $\pm$ STD'%self.lambda_value)

        axes[1,2].plot(self.sample_times, self.function_karcher_mean, 'g-')
        axes[1,2].set_title(r'$f_{mean}$')

        plt.show()





    def run_registration(self):

        # Compute the srvf of each input function
        fy, fx = np.gradient(self.original_functions,self.binsize,self.binsize)
        self.srvf_functions = fy / np.sqrt(np.abs(fy) + eps)

        # Part 1: Set the initial target srvf. Register everything to this
        # =================================================================

        # Find the srvf closest to the mean. Set as initial srvf target
        mnq = self.srvf_functions.mean(1)
        closest_match = pairwise_distances_argmin(mnq[None,:],
                                                  self.srvf_functions.T)[0]
        mean_srvf = self.srvf_functions[:,closest_match]
        mean_function = self.original_functions[:,closest_match]
        normed_mean = np.ascontiguousarray(
                                    mean_srvf / np.linalg.norm(mean_srvf))
        # Normalize the srvfs
        self.normed_srvfs = np.row_stack(
            [srvf_func / np.linalg.norm(srvf_func) for srvf_func \
                                                in self.srvf_functions.T])

        # Register all inputs to the initial mean, store warps in gam
        gam = np.zeros((self.n_functions,self.srvf_functions.shape[0]),
                       dtype=np.float)
        for k in range(self.n_functions):
            q_c = self.normed_srvfs[k]
            G,T = dp(normed_mean, self.sample_times, q_c, self.sample_times,
                     self.sample_times, self.sample_times, self.lambda_value)
            gam0 = np.interp(self.sample_times, T, G)
            gam[k] = (gam0-gam0[0])/(gam0[-1] - gam0[0])  # change scale

        # Apply the inverse to the initial means
        gamI = SqrtMeanInverse(gam)
        gamI_dev = np.gradient(gamI, 1./(self.n_samples-1))
        mean_function = np.interp( self._compose_temp(gamI), self.sample_times,
                                   mean_function)
        # Recalculate mean srvf based on the mean function
        mean_srvf = np.gradient(mean_function, self.binsize) / \
                    np.sqrt(
                        np.abs(np.gradient(mean_function, self.binsize)) + eps)

        # Part 2: Register all srvfs to, then update the mean until convergence
        # =====================================================================
        # Compute Mean
        print('Computing Karcher mean of %d '
                            'functions in SRVF space...' % self.n_functions)
        self.distances = np.zeros(
                                self.max_karcher_iterations+2, dtype=np.float)
        self.distances[0] = np.inf
        self.update_energy = np.zeros(
                                  self.max_karcher_iterations+2, dtype=np.float)

        # Change self.srvf_mean_over_iterations to hold means over iterations
        self.srvf_mean_over_iterations = np.column_stack(
                                    [mean_srvf]*(self.max_karcher_iterations+2))
        self.srvf_mean_over_iterations[:,1:] = 0

        # These are dynamically grown in matlab. preallocate here
        fcollect = np.zeros(
            (self.original_functions.shape[0], self.original_functions.shape[1],
            self.max_karcher_iterations+2),dtype=np.float)

        qcollect = np.zeros(
            (self.srvf_functions.shape[0], self.srvf_functions.shape[1],
             self.max_karcher_iterations+2),dtype=np.float)
        fcollect[:,:,0] = self.original_functions
        qcollect[:,:,0] = self.srvf_functions
        self.update_energy = np.zeros(
                            self.max_karcher_iterations+1, dtype=np.float)

        # Run iterations for calculating Karcher mean
        for r in range(self.max_karcher_iterations):
            print('updating step: r=%d' % r)

            if r == self.max_karcher_iterations:
                print('maximal number of iterations is reached.')

            f_temp = np.zeros_like(self.original_functions)
            q_temp = np.zeros_like(self.srvf_functions)
            # use DP to find the optimal warping for each function to the mean
            gam = np.zeros_like(gam)
            gam_dev = np.zeros((self.n_functions,self.srvf_functions.shape[0]))

            # Get mean from last iteration
            normed_mean = np.ascontiguousarray(
                self.srvf_mean_over_iterations[:,r] / \
                np.linalg.norm(self.srvf_mean_over_iterations[:,r]))
            # Register normed srvfs to last iteration's mean
            for k in range(self.n_functions):
                q_c = self.normed_srvfs[k]
                G,T = dp(normed_mean, self.sample_times, q_c, self.sample_times,
                         self.sample_times, self.sample_times, self.lambda_value)
                gam0 = np.interp(self.sample_times, T, G)
                gam[k,:] = (gam0-gam0[0])/(gam0[-1] - gam0[0]) # rescale
                gam_dev[k,:] = np.gradient(gam[k,:], 1/(self.n_samples-1))
                f_temp[:,k] = np.interp(self._compose_temp(gam[k,:]),
                                    self.sample_times, self.original_functions[:,k])
                q_temp[:,k] = np.gradient(f_temp[:,k], self.binsize) / \
                    np.sqrt(np.abs(np.gradient(f_temp[:,k], self.binsize))+eps)

            qcollect[:,:,r+1] = q_temp
            fcollect[:,:,r+1] = f_temp

            # Update distances with the sum of distances across all inputs
            # for this iteration
            self.distances[r+1] = np.sum(simps(
                (self.srvf_mean_over_iterations[:,r].reshape(-1,1) \
                    - qcollect[:,:,r+1])**2, self.sample_times, axis=0)) + \
                self.lambda_value * np.sum(
                    simps((1-np.sqrt(gam_dev.T))**2,self.sample_times,axis=0))

            # Minimization Step
            # compute the mean of the matched function
            self.srvf_mean_over_iterations[:,r+1] = np.mean(qcollect[:,:,r+1], 1)

            self.update_energy[r] = \
                     np.linalg.norm(self.srvf_mean_over_iterations[:,r+1] \
                                   - self.srvf_mean_over_iterations[:,r]) \
                    / np.linalg.norm(self.srvf_mean_over_iterations[:,r])

            # Check if we have converged or hit the max iterations
            if self.update_energy[r] < self.update_min or \
               r >= self.max_karcher_iterations:
                break

        # Part 3: Register everything to the Karcher mean
        # ===============================================
        r = r+1
        self.srvf_karcher_mean = self.srvf_mean_over_iterations[:,r]
        self.normalized_srvf_karcher_mean = self.srvf_karcher_mean \
                                  / np.linalg.norm(self.srvf_karcher_mean)

        for k in range(self.n_functions):
            q_c = self.normed_srvfs[k]
            G, T = dp(self.normalized_srvf_karcher_mean, self.sample_times, q_c,
                      self.sample_times, self.sample_times, self.sample_times,
                      self.lambda_value)
            gam0 = np.interp(self.sample_times, T, G)
            gam[k,:] = (gam0-gam0[0])/(gam0[-1]-gam0[0])  # change scale
            gam_dev[k,:] = np.gradient(gam[k,:], 1./(self.n_samples-1.))

        gamI = SqrtMeanInverse(gam)
        gamI_dev = np.gradient(gamI, 1./(self.n_samples-1.))
        gamI_range = (self.sample_times[-1]-self.sample_times[0]) \
                     * gamI + self.sample_times[0]

        self.srvf_mean_over_iterations[:,r+1] = np.interp(
            gamI_range, self.sample_times, self.srvf_mean_over_iterations[:,r]
            ) * np.sqrt(gamI_dev)

        for k in range(self.n_functions):
            qcollect[:,k,r+1] = np.interp(gamI_range, self.sample_times,
                                          qcollect[:,k,r]) * np.sqrt(gamI_dev)
            fcollect[:,k,r+1] = np.interp(gamI_range, self.sample_times,
                                          fcollect[:,k,r])
            gam[k,:] = np.interp(gamI_range, self.sample_times, gam[k,:])

        # Aligned data & stats
        self.registered_functions = fcollect[:,:,r+1].copy()
        self.registered_srvfs = qcollect[:,:,r+1].copy()
        self.unregistered_function_mean = self.original_functions.mean(1)
        self.unregistered_function_std = self.original_functions.std(1)

        self.registered_function_mean = self.registered_functions.mean(1)
        self.registered_function_std = self.registered_functions.std(1)

        self.srvf_karcher_mean = self.srvf_mean_over_iterations[:,r+1]
        self.function_karcher_mean = np.concatenate(
            [ [self.original_functions[0].mean()],
             cumtrapz(self.srvf_karcher_mean* np.abs(self.srvf_karcher_mean),
                      self.sample_times)
             ])

        fgam = np.zeros((self.n_samples,self.n_functions),dtype=np.float)
        inv_warps = np.zeros((self.n_samples,self.n_functions),dtype=np.float)
        for ii in range(self.n_functions):
            fgam[:,ii] = np.interp(
                (self.sample_times[-1]-self.sample_times[0]) \
                            * gam[ii] + self.sample_times[0],
                self.sample_times, self.function_karcher_mean)

            inv_warps[:,ii] = np.interp(
                self.sample_times,
                (self.sample_times[-1]-self.sample_times[0]) \
                            * gam[ii] + self.sample_times[0],
                 self.sample_times)

        self.var_fgam = np.var(fgam,1)

        self.orig_var = trapz(self.unregistered_function_std**2,
                              self.sample_times)
        self.amplitude_var = trapz(self.unregistered_function_std**2,
                                   self.sample_times)
        self.phase_var = trapz(self.var_fgam, self.sample_times)

        fy, fx = np.gradient(gam, self.binsize, self.binsize)
        self.psi = np.sqrt(fy+eps)


        self.mean_to_orig_warps = gam.T
        self.orig_to_mean_warps = \
            (inv_warps - self.sample_times[0]) / \
            (self.sample_times[-1] - self.sample_times[0])
        self.registered = True



    def run_registration_parallel(self):
        """
        Run with OpenMP multi-threading
        """

        # Compute the srvf of each input function
        fy, fx = np.gradient(self.original_functions,self.binsize,self.binsize)
        self.srvf_functions = fy / np.sqrt(np.abs(fy) + eps)

        # Part 1: Set the initial target srvf. Register everything to this
        # =================================================================

        # Find the srvf closest to the mean. Set as initial srvf target
        mnq = self.srvf_functions.mean(1)
        closest_match = pairwise_distances_argmin(mnq[None,:],
                                                  self.srvf_functions.T)[0]
        mean_srvf = self.srvf_functions[:,closest_match]
        mean_function = self.original_functions[:,closest_match]
        normed_mean = np.ascontiguousarray(
                                    mean_srvf / np.linalg.norm(mean_srvf))
        # Normalize the srvfs
        self.normed_srvfs = np.row_stack(
            [srvf_func / np.linalg.norm(srvf_func) for srvf_func \
                                                in self.srvf_functions.T])

        # Register all inputs to the initial mean, store warps in gam
        gam = parallel_dp(normed_mean, self.sample_times,
                          self.normed_srvfs, self.sample_times,
                          self.sample_times, self.sample_times,
                          self.lambda_value)

        # Apply the inverse to the initial means
        gamI = SqrtMeanInverse(gam)
        gamI_dev = np.gradient(gamI, 1./(self.n_samples-1))
        mean_function = np.interp( self._compose_temp(gamI), self.sample_times,
                                   mean_function)
        # Recalculate mean srvf based on the mean function
        mean_srvf = np.gradient(mean_function, self.binsize) / \
                    np.sqrt(
                        np.abs(np.gradient(mean_function, self.binsize)) + eps)

        # Part 2: Register all srvfs to, then update the mean until convergence
        # =====================================================================
        # Compute Mean
        print('Computing Karcher mean of %d '
                            'functions in SRVF space...' % self.n_functions)
        self.distances = np.zeros(
                                self.max_karcher_iterations+2, dtype=np.float)
        self.distances[0] = np.inf
        self.update_energy = np.zeros(
                                  self.max_karcher_iterations+2, dtype=np.float)

        # Change self.srvf_mean_over_iterations to hold means over iterations
        self.srvf_mean_over_iterations = np.column_stack(
                                    [mean_srvf]*(self.max_karcher_iterations+2))
        self.srvf_mean_over_iterations[:,1:] = 0

        # These are dynamically grown in matlab. preallocate here
        fcollect = np.zeros(
            (self.original_functions.shape[0], self.original_functions.shape[1],
            self.max_karcher_iterations+2),dtype=np.float)

        qcollect = np.zeros(
            (self.srvf_functions.shape[0], self.srvf_functions.shape[1],
             self.max_karcher_iterations+2),dtype=np.float)
        fcollect[:,:,0] = self.original_functions
        qcollect[:,:,0] = self.srvf_functions
        self.update_energy = np.zeros(
                            self.max_karcher_iterations+1, dtype=np.float)

        # Run iterations for calculating Karcher mean
        for r in range(self.max_karcher_iterations):
            print('updating step: r=%d' % r)

            if r == self.max_karcher_iterations:
                print('maximal number of iterations is reached.')

            f_temp = np.zeros_like(self.original_functions)
            q_temp = np.zeros_like(self.srvf_functions)
            # use DP to find the optimal warping for each function to the mean
            gam_dev = np.zeros((self.n_functions,self.srvf_functions.shape[0]))

            # Get mean from last iteration
            normed_mean = np.ascontiguousarray(
                self.srvf_mean_over_iterations[:,r] / \
                np.linalg.norm(self.srvf_mean_over_iterations[:,r]))
            # Register normed srvfs to last iteration's mean
            gam = parallel_dp(normed_mean, self.sample_times,
                              self.normed_srvfs, self.sample_times,
                              self.sample_times, self.sample_times,
                              self.lambda_value)
            for k in range(self.n_functions):
                gam_dev[k,:] = np.gradient(gam[k,:], 1/(self.n_samples-1))
                f_temp[:,k] = np.interp(self._compose_temp(gam[k,:]),
                                    self.sample_times, self.original_functions[:,k])
                q_temp[:,k] = np.gradient(f_temp[:,k], self.binsize) / \
                    np.sqrt(np.abs(np.gradient(f_temp[:,k], self.binsize))+eps)

            qcollect[:,:,r+1] = q_temp
            fcollect[:,:,r+1] = f_temp

            # Update distances with the sum of distances across all inputs
            # for this iteration
            self.distances[r+1] = np.sum(simps(
                (self.srvf_mean_over_iterations[:,r].reshape(-1,1) \
                    - qcollect[:,:,r+1])**2, self.sample_times, axis=0)) + \
                self.lambda_value * np.sum(
                    simps((1-np.sqrt(gam_dev.T))**2,self.sample_times,axis=0))

            # Minimization Step
            # compute the mean of the matched function
            self.srvf_mean_over_iterations[:,r+1] = np.mean(qcollect[:,:,r+1], 1)

            self.update_energy[r] = \
                     np.linalg.norm(self.srvf_mean_over_iterations[:,r+1] \
                                   - self.srvf_mean_over_iterations[:,r]) \
                    / np.linalg.norm(self.srvf_mean_over_iterations[:,r])

            # Check if we have converged or hit the max iterations
            if self.update_energy[r] < self.update_min or \
               r >= self.max_karcher_iterations:
                break

        # Part 3: Register everything to the Karcher mean
        # ===============================================
        r = r+1
        self.srvf_karcher_mean = self.srvf_mean_over_iterations[:,r]
        self.normalized_srvf_karcher_mean = self.srvf_karcher_mean \
                                  / np.linalg.norm(self.srvf_karcher_mean)

        gam = parallel_dp(self.normalized_srvf_karcher_mean, self.sample_times,
                          self.normed_srvfs, self.sample_times,
                          self.sample_times, self.sample_times,
                          self.lambda_value)

        for k in range(self.n_functions):
            gam_dev[k] = np.gradient(gam[k], 1./(self.n_samples-1.))

        gamI = SqrtMeanInverse(gam)
        gamI_dev = np.gradient(gamI, 1./(self.n_samples-1.))
        gamI_range = (self.sample_times[-1]-self.sample_times[0]) \
                     * gamI + self.sample_times[0]

        self.srvf_mean_over_iterations[:,r+1] = np.interp(
            gamI_range, self.sample_times, self.srvf_mean_over_iterations[:,r]
            ) * np.sqrt(gamI_dev)

        for k in range(self.n_functions):
            qcollect[:,k,r+1] = np.interp(gamI_range, self.sample_times,
                                          qcollect[:,k,r]) * np.sqrt(gamI_dev)
            fcollect[:,k,r+1] = np.interp(gamI_range, self.sample_times,
                                          fcollect[:,k,r])
            gam[k,:] = np.interp(gamI_range, self.sample_times, gam[k,:])

        # Aligned data & stats
        self.registered_functions = fcollect[:,:,r+1].copy()
        self.registered_srvfs = qcollect[:,:,r+1].copy()
        self.unregistered_function_mean = self.original_functions.mean(1)
        self.unregistered_function_std = self.original_functions.std(1)

        self.registered_function_mean = self.registered_functions.mean(1)
        self.registered_function_std = self.registered_functions.std(1)

        self.srvf_karcher_mean = self.srvf_mean_over_iterations[:,r+1]
        tmp_mean = cumtrapz(self.srvf_karcher_mean * np.abs(self.srvf_karcher_mean),
                      self.sample_times)
        ext_tmp_mean = np.concatenate([tmp_mean[0,None], tmp_mean])
        self.function_karcher_mean = ext_tmp_mean \
                                        + self.original_functions[0].mean()

        fgam = np.zeros((self.n_samples,self.n_functions),dtype=np.float)
        inv_warps = np.zeros((self.n_samples,self.n_functions),dtype=np.float)
        for ii in range(self.n_functions):
            fgam[:,ii] = np.interp(
                (self.sample_times[-1]-self.sample_times[0]) \
                            * gam[ii] + self.sample_times[0],
                self.sample_times, self.function_karcher_mean)

            inv_warps[:,ii] = np.interp(
                self.sample_times,
                (self.sample_times[-1]-self.sample_times[0]) \
                            * gam[ii] + self.sample_times[0],
                 self.sample_times)

        self.var_fgam = np.var(fgam,1)

        self.orig_var = trapz(self.unregistered_function_std**2,
                              self.sample_times)
        self.amplitude_var = trapz(self.unregistered_function_std**2,
                                   self.sample_times)
        self.phase_var = trapz(self.var_fgam, self.sample_times)

        fy, fx = np.gradient(gam, self.binsize, self.binsize)
        self.psi = np.sqrt(fx+eps)


        self.mean_to_orig_warps = gam.T
        self.orig_to_mean_warps = \
            (inv_warps - self.sample_times[0]) / \
            (self.sample_times[-1] - self.sample_times[0])
        self.registered = True
