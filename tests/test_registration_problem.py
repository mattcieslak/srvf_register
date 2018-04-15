from __future__ import print_function, division
from scipy.io.matlab import loadmat
import sys
sys.path.append("..")
from srvf_register import RegistrationProblem

input_data = loadmat("test_funcs.mat",squeeze_me=True)

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
lam = 0.0



rp = RegistrationProblem(f,t)
rp.run_registration()
rp.plot_registration()


from srvf_register.dynamic_programming_q2 import \
     parallel_dp
srvfs = rp.normed_srvfs
target_srvf = rp.normalized_srvf_karcher_mean
t = rp.sample_times
gam = parallel_dp(target_srvf, t, srvfs,t,t,t,0.0)

rp2 = RegistrationProblem(f,t)
rp2.run_registration_parallel()
rp2.plot_registration()
