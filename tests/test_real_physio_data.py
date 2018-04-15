from __future__ import print_function, division
import sys
sys.path.append("..")
from srvf_register import RegistrationProblem
from meap.io import load_from_disk
input_file = "/home/matt/projects/SQuaRP_17/proc/SQ204_c_s4_HE_doppler.mea.mat"
phys = load_from_disk(input_file)

import numpy as np
beat_indices = np.arange(phys.dzdt_matrix.shape[0])
resample_order = np.random.shuffle(beat_indices)
from time import time

import numpy as np
np.seterr("raise")


subset_sizes = [20, 50, 100]

subset_sizes = [ 20]
registrations = []
run_times = []
t = np.arange(phys.dzdt_matrix.shape[1],dtype=np.float)
time_min, time_max = 200, 900
t= np.arange(time_max - time_min, dtype=np.float)
for subset_size in subset_sizes:
    print("Subset of size %d" % subset_size)
    t0 = time()
    registrations.append(
        RegistrationProblem(
            phys.dzdt_matrix[beat_indices[:subset_size],time_min:time_max].T, 
            t)
    )
    registrations[-1].run_registration_parallel()
    t1 = time()
    run_times.append(t1-t0)


