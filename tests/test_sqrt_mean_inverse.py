from scipy.io.matlab import loadmat
import numpy as np
eps = np.finfo(np.float64).eps
m = loadmat("sqrtinv.mat", squeeze_me=True)
from SqrtMeanInverse import SqrtMeanInverse

py_gamI = SqrtMeanInverse(m["gam"])
