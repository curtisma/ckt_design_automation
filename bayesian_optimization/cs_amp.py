import numpy as np
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO

import os
from scipy import interpolate
import time

import sys
sys.path.append('./')
from framework.wrapper import ngspice_wrapper as sim

eval_core = sim.CsAmpEvaluationCore("./framework/yaml_files/cs_amp.yaml")
time_vec = []


def func(res_idx, mul_idx):
    start_time = time.time()
    f = -eval_core.cost_fun(res=eval_core.res_vec[int(res_idx)],
                            mul=eval_core.mul_vec[int(mul_idx)],
                            verbose=False)
    end_time = time.time()
    eval_time = end_time - start_time
    time_vec.append(eval_time)
    print("     eval_time   %s seconds " %(eval_time))
    return f

if __name__ == '__main__':
    cov = matern32()
    gp = GaussianProcess(cov)
    acq = Acquisition(mode='ExpectedImprovement')
    param = {'res_idx': ('int', [0, len(eval_core.res_vec)-1]),
             'mul_idx': ('int', [0, len(eval_core.mul_vec)-1]),
             }

    np.random.seed(22)
    gpgo = GPGO(gp, acq, func, param)
    gpgo.run(max_iter=100, init_evals=89)
    result = gpgo.getResult()
    best_params = {}
    best_params['res'] = eval_core.res_vec[int(result[0]['res_idx'])]
    best_params['mul'] = eval_core.mul_vec[int(result[0]['mul_idx'])]
    print(best_params)
