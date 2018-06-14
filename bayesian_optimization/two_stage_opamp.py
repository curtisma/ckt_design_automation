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
from framework.wrapper import TwoStageClass as sim

eval_core = sim.EvaluationCore("./framework/yaml_files/two_stage_opamp.yaml")
time_vec = []

def func(mp1_idx, mn1_idx, mn3_idx, mp3_idx, mn5_idx, mn4_idx , cc_idx):
    start_time = time.time()
    f = -eval_core.cost_fun(mp1=  eval_core.mp1_vec[int(mp1_idx)],
                            mn1=  eval_core.mn1_vec[int(mn1_idx)],
                            mn3=  eval_core.mn3_vec[int(mn3_idx)],
                            mp3=  eval_core.mp3_vec[int(mp3_idx)],
                            mn5=  eval_core.mn5_vec[int(mn5_idx)],
                            mn4=  eval_core.mn4_vec[int(mn4_idx)],
                            cc=   eval_core.cc_vec[int(cc_idx)],
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
    param = {'mp1_idx': ('int', [0, len(eval_core.mp1_vec)-1]),
             'mn1_idx': ('int', [0, len(eval_core.mn1_vec)-1]),
             'mn3_idx': ('int', [0, len(eval_core.mn3_vec)-1]),
             'mp3_idx': ('int', [0, len(eval_core.mp3_vec)-1]),
             'mn5_idx': ('int', [0, len(eval_core.mn5_vec)-1]),
             'mn4_idx': ('int', [0, len(eval_core.mn4_vec)-1]),
             'cc_idx':  ('int', [0, len(eval_core.cc_vec)-1]),
             }

    np.random.seed(20)
    gpgo = GPGO(gp, acq, func, param, n_jobs=1)
    gpgo.run(max_iter=20, init_evals=1000)
    # print(time_vec)

