import re
import numpy as np
import copy
from multiprocessing.dummy import Pool as ThreadPool
import os
import abc
import scipy.interpolate as interp
import scipy.optimize as sciopt
import random
import time
import pprint

debug = True

from framework.wrapper.ngspice_wrapper import NgSpiceWrapper

class TwoStageClass(NgSpiceWrapper):

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        freq, vout,  ibias = self.parse_output(output_path)
        gain = self.find_dc_gain(vout)
        ugbw = self.find_ugbw(freq, vout)
        phm = self.find_phm(freq, vout)


        spec = dict(
            ugbw=ugbw,
            gain=gain,
            phm=phm,
            Ibias=ibias
        )

        return spec

    def parse_output(self, output_path):

        ac_fname = os.path.join(output_path, 'ac.csv')
        dc_fname = os.path.join(output_path, 'dc.csv')

        if not os.path.isfile(ac_fname) or not os.path.isfile(dc_fname):
            print("ac/dc file doesn't exist: %s" % output_path)

        ac_raw_outputs = np.genfromtxt(ac_fname, skip_header=1)
        dc_raw_outputs = np.genfromtxt(dc_fname, skip_header=1)
        freq = ac_raw_outputs[:, 0]
        vout_real = ac_raw_outputs[:, 1]
        vout_imag = ac_raw_outputs[:, 2]
        vout = vout_real + 1j*vout_imag
        ibias = -dc_raw_outputs[1]

        return freq, vout, ibias

    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    def find_ugbw(self, freq, vout):
        gain = np.abs(vout)
        return self._get_best_crossing(freq, gain, val=1)

    def find_phm(self, freq, vout):
        gain = np.abs(vout)
        phase = np.angle(vout, deg=False)
        phase = np.unwrap(phase) # unwrap the discontinuity
        phase = np.rad2deg(phase) # convert to degrees

        phase_fun = interp.interp1d(freq, phase, kind='quadratic')
        ugbw = self._get_best_crossing(freq, gain, val=1)
        if phase[0] <= 0:
            if phase_fun(ugbw) > 0:
                return -180+phase_fun(ugbw)
            else:
                return 180 + phase_fun(ugbw)
        else:
            print ('stuck in else statement')


    def _get_best_crossing(cls, xvec, yvec, val):
        interp_fun = interp.InterpolatedUnivariateSpline(xvec, yvec)

        def fzero(x):
            return interp_fun(x) - val

        xstart, xstop = xvec[0], xvec[-1]
        try:
            return sciopt.brentq(fzero, xstart, xstop)
        except ValueError:
            # avoid no solution
            if abs(fzero(xstart)) < abs(fzero(xstop)):
                return xstart
            return xstop

class EvaluationCore(object):

    def __init__(self, cir_yaml):
        import yaml
        with open(cir_yaml, 'r') as f:
            yaml_data = yaml.load(f)

        # specs
        specs = yaml_data['target_specs']
        self.ugbw_min     = specs['ugbw_min']
        self.gain_min   = specs['gain_min']
        self.phm_min   = specs['phm_min']
        self.bias_max = 10e-3

        num_process = yaml_data['num_process']
        dsn_netlist = yaml_data['dsn_netlist']
        self.env = TwoStageClass(num_process=num_process, design_netlist=dsn_netlist)

        params = yaml_data['params']
        self.mp1_vec = np.arange(params['mp1'][0], params['mp1'][1], params['mp1'][2])
        self.mn1_vec = np.arange(params['mn1'][0], params['mn1'][1], params['mn1'][2])
        self.mn3_vec = np.arange(params['mn3'][0], params['mn3'][1], params['mn3'][2])
        self.mn4_vec = np.arange(params['mn4'][0], params['mn4'][1], params['mn4'][2])
        self.mp3_vec = np.arange(params['mp3'][0], params['mp3'][1], params['mp3'][2])
        self.mn5_vec = np.arange(params['mn5'][0], params['mn5'][1], params['mn5'][2])
        self.cc_vec = np.arange(params['cc'][0], params['cc'][1], params['cc'][2])

    def cost_fun(self, mp1, mn1, mp3, mn3, mn4, mn5 , cc,  verbose=False):
        """

        :param res:
        :param mul:
        :param verbose: if True will print the specification performance of the best individual and file name of
        the netlist
        :return:
        """
        if verbose:
            print("state_before_rounding:{}".format([mp1, mn1, mn3, mp3, mn5, mn4, cc]))

        state = [{'mp1': int(mp1),
                  'mn1': int(mn1),
                  'mp3': int(mp3),
                  'mn3': int(mn3),
                  'mn4': int(mn4),
                  'mn5': int(mn5),
                  'cc':  cc
                  }]
        results = self.env.run(state, verbose=verbose)
        ugbw_cur = results[0][1]['ugbw']
        gain_cur = results[0][1]['gain']
        phm_cur = results[0][1]['phm']
        ibias_cur = results[0][1]['Ibias']

        if verbose:
            print('gain = %f vs. gain_min = %f' %(gain_cur, self.gain_min))
            print('ugbw = %f vs. ugbw_min = %f' %(ugbw_cur, self.ugbw_min))
            print('phm = %f vs. phm_min = %f' %(phm_cur, self.phm_min))
            print('Ibias = %f' %(ibias_cur))

        cost = 0
        if ugbw_cur < self.ugbw_min:
            cost += abs(ugbw_cur/self.ugbw_min - 1.0)
        if gain_cur < self.gain_min:
            cost += abs(gain_cur/self.gain_min - 1.0)
        if phm_cur < self.phm_min:
            cost += abs(phm_cur/self.phm_min - 1.0)
        cost += abs(ibias_cur/self.bias_max)/10

        return cost


if __name__ == '__main__':

    num_process = 1
    dsn_netlist = './genome/netlist/two_stage_opamp.cir'
    env = TwoStageClass(num_process=num_process, design_netlist=dsn_netlist)

    # example of running it for one example point and getting back the data
    state_list = [{'mp1': 68,
                   'mn1': 75,
                   'mn3': 52,
                   'mp3': 58,
                   'mn5': 18,
                   'mn4': 67,
                   'cc': 8e-13
                   }]

    results = env.run(state_list, verbose=True)
    if debug:
        print(results[0][1])
