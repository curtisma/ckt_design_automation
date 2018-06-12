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

debug = False

class NgSpiceWrapper(object):

    BASE_TMP_DIR = os.path.abspath("/tmp/circuit_drl")

    def __init__(self, num_process, design_netlist):

        _, dsg_netlist_fname = os.path.split(design_netlist)
        self.base_design_name = os.path.splitext(dsg_netlist_fname)[0]
        self.num_process = num_process
        self.gen_dir = os.path.join(NgSpiceWrapper.BASE_TMP_DIR, "designs_" + self.base_design_name)

        os.makedirs(NgSpiceWrapper.BASE_TMP_DIR, exist_ok=True)
        os.makedirs(self.gen_dir, exist_ok=True)

        raw_file = open(design_netlist, 'r')
        self.tmp_lines = raw_file.readlines()
        raw_file.close()


    def get_design_name(self, state):
        fname = self.base_design_name
        for keyword, value in state.items():
            fname += "_" + keyword + "_" + str(value)
        return fname

    def create_design(self, state):
        new_fname = self.get_design_name(state)
        design_folder = os.path.join(self.gen_dir, new_fname)
        os.makedirs(design_folder, exist_ok=True)

        fpath = os.path.join(design_folder, new_fname + '.cir')

        lines = copy.deepcopy(self.tmp_lines)
        for line_num, line in enumerate(lines):
            if '.param' in line:
                for key, value in state.items():
                    regex = re.compile("%s=(\S+)" % (key))
                    found = regex.search(line)
                    if found:
                        new_replacement = "%s=%s" % (key, str(value))
                        lines[line_num] = lines[line_num].replace(found.group(0), new_replacement)
            if 'wrdata' in line:
                regex = re.compile("wrdata\s*(\w+\.\w+)\s*")
                found = regex.search(line)
                if found:
                    replacement = os.path.join(design_folder, found.group(1))
                    lines[line_num] = lines[line_num].replace(found.group(1), replacement)

        with open(fpath, 'w') as f:
            f.writelines(lines)
            f.close()
        return design_folder, fpath

    def simulate(self, fpath):
        info = 0 # this means no error occurred
        command = "ngspice -b %s >/dev/null 2>&1" %fpath
        exit_code = os.system(command)
        if debug:
            print(command)
            print(fpath)

        if (exit_code % 256):
           # raise RuntimeError('program {} failed!'.format(command))
            info = 1 # this means an error has occurred
        return info


    def create_design_and_simulate(self, state, verbose=False):
        if debug:
            print('state', state)
            print('verbose', verbose)
        dsn_name = self.get_design_name(state)
        if verbose:
            print(dsn_name)
        design_folder, fpath = self.create_design(state)
        info = self.simulate(fpath)
        specs = self.translate_result(design_folder)
        return state, specs, info


    def run(self, states, verbose=False):
        """

        :param states:
        verbose: If True it will print the design name that was created
        :return:
            results = [(state: dict(param_kwds, param_value), specs: dict(spec_kwds, spec_value), info: int)]
        """
        pool = ThreadPool(processes=self.num_process)
        arg_list = [(state, verbose) for state in states]
        specs = pool.starmap(self.create_design_and_simulate, arg_list)
        pool.close()
        return specs

    def translate_result(self, output_path):
        """
        This method needs to be overwritten according to cicuit needs,
        parsing output, playing with the results to get a cost function, etc.
        The designer should look at his/her netlist and accordingly write this function.

        :param output_path:
        :return:
        """
        result = None
        return result


"""
this is an example of using NgSpiceWrapper for a cs_amp ac and dc simulation
Look at the cs_amp.cir as well to make sense out of the parser.
When you run the template netlist it will generate the data in a way that can be easily handled with the following
class methods.
"""
class CsAmpClass(NgSpiceWrapper):

    def translate_result(self, output_path):
        """

        :param output_path:
        :return
            result: dict(spec_kwds, spec_value)
        """

        # use parse output here
        freq, vout, Ibias = self.parse_output(output_path)
        bw = self.find_bw(vout, freq)
        gain = self.find_dc_gain(vout)


        spec = dict(
            bw=bw,
            gain=gain,
            Ibias=Ibias
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
        vout = ac_raw_outputs[:, 1]
        ibias = -dc_raw_outputs[1]

        return freq, vout, ibias

    def find_dc_gain (self, vout):
        return np.abs(vout)[0]

    def find_bw(self, vout, freq):
        gain = np.abs(vout)
        gain_3dB = gain[0] / np.sqrt(2)
        return self._get_best_crossing(freq, gain, gain_3dB)


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

## helper function for demonstration
def generate_random_state (len):
    states = []
    for _ in range(len):
        vbias = random.random() * 1.8
        mul = int(random.random() * (100 - 1) + 1)
        rload = random.random() * (1000 - 10) + 10
        #cload = random.random() * (1e-12 - 1e-15) + 1e-15
        states.append(dict(
            vbias=vbias,
            mul=mul,
            rload=rload,
            #cload=cload
        ))
    return states

def load_array(fname):
    with open(fname, "rb") as f:
        arr = np.load(f)
    return arr

def save_array(fname, arr):
    with open(fname, "wb") as f:
        np.save(f, arr)
    return arr

if __name__ == '__main__':

    """
    example usage of CsAmpClass we just wrote:
    This common source is comprised of a single nmos transistor and single resistor.
    The parameters are Width, Length and multiplier of the nmos, resistor value, VGS bias of the nmos.
    Cload is assumed to be given. The goal is to satisfy some functionality specs (gain_min, bw_min) while
    getting the minimum ibias as the objective of the optimization. This section of code only illustrates
    how to generate the data and validate different points in the search space.
    
    """
    num_process = 4
    dsn_netlist = './genome/netlist/cs_amp.cir'
    cs_env = CsAmpClass(num_process=num_process, design_netlist=dsn_netlist)

    # example of running it for one example point and getting back the data
    state_list = [{'mul': 4, 'rload': 1600}]
    results = cs_env.run(state_list, verbose=True)
    if debug:
        print(results)

    ##  generate the two-axis grid world of the cs_amp example: (rload, mul) in each cell
    #   store bw, gain, Ibias and store them in file for later use:
    gen_data = True
    verbose = True

    if gen_data:
        mul_vec = np.arange(1, 100, 1)
        res_vec = np.arange(10, 5000, 20)
        result_list = []

        for i, mul in enumerate(mul_vec):
            for j, res in enumerate(res_vec):
                state_list = [{'mul': mul, 'rload': res}]
                results = cs_env.run(state_list, verbose=verbose)
                result_list.append(results[0][1])

        Ibias_vec = [result['Ibias'] for result in result_list]
        bw_vec = [result['bw'] for result in result_list]
        gain_vec = [result['gain'] for result in result_list]

        Ibias_mat = np.reshape(Ibias_vec, [len(mul_vec), len(res_vec)])
        bw_mat = np.reshape(bw_vec, [len(mul_vec), len(res_vec)])
        gain_mat = np.reshape(gain_vec, [len(mul_vec), len(res_vec)])

        # the data is going to be mul as x and res as y
        save_array("./genome/sweeps/bw.array", bw_mat)
        save_array("./genome/sweeps/gain.array", gain_mat)
        save_array("./genome/sweeps/ibias.array", Ibias_mat)
        save_array("./genome/sweeps/mul_vec.array", mul_vec)
        save_array("./genome/sweeps/res_vec.array", res_vec)

    if not gen_data:
        bw_mat =    load_array("./genome/sweeps/bw.array")
        gain_mat =  load_array("./genome/sweeps/gain.array")
        Ibias_mat = load_array("./genome/sweeps/ibias.array")
        mul_vec =   load_array("./genome/sweeps/mul_vec.array")
        res_vec =   load_array("./genome/sweeps/res_vec.array")

    # Plotting the data
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm

    mul_mat, res_mat = np.meshgrid(mul_vec, res_vec, indexing='ij', copy=False)
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(mul_mat, res_mat, Ibias_mat, rstride=1, cstride=1, linewidth=0, cmap=cm.cubehelix)
    ax.set_xlabel('multiplier')
    ax.set_ylabel('res_values')
    ax.set_zlabel('Ibias')
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(mul_mat, res_mat, bw_mat, rstride=1, cstride=1, linewidth=0, cmap=cm.cubehelix)
    ax.set_xlabel('multiplier')
    ax.set_ylabel('res_values')
    ax.set_zlabel('bandwidth')
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(mul_mat, res_mat, gain_mat, rstride=1, cstride=1, linewidth=0, cmap=cm.cubehelix)
    ax.set_xlabel('multiplier')
    ax.set_ylabel('res_values')
    ax.set_zlabel('gain')

    plt.show()




