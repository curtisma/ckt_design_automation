import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np
import os
from scipy import interpolate

from framework.wrapper import TwoStageClass as sim

######################################################################
## helper functions for working with files
def rel_path(fname):
    return os.path.join(os.path.dirname(__file__), fname)

def load_array(fname):
    with open(rel_path(fname), "rb") as f:
        arr = np.load(f)
    return arr
######################################################################
## function and classes related to this specific problem and dealing with the evaluation core
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
        self.env = sim.TwoStageClass(num_process=num_process, design_netlist=dsn_netlist)

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


eval_core = EvaluationCore("./genome/yaml_files/two_stage_opamp.yaml")

def init_inividual():
    # TODO
    # returns a vector representing a random individual
    # ind = [random.randint(1, 100) for _ in range(5)]
    # return creator.Individual(ind)
    mp1 = random.choice(eval_core.mp1_vec)
    mn1 = random.choice(eval_core.mn1_vec)
    mn3 = random.choice(eval_core.mn3_vec)
    mp3 = random.choice(eval_core.mp3_vec)
    mn5 = random.choice(eval_core.mn5_vec)
    mn4 = random.choice(eval_core.mn4_vec)
    cc = random.choice(eval_core.cc_vec)

    return creator.Individual([mp1,
                               mn1,
                               mn3,
                               mp3,
                               mn5,
                               mn4,
                               cc])

def evaluate_individual(individual, verbose=False):
    # TODO
    # returns a scalar number representing the cost function of that individual
    # return (sum(individual),)
    mp1 = individual[0]
    mn1 = individual[1]
    mn3 = individual[2]
    mp3 = individual[3]
    mn5 = individual[4]
    mn4 = individual[5]
    cc  = individual[6]
    cost_val = eval_core.cost_fun(mp1, mn1, mp3, mn3, mn4, mn5, cc, verbose=verbose)
    return (cost_val,)


######################################################################
## helper functions for opt_core
def print_best_ind(population):
    best_ind = None
    best_fitness = float('inf')
    print("--------- best_individual in the final population ---------")
    for ind in population:
        if ind.fitness.values[0] < best_fitness:
            best_fitness = ind.fitness.values[0]
            best_ind = ind
    print("best_sol = {}" .format(best_ind))
    cost = evaluate_individual(best_ind, verbose=True)
    print("cost = %f" %cost)

## optimization core
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
# stats_res = tools.Statistics(key=lambda ind: ind[0])
# stats_mul = tools.Statistics(key=lambda ind: ind[1])
mStat = tools.MultiStatistics(fit=stats_fit)
mStat.register("avg", np.mean)
mStat.register("std", np.std)
mStat.register("min", np.min)
mStat.register("max", np.max)

history = tools.History()

toolbox.register("individual", init_inividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selBest)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=[50, 50, 50, 50, 50, 50, 5e-12], sigma=[10, 10, 10, 10, 10, 10, 1e-12], indpb=0.05)

# Decorate the variation operators
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

init_pop_size = 2048
pop_size = 1024
offspring_size = 1024
cxpb = 0.6
mutpb = 0.05
ngen = 30

def main():

    pop = toolbox.population(n=init_pop_size)
    history.update(pop)
    print(pop)
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=offspring_size, cxpb=cxpb,
                                             mutpb=mutpb, ngen=ngen, stats=mStat, verbose=True)
    import pprint
    print_best_ind(pop)
    # pprint.pprint(history.genealogy_history)
    # pprint.pprint(history.genealogy_tree)
    import pickle
    with open("./genome/log/logbook.pickle", 'wb') as f:
        pickle.dump(logbook, f)


if __name__ == '__main__':
    main()
