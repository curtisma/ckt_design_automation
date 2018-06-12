import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np
import os
from scipy import interpolate

from framework.wrapper import ngspice_wrapper as sim

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
        self.bw_min     = specs['bw_min']
        self.gain_min   = specs['gain_min']
        self.bias_max   = specs['ibias_max']

        num_process = yaml_data['num_process']
        dsn_netlist = yaml_data['dsn_netlist']
        self.env = sim.CsAmpClass(num_process=num_process, design_netlist=dsn_netlist)

        params = yaml_data['params']
        self.res_vec = np.arange(params['rload'][0], params['rload'][1], params['rload'][2])
        self.mul_vec = np.arange(params['mul'][0], params['mul'][1], params['mul'][2])

    def cost_fun(self, res, mul, verbose=False):
        """

        :param res:
        :param mul:
        :param verbose: if True will print the specification performance of the best individual
        :return:
        """

        state = [{'rload': res, 'mul': int(mul)}]
        results = self.env.run(state, verbose=False)
        bw_cur = results[0][1]['bw']
        gain_cur = results[0][1]['gain']
        ibias_cur = results[0][1]['Ibias']

        if verbose:
            print('bw = %f vs. bw_min = %f' %(bw_cur, self.bw_min))
            print('gain = %f vs. gain_min = %f' %(gain_cur, self.gain_min))
            print('Ibias = %f vs. Ibias_max = %f' %(ibias_cur, self.bias_max))

        cost = 0
        if bw_cur < self.bw_min:
            cost += abs(bw_cur/self.bw_min - 1.0)
        if gain_cur < self.gain_min:
            cost += abs(gain_cur/self.gain_min - 1.0)
        cost += abs(ibias_cur/self.bias_max)/10

        return cost


eval_core = EvaluationCore("./genome/yaml_files/cs_amp.yaml")

def init_inividual():
    # TODO
    # returns a vector representing a random individual
    # ind = [random.randint(1, 100) for _ in range(5)]
    # return creator.Individual(ind)
    res = random.choice(eval_core.res_vec)
    mul = random.choice(eval_core.mul_vec)
    return creator.Individual([res, mul])

def evaluate_individual(individual, verbose=False):
    # TODO
    # returns a scalar number representing the cost function of that individual
    # return (sum(individual),)
    res = individual[0]
    mul = individual[1]
    cost_val = eval_core.cost_fun(res, mul, verbose=verbose)
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
    cost = evaluate_individual(ind, verbose=True)
    print("cost = %f" %cost)
## optimization core
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_res = tools.Statistics(key=lambda ind: ind[0])
stats_mul = tools.Statistics(key=lambda ind: ind[1])
mStat = tools.MultiStatistics(fit=stats_fit, res=stats_res, mul=stats_mul)
mStat.register("avg", np.mean)
mStat.register("std", np.std)
# mStat.register("min", np.min)
# mStat.register("max", np.max)

history = tools.History()

toolbox.register("individual", init_inividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selBest)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=[50, 50], sigma=[10, 10], indpb=0.05)

# Decorate the variation operators
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

init_pop_size = 512
pop_size = 256
offspring_size = 256
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
