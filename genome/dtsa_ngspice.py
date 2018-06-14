import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np
import os
from scipy import interpolate

from framework.wrapper import DTSA as sim

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

eval_core = sim.EvaluationCore("./framework/yaml_files/dtsa.yaml")

def init_inividual():
    # TODO
    # returns a vector representing a random individual
    # ind = [random.randint(1, 100) for _ in range(5)]
    # return creator.Individual(ind)
    ind_list = [random.choice(eval_core.param_vec_dict[str(key)]) for key in eval_core.param_vec_dict.keys()]
    return creator.Individual(ind_list)

def evaluate_individual(individual, verbose=False):
    # TODO
    # returns a scalar number representing the cost function of that individual
    # return (sum(individual),)
    mapping = eval_core.param_vec_dict.keys()
    param_dict = dict(zip(mapping, individual))
    cost_val = eval_core.cost_fun(verbose=verbose, **param_dict)
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
toolbox.register("mutate", tools.mutGaussian, mu=[50, 50, 50, 50, 50, 50, 50], sigma=[10, 10, 10, 10, 10, 10, 10], indpb=0.05)

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
    with open("./genome/log/two_stage_full_logbook.pickle", 'wb') as f:
        pickle.dump(logbook, f)


if __name__ == '__main__':
    main()
