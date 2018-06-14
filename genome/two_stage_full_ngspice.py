import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np
import os
from scipy import interpolate

from framework.wrapper import TwoStageComplete as sim

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

eval_core = sim.EvaluationCore("./framework/yaml_files/two_stage_full.yaml")

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
    with open("./genome/log/two_stage_full_logbook.pickle", 'wb') as f:
        pickle.dump(logbook, f)


if __name__ == '__main__':
    main()
