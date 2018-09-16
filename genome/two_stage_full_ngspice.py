import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy as np
import os
from scipy import interpolate

from framework.wrapper import TwoStageComplete as sim
import genome.alg as alg
import sys

log_file = './genome/log.txt'
file = open(log_file,'w')
# origin_stdout = sys.stdout
# sys.stdout = file

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
    mp1_idx = random.randint(0,len(eval_core.mp1_vec)-1)
    mn1_idx = random.randint(0,len(eval_core.mn1_vec)-1)
    mn3_idx = random.randint(0,len(eval_core.mn3_vec)-1)
    mp3_idx = random.randint(0,len(eval_core.mp3_vec)-1)
    mn5_idx = random.randint(0,len(eval_core.mn5_vec)-1)
    mn4_idx = random.randint(0,len(eval_core.mn4_vec)-1)
    cc_idx =  random.randint(0,len(eval_core.cc_vec)-1)

    return creator.Individual([mp1_idx,
                               mn1_idx,
                               mn3_idx,
                               mp3_idx,
                               mn5_idx,
                               mn4_idx,
                               cc_idx])

def evaluate_individual(individual, verbose=False):
    # TODO
    # returns a scalar number representing the cost function of that individual
    # return (sum(individual),)
    mp1_idx = int(individual[0])
    mn1_idx = int(individual[1])
    mn3_idx = int(individual[2])
    mp3_idx = int(individual[3])
    mn5_idx = int(individual[4])
    mn4_idx = int(individual[5])
    cc_idx  = int(individual[6])
    cost_val = eval_core.cost_fun(mp1_idx, mn1_idx, mp3_idx, mn3_idx, mn4_idx, mn5_idx, cc_idx, verbose=verbose)
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
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxOnePoint)
# toolbox.register("mutate", tools.mutGaussian, mu=[50, 50, 50, 50, 50, 50, 5e-12], sigma=[10, 10, 10, 10, 10, 10, 1e-12], indpb=0.05)
toolbox.register("mutate", tools.mutUniformInt, low=[0 for _ in range(7)], up=[len(eval_core.mp1_vec)-1,
                                                                               len(eval_core.mn1_vec)-1,
                                                                               len(eval_core.mp3_vec)-1,
                                                                               len(eval_core.mn3_vec)-1,
                                                                               len(eval_core.mn4_vec)-1,
                                                                               len(eval_core.mn5_vec)-1,
                                                                               len(eval_core.cc_vec)-1], indpb=0.5) # mp1, mn1, mp3, mn3, mn4, mn5, cc

toolbox.register("mutUNDO", alg.mutUNDO)
toolbox.register("selectParents", alg.selParentRandom)

# Decorate the variation operators
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

init_pop_size = 512
pop_size = int(init_pop_size/2)
offspring_size = 256 #int(init_pop_size/8)
cxpb = 0.6
mutpb = 0.4
ngen = 150
T0 = 0.01

def main():

    # random.seed(11)
    pop = toolbox.population(n=init_pop_size)
    history.update(pop)
    print(pop)
    mut_scheduler = alg.LinearSchedule(ngen, 0.05, 0.4)
    pop, logbook = alg.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=offspring_size, cxpb=cxpb,
                                      mut_scheduler=mut_scheduler, ngen=ngen, T0=T0, stats=mStat, verbose=True)
    # pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=offspring_size, cxpb=cxpb,
    #                                          mutpb=mutpb, ngen=ngen, stats=mStat, verbose=True)
    import pprint
    print_best_ind(pop)
    gen = logbook.select('gen')

    fit_avg, fit_min, fit_max, fit_std = np.array(logbook.chapters["fit"].select('avg', 'min', 'max', 'std'))

    print (fit_min)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(gen, fit_avg)
    ax.plot(gen, fit_min, '--', color='r')
    ax.plot(gen, fit_max, '--', color='r')
    ax.plot(gen, fit_avg+fit_std, ':', color='m')
    ax.plot(gen, fit_avg-fit_std, ':', color='m')
    ax.set_ylabel('cost function')
    # pprint.pprint(history.genealogy_history)
    # pprint.pprint(history.genealogy_tree)
    import pickle
    with open("./genome/log/two_stage_full_logbook.pickle", 'wb') as f:
        pickle.dump(logbook, f)


if __name__ == '__main__':
    print(evaluate_individual([98,24,77,71,25,35,19]))
    # main()
