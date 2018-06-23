'''
The difference between this and cs_amp_lut.py is that some of the ea methods have been modified able us to plot the
characteristics of this circuit as the algorithm progresses. These changes are in evaluate_individual() and the
main eaMuPlusLambda() function.
'''

import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from deap.tools.support import Logbook

import numpy as np
import os
from scipy import interpolate

import genome.alg as alg

# from scoop import futures

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
        # specs
        import yaml
        with open(cir_yaml, 'r') as f:
            yaml_data = yaml.load(f)

        # specs
        specs = yaml_data['target_specs']
        self.bw_min     = specs['bw_min']
        self.gain_min   = specs['gain_min']
        self.bias_max   = specs['ibias_max']


        self.res_vec = load_array("sweeps/res_vec.array")
        self.mul_vec = load_array("sweeps/mul_vec.array")
        self.bw_mesh = load_array("sweeps/bw.array")
        self.ibias_mesh = load_array("sweeps/ibias.array")
        self.gain_mesh = load_array("sweeps/gain.array")

        self.bw_fun = interpolate.interp2d(self.res_vec, self.mul_vec, self.bw_mesh, kind="cubic")
        self.bias_fun = interpolate.interp2d(self.res_vec, self.mul_vec, self.ibias_mesh, kind="cubic")
        self.gain_fun = interpolate.interp2d(self.res_vec, self.mul_vec, self.gain_mesh, kind="cubic")

    def cost_fun(self, res_idx, mul_idx, verbose=False):
        bw_cur = self.bw_fun(self.res_vec[res_idx], self.mul_vec[mul_idx])
        gain_cur = self.gain_fun(self.res_vec[res_idx], self.mul_vec[mul_idx])
        ibias_cur = self.bias_fun(self.res_vec[res_idx], self.mul_vec[mul_idx])
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

        return cost, bw_cur, gain_cur, ibias_cur


eval_core = EvaluationCore("./framework/yaml_files/cs_amp.yaml")

def init_inividual():
    # TODO
    # returns a vector representing a random individual
    # ind = [random.randint(1, 100) for _ in range(5)]
    # return creator.Individual(ind)
    res_idx = random.randint(0, len(eval_core.res_vec)-1)
    mul_idx = random.randint(0, len(eval_core.mul_vec)-1)
    return creator.Individual([res_idx, mul_idx])

def evaluate_individual(individual, verbose=False):
    # TODO
    # returns a scalar number representing the cost function of that individual
    # return (sum(individual),)
    res_idx = individual[0]
    mul_idx = individual[1]
    cost_val, bw_val, gain_val, ibias_val = eval_core.cost_fun(int(res_idx), int(mul_idx), verbose)
    return (cost_val, bw_val, gain_val, ibias_val)

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
    cost, _, _, _= evaluate_individual(ind, verbose=True)
    print("cost = %f" %cost)

## EA algorithms modified here to fit our need

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'pop'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    results = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, res in zip(invalid_ind, results):
        ind.fitness.values = (res[0],)
        ind.bw =    res[1]
        ind.gain =  res[2]
        ind.ibias = res[3]

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    # print (population)
    logbook.record(gen=0, nevals=len(invalid_ind), pop=population.copy(),  **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        results = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, res in zip(invalid_ind, results):
            ind.fitness.values = (res[0],)
            ind.bw =    res[1]
            ind.gain =  res[2]
            ind.ibias = res[3]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)
        # print(population)
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), pop=population.copy(), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

## optimization core
creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax,
               gain=float, bw=float, ibias=float)

toolbox = base.Toolbox()

stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
stats_res = tools.Statistics(key=lambda ind: eval_core.res_vec[ind[0]])
stats_mul = tools.Statistics(key=lambda ind: eval_core.mul_vec[ind[1]])
stats_gain = tools.Statistics(key=lambda ind: ind.gain)
stats_bw = tools.Statistics(key=lambda ind: ind.bw)
stats_ibias = tools.Statistics(key=lambda ind: ind.ibias)
mStat = tools.MultiStatistics(fit=stats_fit, res=stats_res, mul=stats_mul, gain=stats_gain,
                              bw=stats_bw, ibias=stats_ibias)
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
toolbox.register("mutate", tools.mutUniformInt, low=[0, 0], up=[len(eval_core.res_vec)-1,
                                                                len(eval_core.mul_vec)-1], indpb=0.5)
toolbox.register("mutUNDO", alg.mutUNDO)
toolbox.register("selectParents", alg.selParentRandom)

# toolbox.register("map", futures.map)

# Decorate the variation operators
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

init_pop_size = 64
pop_size = 32
offspring_size = 8
cxpb = 0.5
mutpb = 0.4
ngen = 150
T0 = 0

def main():

    random.seed()
    pop = toolbox.population(n=init_pop_size)
    history.update(pop)
    pop, logbook = alg.eaMuPlusLambda_cs(pop, toolbox, mu=pop_size, lambda_=offspring_size, cxpb=cxpb,
                                         mutpb=mutpb, ngen=ngen, T0=T0, stats=mStat, verbose=True)
    # pop, logbook = eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=offspring_size, cxpb=cxpb,
    #                               mutpb=mutpb, ngen=ngen, stats=mStat, verbose=True)
    import pprint
    print_best_ind(pop)
    gen = logbook.select('gen')
    fit_avg, fit_min, fit_max, fit_std = np.array(logbook.chapters["fit"].select('avg', 'min', 'max', 'std'))
    bw_avg, bw_min, bw_max, bw_std = np.array(logbook.chapters["bw"].select('avg', 'min', 'max', 'std'))
    gain_avg, gain_min, gain_max, gain_std = np.array(logbook.chapters["gain"].select('avg', 'min', 'max', 'std'))
    ibias_avg, ibias_min, ibias_max, ibias_std = np.array(logbook.chapters["ibias"].select('avg', 'min', 'max', 'std'))
    mul_avg, mul_min, mul_max, mul_std = np.array(logbook.chapters["mul"].select('avg', 'min', 'max', 'std'))
    res_avg, res_min, res_max, res_std = np.array(logbook.chapters["res"].select('avg', 'min', 'max', 'std'))

    populations = logbook.select('pop')

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(321)
    ax.plot(gen, fit_avg)
    ax.plot(gen, fit_min, '--', color='r')
    ax.plot(gen, fit_max, '--', color='r')
    ax.plot(gen, fit_avg+fit_std, ':', color='m')
    ax.plot(gen, fit_avg-fit_std, ':', color='m')
    ax.set_ylabel('cost function')

    ax = fig.add_subplot(322)
    ax.plot(gen, bw_avg)
    ax.plot(gen, bw_min, '--', color='r')
    ax.plot(gen, bw_max, '--', color='r')
    ax.plot(gen, bw_avg+bw_std, ':', color='m')
    ax.plot(gen, bw_avg-bw_std, ':', color='m')
    ax.set_ylabel('bandwidth')

    ax = fig.add_subplot(323)
    ax.plot(gen, gain_avg)
    ax.plot(gen, gain_min, '--', color='r')
    ax.plot(gen, gain_max, '--', color='r')
    ax.plot(gen, gain_avg+gain_std, ':', color='m')
    ax.plot(gen, gain_avg-gain_std, ':', color='m')
    ax.set_ylabel('gain')

    ax = fig.add_subplot(324)
    ax.plot(gen, ibias_avg)
    ax.plot(gen, ibias_min, '--', color='r')
    ax.plot(gen, ibias_max, '--', color='r')
    ax.plot(gen, ibias_avg+ibias_std, ':', color='m')
    ax.plot(gen, ibias_avg-ibias_std, ':', color='m')
    ax.set_ylabel('ibias')

    ax = fig.add_subplot(325)
    ax.plot(gen, mul_avg)
    ax.plot(gen, mul_min, '--', color='r')
    ax.plot(gen, mul_max, '--', color='r')
    ax.plot(gen, mul_avg+mul_std, ':', color='m')
    ax.plot(gen, mul_avg-mul_std, ':', color='m')
    ax.set_ylabel('mul')
    ax.set_xlabel('number of generation')

    ax = fig.add_subplot(326)
    ax.plot(gen, res_avg)
    ax.plot(gen, res_min, '--', color='r')
    ax.plot(gen, res_max, '--', color='r')
    ax.plot(gen, res_avg+res_std, ':', color='m')
    ax.plot(gen, res_avg-res_std, ':', color='m')
    ax.set_ylabel('res')
    ax.set_xlabel('number of generation')


    fig.tight_layout()

    ## helper function for demonstration

    def cost_fc(ibias_cur, gain_cur, bw_cur):
        bw_min = 1e9
        gain_min = 3
        bias_max = 1e-3

        cost = 0
        if bw_cur < bw_min:
            cost += abs(bw_cur/bw_min - 1.0)
        if gain_cur < gain_min:
            cost += abs(gain_cur/gain_min - 1.0)
        cost += abs(ibias_cur/bias_max)/10

        return cost

    def load_array(fname):
        with open(fname, "rb") as f:
            arr = np.load(f)
        return arr

    bw_mat =    load_array("./genome/sweeps/bw.array")
    gain_mat =  load_array("./genome/sweeps/gain.array")
    Ibias_mat = load_array("./genome/sweeps/ibias.array")
    mul_vec =   load_array("./genome/sweeps/mul_vec.array")
    res_vec =   load_array("./genome/sweeps/res_vec.array")

    cost_mat = [[cost_fc(Ibias_mat[mul_idx][res_idx],
                         gain_mat[mul_idx][res_idx],
                         bw_mat[mul_idx][res_idx]) for res_idx in \
                 range(len(res_vec))] for mul_idx in range(len(mul_vec))]

    cost_min = np.min(np.min(cost_mat))
    cost_max = np.max(np.max(cost_mat))

    fig = plt.figure()
    for i in range(35):
        ax = fig.add_subplot(7,5,i+1)
        mappable = ax.pcolormesh(cost_mat, cmap='OrRd', vmin=cost_min, vmax=cost_max)
        plt.colorbar(mappable)
        if i>=15:
            ax.set_xlabel('res_idx')
        if i%5==0:
            ax.set_ylabel('mul_idx')
        x = [ind[0] for ind in populations[i]]
        y = [ind[1] for ind in populations[i]]
        ax.scatter(x, y, marker = 'x')
        ax.axis([0, 250, 0, 100])

    # fig.tight_layout()

    import pickle
    with open("./genome/log/detailed_logbook.pickle", 'wb') as f:
        pickle.dump(logbook, f)


if __name__ == '__main__':
    main()
