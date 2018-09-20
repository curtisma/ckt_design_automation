"""
>>> run -i genetic_nn/run_expert_ga.py two_stage_full -n 150 -i 200 -k 20 --seed 10 --exp_name i_200_k_20
"""
import numpy as np
import random
import os
import yaml
import importlib
import copy
import math
import pickle
import time

from deap import tools

import genetic_nn.es as es
import genetic_nn.util as util
from util import Design

## function and classes related to this specific problem and dealing with the evaluation core

FRAMEWORK_YAML_DIR = "./framework/yaml_files"

######################################################################

def generate_data_set(n, eval_core, evaluate=True):
    if evaluate:
        print("[info] generating %d random data" %n)
    data_set = []

    for _ in range(n):

        param_list = []
        for value in eval_core.params.values():
            len_param_vec = math.floor((value[1]-value[0])/value[2])
            param_value = random.randint(0, len_param_vec-1)
            param_list.append(param_value)

        sample_dsn = Design(param_list)
        if evaluate:
            result = eval_core.cost_fun(sample_dsn)
            sample_dsn.cost = result[0]
            for i, key in enumerate(sample_dsn.specs.keys()):
                sample_dsn.specs[key] = result[i+1]
        data_set.append(sample_dsn)

    return data_set

def generate_offspring(population, eval_core, cxpb, mutpb):

    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")
    op_choice = random.random()
    offsprings = []
    lows, ups = [], []
    for value in eval_core.params.values():
        lows.append(0)
        len_param_vec = math.floor((value[1]-value[0])/value[2])
        ups.append(len_param_vec-1)

    if op_choice <= cxpb:            # Apply crossover
        ind1, ind2 = es.selectParents(population)
        ind1, ind2 = es.mate(ind1, ind2, low=lows, up=ups)
        offsprings += [ind1, ind2]
    elif op_choice < cxpb + mutpb:      # Apply mutation
        ind = es.select_for_mut(population)
        ind, = es.mutate(ind, low=lows, up=ups)
        offsprings.append(ind)
    return offsprings

def run_ga(init_pop, pop_size, offspring_size, ngen, eval_core, stats, history, max_iter=1000, verbose=True):

    total_n_evals = len(init_pop)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    history.append(init_pop)
    record = stats.compile(init_pop) if stats is not None else {}
    logbook.record(gen=0, nevals=len(init_pop),  **record)
    if verbose:
        print(logbook.stream)

    pop = copy.deepcopy(init_pop)
    for gen in range(1, ngen+1):
        offsprings = []
        n_iter = 0
        while len(offsprings) < offspring_size and n_iter < max_iter:
            n_iter += 1
            new_designs = generate_offspring(copy.deepcopy(pop), eval_core,  es.G.cxpb, es.G.mutpb)
            for new_design in new_designs:
                if any([(new_design == row) for row in pop]):
                    # if design is already in the design pool skip ...
                    # print("[debug] design {} already exists".format(new_design))
                    continue
                new_design.cost = eval_core.cost_fun(new_design)[0]
            offsprings += new_designs
        total_n_evals += len(offsprings)
        pop[:] = es.select(pop+offsprings, pop_size)

        history.append(pop)
        record = stats.compile(pop) if stats is not None else {}
        logbook.record(gen=gen, nevals=offspring_size,  **record)
        if verbose:
            print(logbook.stream)

        costs = [x.cost for x in pop]
        if np.std(costs) < 0.1:
            break

    return pop, total_n_evals

def main():

    start = time.time()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('yaml_fname', type=str)
    parser.add_argument('--ngen', '-n', type=int, default=150)
    parser.add_argument('--n_init_pop', '-i', type=int, default=200)
    parser.add_argument('--n_offsprings', '-k', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='vanilla')
    args = parser.parse_args()

    # get the yaml file
    cir_yaml = os.path.join(FRAMEWORK_YAML_DIR, args.yaml_fname+".yaml")
    with open(cir_yaml, 'r') as f:
        yaml_data = yaml.load(f)

    # setup the simulator module
    wrapper_name = yaml_data['wrapper_name']
    sim = importlib.import_module('framework.wrapper.%s' % wrapper_name)
    eval_core = sim.EvaluationCore(cir_yaml)

    # setup the seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # setup the log and history stuff
    history = []
    stats_fit = tools.Statistics(key=lambda ind: ind.cost)
    mStat = tools.MultiStatistics(fit=stats_fit)
    mStat.register("avg", np.mean)
    mStat.register("std", np.std)
    mStat.register("min", np.min)
    mStat.register("max", np.max)

    if os.path.exists('genetic_nn/checkpoint/two_stage/init_data.pkl'):
        with open('genetic_nn/checkpoint/two_stage/init_data.pkl', 'rb') as f:
            pop = pickle.load(f)
    else:
        pop = generate_data_set(args.n_init_pop, eval_core)
        with open('genetic_nn/checkpoint/two_stage/init_data.pkl', 'wb') as f:
            pickle.dump(pop, f)

    init_data_time = time.time()
    pop, total_n_evals = run_ga(init_pop=pop,
                                pop_size=len(pop),
                                offspring_size=args.n_offsprings,
                                ngen=args.ngen,
                                eval_core=eval_core,
                                stats=mStat,
                                history=history,
                                max_iter=args.max_iter)

    end_time = time.time()
    log_dir = 'genetic_nn/log_files/'+args.yaml_fname+'_'+args.exp_name+'_ga_history_'+\
              time.strftime("%d-%m-%Y_%H-%M-%S")+'.pkl'
    with open(log_dir, 'wb') as f:
        pickle.dump(history, f)

    pop_sorted = sorted(pop, key=lambda x: x.cost)
    print("[finished] total_time = {}".format(end_time-start))
    print("[finished] ga_time = {}".format(end_time-init_data_time))
    print("[finished] total_n_evals = {}".format(total_n_evals))
    print("[finished] best_solution = {}".format(pop_sorted[0]))
    print("[finished] cost = {}".format(pop_sorted[0].cost))
    print("[finished] performance \n{} ".format(pop_sorted[0].specs))

if __name__ == '__main__':
    main()
