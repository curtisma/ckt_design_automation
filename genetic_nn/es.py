import random
import numpy as np
from deap import tools
import math

from framework.wrapper import TwoStageComplete as sim

def selParentRandom(population):
    return random.sample(population, 2)

def selectParents(population):
    return selParentRandom(population)

def mate(ind1, ind2,low, up, blend_prob=0.5):
    # a mixture of blend and 2 point crossover
    if random.random() < blend_prob:
        ind1, ind2 = tools.cxBlend(ind1, ind2, alpha=0.5)
        size = min(len(ind1), len(ind2))
        for i, u, l in zip(range(size), up, low):
            ind1[i] = math.floor(ind1[i])
            ind2[i] = math.ceil(ind2[i])
            if ind1[i] > u:
                ind1[i] = u
            elif ind1[i] < l:
                ind1[i] = l
            if ind2[i] > u:
                ind2[i] = u
            elif ind2[i] < l:
                ind2[i] = l
        # ind1 = [math.floor(ind1[i]) for i in range(len(ind1))]
        # ind2 = [math.ceil(ind2[i]) for i in range(len(ind2))]
        return ind1, ind2

    else:
        return tools.cxUniform(ind1, ind2, indpb=0.5)
    # return tools.cxOnePoint(ind1, ind2)
    # return tools.cxTwoPoint(ind1, ind2)

def mutate(ind, low, up):
    return tools.mutUniformInt(ind, low=low, up=up, indpb=0.5)

def select(pop, mu):
    # The best sample in pop is going to have prob=1 but the other samples
    # are going to have linearly diminishing chance in being in the next generation
    # do this until mu designs are chosen
    # sorted_pop = sorted(pop, key=lambda x: x.cost)
    # prob = [(1-i/(len(pop)-1)) for i in range(len(pop))]
    # selected_individuals = []
    # index = 0
    # while len(selected_individuals) < mu:
    #     if not (any([(sorted_pop[index] == row) for row in selected_individuals])):
    #         if random.random() < prob[index]:
    #             selected_individuals.append(sorted_pop[index])
    #
    #     index += 1
    #     index = index % len(sorted_pop)
    # return selected_individuals
    return tools.selBest(pop, mu, fit_attr='fitness')

def select_for_mut(population):
    reverse_sorted_pop = sorted(population, key=lambda x: x.cost, reverse=True)
    ranks = np.arange(1, len(reverse_sorted_pop)+1, 1)
    prob = ranks / np.sum(ranks)
    ind = random.choices(reverse_sorted_pop, weights=prob)[0]
    return ind

class G:
    cxpb = 0.6
    mutpb = 0.4