
import random

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import numpy

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

history = tools.History()

def init_inividual():
    # TODO
    # returns a vector representing a random individual
    ind = [random.randint(1, 100) for _ in range(5)]

    return creator.Individual(ind)

def evaluate_individual(individual):
    # TODO
    # returns a scalar number representing the cost function of that individual
    return (sum(individual),)

toolbox.register("individual", init_inividual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selBest)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutGaussian, mu=[50, 50, 50, 50, 50], sigma=[10, 10, 10, 10, 10], indpb=0.05)

# Decorate the variation operators
toolbox.decorate("mate", history.decorator)
toolbox.decorate("mutate", history.decorator)

init_pop_size = 1024
pop_size = 1024
offspring_size = 1024
cxpb = 0.5
mutpb = 0.1
ngen = 20

def main():

    pop = toolbox.population(n=init_pop_size)
    history.update(pop)
    print(pop)
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=offspring_size, cxpb=cxpb,
                                              mutpb=mutpb, ngen=ngen, stats=stats, verbose=True)
    import pprint
    print (pop)
    # pprint.pprint(history.genealogy_history)
    # pprint.pprint(history.genealogy_tree)
    import pickle
    with open("./genome/log/logbook.pickle", 'wb') as f:
        pickle.dump(logbook, f)


if __name__ == '__main__':
    main()
