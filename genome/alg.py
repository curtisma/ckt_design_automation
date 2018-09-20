import random

from deap import tools
import math
import numpy as np

class LinearSchedule(object):

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)



def selParentRandom(population):
    return random.sample(population, 2)

def mutUNDO_cs(last_ind, toolbox, temp, fit_attr="fitness"):
    num_evaluations = 0
    if getattr(last_ind, fit_attr).valid:
        last_fit = getattr(last_ind, fit_attr).values[0]
    else:
        last_res  = toolbox.evaluate(last_ind)
        last_fit = last_res[0]
        last_ind.fitness.values = (last_res[0],)
        last_ind.bw =    last_res[1]
        last_ind.gain =  last_res[2]
        last_ind.ibias = last_res[3]
        num_evaluations += 1
    new_ind = toolbox.clone(last_ind)
    new_ind, = toolbox.mutate(new_ind)

    new_res = toolbox.evaluate(new_ind)
    new_fit = new_res[0]
    new_ind.fitness.values = (new_fit,)
    new_ind.bw =    new_res[1]
    new_ind.gain =  new_res[2]
    new_ind.ibias = new_res[3]
    num_evaluations += 1
    # print("last_ind", last_ind)
    # print("last_fit", last_fit)
    # print("new_ind", new_ind)
    # print("new_fit", new_fit)

    if (new_fit <= last_fit):
        return new_ind, num_evaluations

    else:
        PAF = math.exp(-abs(last_fit-new_fit)/temp)
        if random.random() < PAF: #accept new state with probability PAF
            return new_ind, num_evaluations
        else: #reject new state, keep the last one
            return last_ind, num_evaluations

def mutUNDO(last_ind, toolbox, temp, fit_attr="fitness"):
    num_evaluations = 0
    if getattr(last_ind, fit_attr).valid:
        last_fit = getattr(last_ind, fit_attr).values[0]
    else:
        last_ind.fitness.values = toolbox.evaluate(last_ind)
        num_evaluations += 1
    new_ind = toolbox.clone(last_ind)
    new_ind, = toolbox.mutate(new_ind)

    new_ind.fitness.values = toolbox.evaluate(new_ind)
    new_fit = new_ind.fitness.values[0]
    last_fit = last_ind.fitness.values[0]

    num_evaluations += 1
    # print("last_ind", last_ind)
    # print("last_fit", last_fit)
    # print("new_ind", new_ind)
    # print("new_fit", new_fit)

    if (new_fit <= last_fit):
        return new_ind, num_evaluations

    else:
        PAF = math.exp(-abs(last_fit-new_fit)/temp)
        if random.random() < PAF: #accept new state with probability PAF
            return new_ind, num_evaluations
        else: #reject new state, keep the last one
            return last_ind, num_evaluations



def gen_offspring_cs(population, toolbox, lambda_, cxpb, mutpb, temp):
    """Part of the genetic algorithm for generating the offsprings

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population
    :returns: the total number of evaluations done for generating the new offsprings
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    n_eval_offspring = 0

    for i in range(lambda_):
        op_choice = random.random()
        num_evaluations = 0
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, toolbox.selectParents(population)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            if random.random() < mutpb:
                # mutation method should return the number of new evaluations it runs
                ind1, num_evaluations = toolbox.mutUNDO(ind1, toolbox, temp)
            n_eval_offspring += num_evaluations
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            # print (i,"inside mutation only")
            ind = toolbox.clone(np.random.choice(population))
            ind, num_evaluations = toolbox.mutUNDO(ind, toolbox, temp)
            n_eval_offspring += num_evaluations
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))
    return offspring, n_eval_offspring

def gen_offspring(population, toolbox, lambda_, cxpb, mutpb, temp):
    """Part of the genetic algorithm for generating the offsprings

    :param population: A list of individuals to vary.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param lambda\_: The number of children to produce
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :returns: The final population
    :returns: the total number of evaluations done for generating the new offsprings
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    n_eval_offspring = 0
    population_cloned = list(map(toolbox.clone, population))

    for i in range(lambda_):
        op_choice = random.random()
        num_evaluations = 0
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, toolbox.selectParents(population_cloned)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            if random.random() < mutpb:
                # mutation method should return the number of new evaluations it runs
                ind1, num_evaluations = toolbox.mutUNDO(ind1, toolbox, temp)
            n_eval_offspring += num_evaluations
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            # mutation gives priority to those which have the best fitness.
            population_cloned = sorted(population_cloned, key=lambda x: x.fitness.values[0], reverse=True)
            fitnesses = [ind.fitness.values[0] for ind in population_cloned]
            fitnesses = fitnesses[::-1]
            prob = fitnesses/np.sum(fitnesses)
            ind = toolbox.clone(random.choices(population_cloned, weights=prob))[0]
            population_cloned.remove(ind)
            ind, num_evaluations = toolbox.mutUNDO(ind, toolbox, temp)
            n_eval_offspring += num_evaluations
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))
    return offspring, n_eval_offspring


def eaMuPlusLambda_cs(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                      T0, stats=None, halloffame=None, verbose=__debug__):

    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    """


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
    logbook.record(gen=0, nevals=len(invalid_ind), pop=population.copy(),  **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        temp = 0.9 * T0 / gen
        # Vary the population
        offspring, n_extra_evals = gen_offspring_cs(population, toolbox, lambda_, cxpb, mutpb, temp)
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

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind)+n_extra_evals, pop=population.copy(),  **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

def eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mut_scheduler, ngen,
                       T0, stats=None, halloffame=None, verbose=__debug__):

    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda\_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    """


    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'pop'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), pop=population.copy(),  **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        temp = 0.9 * T0 / gen
        # Vary the population
        mutpb = mut_scheduler.value(gen)
        offspring, n_extra_evals = gen_offspring(population, toolbox, lambda_, cxpb, mutpb, temp)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        fitnesses = [ind.fitness.values[0] for ind in population]
        if np.std(fitnesses) < 0.001:
            break
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind)+n_extra_evals, pop=population.copy(),  **record)
        if verbose:
            print(logbook.stream)

    return population, logbook