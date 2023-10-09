# Code from https://deap.readthedocs.io/en/master/examples/ga_onemax.html
# modified to run on our neural network and evoman objective.
# go step-by-step through tutorial and see what differences you can
# spot between this code and code in the tutorial.

import hydra
import os
import random
import numpy as np
import multiprocessing

from deap import base
from deap import creator
from deap import tools
from evoman.environment import Environment
from evolve.logging import DataVisualizer
from demo_controller import player_controller

# disable visuals and thus make experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"
# hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

n_hidden_neurons = 10

experiment_name = "nn_test"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

env = Environment(
    experiment_name=experiment_name,
    multiplemode="yes",
    enemies=[1, 2, 3, 4, 5, 7, 8],
    speed="fastest",
    logs="off",
    savelogs="no",
    player_controller=player_controller(n_hidden_neurons),
    visuals=False,
)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    toolbox = prepare_toolbox(config)

    # create a data gatherer object
    logger = DataVisualizer(experiment_name, "plus")

    NUM_RUNS = 1
    for i in range(NUM_RUNS):
        print(f"=====RUN {i + 1}/{NUM_RUNS}=====")
        new_seed = 2137 + i * 10
        # best_ind = train_loop(toolbox, config, logger, new_seed, "plus")

        # island
        best_ind = train_loop_island(toolbox, config, logger, new_seed, "plus")
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    #best_ind.save_weights(os.path.join(EXPERIMENT_NAME, "weights.txt"))
    np.savetxt(experiment_name + '/best.txt',best_ind)
    #logger.draw_plots()


def prepare_toolbox(config):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_float", np.random.uniform, -1, 1)
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, 265)


    # define the population to be a np.ndarray of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # ----------
    # Operator registration
    # ----------
    # register the goal / fitness function
    toolbox.register("evaluate", eval_fitness)

    # register the crossover operator
    toolbox.register("mate", tools.cxSimulatedBinary, eta=config.evolve.eta_crossover)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=config.evolve.sigma_mutation,
                     indpb=config.evolve.indpb_mutation)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register(
        "parent_select", tools.selTournament, tournsize=config.evolve.selection_pressure
    )
    toolbox.register(
        "survivor_select", tools.selBest
    )
    # ----------
    return toolbox


# the goal ('fitness') function to be maximized
def eval_fitness(individual):
    f, p, e, t = env.play(pcont=individual)
    return (p - e,)
    #return (env.play(pcont=individual)[0],)

def eval_gain(individual, logger, winner_num, survivor_selection):
    NUM_RUNS = 5
    for _ in range(NUM_RUNS):
        _, p, e, _ = env.play(pcont=individual)
        gain = p - e
        logger.gather_box(winner_num, gain, survivor_selection)

def migrate(islands, migration_size, num_islands):
    for i in range(num_islands):
        # Select individuals to migrate from random island
        migrants = random.sample(islands[i], migration_size)

        # Choose a random target island for migration
        target_island = random.choice([j for j in range(num_islands) if j != i])

        # Create a set of migrant IDs for faster lookup
        migrant_ids = set(id(ind) for ind in migrants)

        # Remove migrants from the source island using a list comprehension
        islands[i] = [ind for ind in islands[i] if id(ind) not in migrant_ids]

        # Add migrants to the target island
        islands[target_island].extend(migrants)

def train_loop_island(toolbox, config, logger, seed, survivor_selection):

    # Could be added to the config if we decide to use this model
    num_gens = 100
    num_islands = 4
    migration_interval = 25
    migration_size = 5
    pop_size = 25
    fits_all = []

    # generation counter
    g = 0

    random.seed(seed)
    np.random.seed(seed)

    islands = [toolbox.population(n=pop_size) for _ in range(num_islands)]

    for island in islands:
        for ind in island:
            # Initialize individuals in each island
            ind[:] = toolbox.individual()

    # for island in islands:
    #     # Evaluate the initial population
    #     fits = toolbox.map(toolbox.evaluate, island)
    #     for ind, fit in zip(island, fits):
    #         ind.fitness.values = fit

    for island in islands:
        # Evaluate the initial population
        update_fitness(toolbox.evaluate, island)
        fits = [ind.fitness.values[0] for ind in island]
        fits_all.append(fits)

    # save gen, max, mean, std
    logger.gather_line(fits_all, g, survivor_selection)

    for generation in range(num_gens):
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Empty list for fitnesses of all islands per generation
        fits_all = []

        # Perform evolution on each island
        for i in range(num_islands):

            print(f'Island {i + 1}')

            # create offspring
            offspring = toolbox.parent_select(islands[i], config.evolve.lambda_coeff * len(islands[i]))

            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Shuffle the offspring
            random.shuffle(offspring)

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # cross two individuals with probability CXPB
                if random.random() < config.evolve.cross_prob:
                    toolbox.mate(child1, child2)
                    # fitness values of the children
                    # must be recalculated later
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                # mutate an individual with probability MUTPB
                if random.random() < config.evolve.mutation_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            update_fitness(toolbox.evaluate, invalid_ind)

            # Replace parents with offspring
            islands[i][:] = offspring

            # Select survivors
            islands[i] = toolbox.survivor_select(islands[i], pop_size)

            # Clone
            islands[i] = list(map(toolbox.clone, islands[i]))

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in islands[i]]
            print_statistics(fits, len(invalid_ind), len(islands[i]))

            fits_all.append(fits)

        # save gen, max, mean
        logger.gather_line(fits_all, g, survivor_selection)

        # Perform migration every `migration_interval` generations
        if generation % migration_interval == 0:
            migrate(islands, migration_size, num_islands)

    print("-- End of (successful) evolution --")

    # Merge all island populations
    pop = []
    for island in islands:
        pop = pop + island

    return tools.selBest(pop, 1)[0]


def train_loop(toolbox, config, logger, seed, survivor_selection):
    random.seed(seed)
    np.random.seed(seed)
    # create an initial population of POP_SIZE individuals
    # (where each individual is a neural net)
    pop = toolbox.population(n=config.train.pop_size)

    # Variable keeping track of the number of generations
    g = 0

    print("Start of evolution")

    # Evaluate and update fitness for the entire population
    update_fitness(toolbox.evaluate, pop)

    # Extracting all the fitnesses of
    fits = [ind.fitness.values[0] for ind in pop]

    print_statistics(fits, len(pop), len(pop))

    # save gen, max, mean, std
    logger.gather_line(fits, g, survivor_selection)

    # Begin the evolution
    while g < config.train.num_gens:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.parent_select(pop, config.evolve.lambda_coeff * len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Shuffle the offspring
        random.shuffle(offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            # cross two individuals with probability CXPB
            if random.random() < config.evolve.cross_prob:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < config.evolve.mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        update_fitness(toolbox.evaluate, invalid_ind)
        if config.evolve.selection_strategy == "comma":
            pop_len = len(pop)
            pop[:] = offspring
            pop = toolbox.survivor_select(pop, pop_len)
            pop = list(map(toolbox.clone, pop))
        elif config.evolve.selection_strategy == "plus":
            pop_len = len(pop)
            pop[:] = pop + offspring
            pop = toolbox.survivor_select(pop, pop_len)
            pop = list(map(toolbox.clone, pop))

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        print_statistics(fits, len(invalid_ind), len(pop))

        # save gen, max, mean
        # logger.gather_line(fits, g, survivor_selection) # resulted in an error all of a sudden

    print("-- End of (successful) evolution --")
    return tools.selBest(pop, 1)[0]


def update_fitness(eval_func, pop):
    # cpu_count = multiprocessing.cpu_count() - 1
    # with multiprocessing.Pool(processes=cpu_count) as pool:
    #     fitnesses = pool.map(eval_func, pop)
    fitnesses = map(eval_func, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    return fitnesses


def print_statistics(fits, len_evaluated, len_pop):
    print("  Evaluated %i individuals" % len_evaluated)
    mean = sum(fits) / len_pop
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / len_pop - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)


if __name__ == "__main__":
    main()
