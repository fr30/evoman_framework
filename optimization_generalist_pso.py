import operator
import random
import os
import numpy as np
import math

from deap import base
from deap import creator
from deap import tools

from demo_controller import player_controller
from evoman.environment import Environment

# disable visuals and thus make experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"
# hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

EXPERIMENT_NAME = "nn_test"
ENEMY_IDX = [1, 2, 3, 4, 5, 6, 7, 8]

env = Environment(
    experiment_name=EXPERIMENT_NAME,
    enemies=ENEMY_IDX,
    multiplemode="yes",
    speed="fastest",
    logs="off",
    savelogs="no",
    player_controller=player_controller(10),
    visuals=False,
)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
               smin=None, smax=None, best=None)


def eval_fitness(env, individual):
    f, p, e, t = env.play(pcont=individual)
    return ((p-e), )
    # return (env.play(pcont=individual)[0],)


def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part





def updateParticle(part, best, w, phi1, phi2):
    # u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    # u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    # v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
    # v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
    # part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
    # for i, speed in enumerate(part.speed):
    #     if abs(speed) < part.smin:
    #         part.speed[i] = math.copysign(part.smin, speed)
    #     elif abs(speed) > part.smax:
    #         part.speed[i] = math.copysign(part.smax, speed)
    # part[:] = list(map(operator.add, part, part.speed))


    u1 = np.random.uniform(0, phi1, size=len(part))
    u2 = np.random.uniform(0, phi2, size=len(part))
    v_u1 = u1 * (np.array(part.best) - np.array(part))
    v_u2 = u2 * (np.array(best) - np.array(part))
    part.speed = (w * np.array(part.speed) + v_u1 + v_u2).tolist()
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = math.copysign(part.smin, speed)
        elif abs(speed) > part.smax:
            part.speed[i] = math.copysign(part.smax, speed)
    part[:] = (np.array(part) + np.array(part.speed)).tolist()


toolbox = base.Toolbox()
toolbox.register("particle", generate, 265, pmin=-1, pmax=1, smin=-3, smax=3)
toolbox.register("population", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", eval_fitness)


def main():
    pop = toolbox.population(n=100)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # logbook = tools.Logbook()
    # logbook.header = ["gen", "evals"] + stats.fields

    GEN = 100000
    best = None

    w = 1
    w_dec = 0.0035
    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(env, individual=part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best, w)
        w -= w_dec
        print(g, best.fitness)

        # Gather all the fitnesses in one list and print the stats
        # logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        # print(logbook.stream)
    # return pop, best


if __name__ == "__main__":
    main()
