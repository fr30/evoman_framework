# imports framework
import neat
import os
import pickle

from evoman.environment import Environment
from controller_neat import PlayerControllerNeat


# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'optimization_neat'


# initializes simulation in multi evolution mode, for multiple static enemies.
env = Environment(experiment_name=experiment_name,
                  multiplemode="yes",
                  enemies=[1, 2, 3, 4, 5, 7, 8],
                  speed="fastest",
                  logs="off",
                  savelogs="no",
                  player_controller=PlayerControllerNeat(),  # you  can insert your own controller here
                  visuals=False)

# start writing your own code from here

# global variable to keep track of the generation
gen = 0


# runs simulation
def simulation(env, x):
    f, _, _, _ = env.play(pcont=x)
    return f


# evaluation
def eval_genomes(genomes, config):
    global gen
    gen += 1
    nets = []
    ge = []
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)
        genome.fitness = simulation(env, net)
        # Uncomment to see the fitness of each genome in each generation
        # print("Gen: ", gen, "Fitness: ", genome.fitness)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 100 generations.
    winner = p.run(eval_genomes, 100)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner))

    # Save the best genome in winner_neat_....pkl
    pickle.dump(winner, open('neat/winner_neat_all.pkl', 'wb'))


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward_neat.txt')
    run(config_path)
