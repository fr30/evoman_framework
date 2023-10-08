# imports framework
import neat
import pickle

from evoman.environment import Environment
from controller_neat import PlayerControllerNeat


# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name="controller_generalist_neat",
                  speed="fastest",
                  logs="off",
                  savelogs="no",
                  player_controller=PlayerControllerNeat(),
                  visuals=True)

# Load the best genome saved in the winner_neat_25678.pkl file
winner = pickle.load(open('neat/winner_neat_all.pkl', 'rb'))
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'neat/config-feedforward_neat.txt')
net = neat.nn.FeedForwardNetwork.create(winner, config)
print('\n LOADING SAVED SPECIALIST NEAT SOLUTION FOR ALL ENEMIES \n')

# tests saved neat solutions for selected enemy
for en in range(1, 9):
    # Update the enemy
    env.update_parameter('enemies', [en])

    _, p, e, _ = env.play(pcont=net)
    if e == 0:
        print("Enemy " + str(en) + " defeated!\tGain: " + str(p - e))
print('\n  \n')