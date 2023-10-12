#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import os

import numpy as np

from evoman.environment import Environment
from evolve.neural_net import NNController, NeuralNetwork
from demo_controller import player_controller

# disable visuals and thus make experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"
# hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

INPUT_SIZE = 20
HIDDEN = 10
OUTPUT_SIZE = 5
EXPERIMENT_NAME = 'nn_test'

if not os.path.exists(EXPERIMENT_NAME):
    os.makedirs(EXPERIMENT_NAME)

# controller = NNController()
# neural_net = NeuralNetwork(INPUT_SIZE, HIDDEN, OUTPUT_SIZE)


# initializes environment for multi objetive mode (generalist)  with static enemy and ai player
env = Environment(experiment_name=EXPERIMENT_NAME,
                  speed="fastest",
                  logs="off",
                  savelogs="no",
                  player_controller=player_controller(10),
                  visuals=False)

# tests saved demo solutions for each enemy
# neural_net.load_weights(os.path.join(EXPERIMENT_NAME, 'weights_all.txt'))
print('\n LOADING SAVED SPECIALIST DEAP SOLUTION FOR ALL ENEMEIES \n')
directory = 'island_test'
sol = np.loadtxt(os.path.join(directory, 'island_gain_69.87500000000009.txt'))
num_of_defeated_enemies = 0
total_fitness = 0
total_gain = 0
for en in range(1, 9):
    # Update the enemy
    env.update_parameter('enemies', [en])

    f, p, e, t = env.play(sol)
    total_fitness += f
    total_gain += p - e
    if e == 0:
        print("Enemy " + str(en) + " defeated!\tGain: " + str(p - e))
        num_of_defeated_enemies += 1
    else:
        print("Enemy " + str(en) + " not defeated!\tGain: " + str(p - e))

print('\nTotal Firness: ' + str(total_fitness) + '\nTotal Gain: ' + str(total_gain))
print('\nAverage Fitness: ' + str(total_fitness / 8) + '\nAverage Gain: ' +
      str(total_gain / 8) + '\n\nNumber of defeated enemies: ' + str(num_of_defeated_enemies))
print('=====================================================================')

# for weight in os.listdir(directory):
#     sol = np.loadtxt(os.path.join(directory, weight))
#     num_of_defeated_enemies = 0
#     total_fitness = 0
#     total_gain = 0
#     for en in range(1, 9):
#         # Update the enemy
#         env.update_parameter('enemies', [en])
#
#         f, p, e, t = env.play(sol)
#         total_fitness += f
#         total_gain += p - e
#         if e == 0:
#             print("Enemy " + str(en) + " defeated!\tGain: " + str(p - e))
#             num_of_defeated_enemies += 1
#         else:
#             print("Enemy " + str(en) + " not defeated!\tGain: " + str(p - e))
#
#     print('\nTotal Firness: ' + str(total_fitness) + '\nTotal Gain: ' + str(total_gain))
#     print('\nAverage Fitness: ' + str(total_fitness / 8) + '\nAverage Gain: ' +
#           str(total_gain / 8) + '\n\nNumber of defeated enemies: ' + str(num_of_defeated_enemies))
#     if num_of_defeated_enemies >= 6:
#         print(weight)
#     print('=====================================================================')
