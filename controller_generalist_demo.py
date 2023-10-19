#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                                 		  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        general solution for enemies (games)                                         #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np

# disable visuals and thus make experiments faster
os.environ["SDL_VIDEODRIVER"] = "dummy"
# hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

experiment_name = "controller_generalist_demo"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for multi objetive mode (generalist)  with static enemy and ai player
env = Environment(
    experiment_name=experiment_name,
    speed="fastest",
    logs="off",
    savelogs="no",
    player_controller=player_controller(10),
    visuals=False
)

directory = "optimization_generalist_island"
# sol = np.loadtxt("nn_test/weights.txt")
print("\n LOADING SAVED GENERALIST SOLUTION FOR ALL ENEMIES \n")

for weights in os.listdir(directory):
    if weights.endswith(".txt"):
        sol = np.loadtxt(os.path.join(directory, weights))
        num_of_defeated_enemies = 0
        total_fitness = 0
        total_gain = 0
        # tests saved demo solutions for each enemy
        for en in range(1, 9):
            # Update the enemy
            env.update_parameter("enemies", [en])

            f, p, e, t = env.play(sol)
            total_fitness += f
            total_gain += p - e
            if e == 0:
                print("Enemy " + str(en) + " defeated!\tGain: " + str(p - e))
                num_of_defeated_enemies += 1
            else:
                print("Enemy " + str(en) + " not defeated!\tGain: " + str(p - e))

        print("\nTotal Firness: " + str(total_fitness) + "\nTotal Gain: " + str(total_gain))
        print("\nAverage Fitness: " + str(total_fitness / 4) + "\nAverage Gain: " + str(total_gain / 8))
        print("\nNumber of defeated enemies: " + str(num_of_defeated_enemies))
        print(weights)
        print("=====================================================")
        print("\n\n")
print("\n  \n")
