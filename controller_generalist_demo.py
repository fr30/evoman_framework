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

experiment_name = 'controller_generalist_demo'
# if not os.path.exists(experiment_name):
#     os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for multi objetive mode (generalist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  speed="fastest",
                  logs="off",
                  savelogs="no",
                  player_controller=player_controller(n_hidden_neurons),
                  visuals=True)

sol = np.loadtxt('multi_demo/best.txt')
print('\n LOADING SAVED GENERALIST SOLUTION FOR ALL ENEMIES \n')
total_fitness = 0
total_gain = 0
# tests saved demo solutions for each enemy
for en in range(1, 9):
    # Update the enemy
    env.update_parameter('enemies', [en])

    f, p, e, t = env.play(sol)
    total_fitness += f
    total_gain += p - e
    print("Enemy " + str(en) + " Fitness: " + str(f) + " Gain: " + str(p - e))
    if e == 0:
        print("Defeated!")

print('Total Fitness: ' + str(total_fitness) + ' Total Gain: ' + str(total_gain) + '\n\n')
print('\n  \n')
