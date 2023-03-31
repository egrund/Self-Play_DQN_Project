from envs.env_wrapper2 import SelfPLayWrapper

# decide env
from envs.tiktaktoe_env import TikTakToeEnv as GameEnv

import numpy as np
import datetime
import tensorflow as tf
import random as rnd

from agentmodule.agent import DQNAgent
from agentmodule.buffer import Buffer
from agentmodule.training import train_self_play_best

# seeds
seed = 42
#np.random.seed(seed)
#rnd.seed(seed) # otherwise always the same player starts
#tf.random.set_seed(seed)

#Subfolder from model
config_name = "test"
time_string = "20230327-191103"
agent = 1

model_path_best = f"model/{config_name}/{time_string}/best{agent}"

# Hyperparameter
#*****************
iterations = 10001
INNER_ITS = 50 *2
BATCH_SIZE = 256 #512
#reward_function_adapting_agent = lambda d,r: tf.where(r==-0.1, tf.constant(0.1), tf.where(r==0.0,tf.constant(1.0),tf.where(r==1.0,tf.constant(-1.0), r)))

epsilon = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.99
opponent_epsilon_function = lambda x: (x/2)

POLYAK = 0.9
dropout_rate = 0
normalisation = True

BATCH_SIZE_SAMPLING = 512
SAMPLING = 2
AGENT_NUMBER = 1 # how many agents will play against each other while training
discount_factor_gamma = tf.constant(0.3)
unavailable_action_reward = False

# Model architecture
#********************
CONV_KERNEL = [3]
FILTERS = 128
HIDDEN_UNITS = [64,]
loss = tf.keras.losses.MeanSquaredError()
output_activation = None

# playing hyperparameter
index = 1200

# create buffer
best_buffer = Buffer(capacity = 100000,min_size = 5000)

# create agent
env = SelfPLayWrapper(GameEnv)
best_agent =  DQNAgent(env,
        best_buffer, 
        batch = BATCH_SIZE, 
        model_path = model_path_best, 
        polyak_update = POLYAK, 
        inner_iterations = INNER_ITS, 
        conv_kernel = CONV_KERNEL,
        filters = FILTERS,
        hidden_units = HIDDEN_UNITS,
        dropout_rate = dropout_rate, 
        normalisation = normalisation, 
        gamma = discount_factor_gamma,
        loss_function=loss,
        output_activation=output_activation)

best_agent.load_models(index)
env.set_opponent(best_agent)

while(True):

    print()
    print("New Game")

    env.reset()

    player = rnd.randint(0,1) 
    done=False

    print("Start Player ", player)
    if player: # if opponent starts
        env.opponent_starts()
    env.render()

    while(True):

        print("Your turn ")

        # choose action
        input_action = int(input())
        while(input_action not in env.available_actions):
            print("This action is not valid. Please try again. ")
            input_action = int(input())
        
        # do step, opponent is done automatically inside
        state, r, done = env.step(input_action)
        env.render()
        
        if(done):
            end = "won" if r==env.win_reward else "lost"
            print("You ", end) if r != env.draw_reward else print("Draw")
            break

        player = int(not player)