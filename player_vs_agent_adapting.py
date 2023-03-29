from env_wrapper2 import SelfPLayWrapper

# decide env
from tiktaktoe_env import TikTakToeEnv as GameEnv

import numpy as np
import datetime
import tensorflow as tf
import random as rnd


from agent import DQNAgent, AdaptingDQNAgent, AdaptingAgent3, Agent
from buffer import Buffer
from training import train_self_play_best

# seeds
seed = 42
#np.random.seed(seed)
#rnd.seed(seed) # otherwise always the same player starts
#tf.random.set_seed(seed)

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

# best Model architecture
#********************
CONV_KERNEL = [3]
FILTERS = 128
HIDDEN_UNITS_BEST = [64]
loss_best = tf.keras.losses.MeanSquaredError()
output_activation_best = None

# adapting Model architecture
HIDDEN_UNITS = [64]
loss = tf.keras.losses.MeanSquaredError()
output_activation = None
GAME_BALANCE_MAX = 25

#Subfolder for Logs
config_name_adapting = "best_agent_tiktaktoe_0998"  
time_string_adapting = "20230327-191103"
model_path = f"model/{config_name_adapting}/{time_string_adapting}"
model_path_adapting = model_path + "/adapting"
# playing hyperparameter for adapting agent
playing_index = 180

# create agent
env = SelfPLayWrapper(GameEnv)
best_agent =  DQNAgent(env,
        None, 
        batch = BATCH_SIZE, 
        model_path = model_path + "/best1",
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
best_agent.load_models(3400)

adapting_agent = AdaptingAgent3(best_agent=best_agent,
                                calculation_value = tf.constant(0.5),
                                #env = env, 
                                #buffer = None,
                                #batch = BATCH_SIZE,
                                #model_path=model_path_adapting,
                                #polyak_update=POLYAK,
                                #inner_iterations=INNER_ITS,
                                #hidden_units=HIDDEN_UNITS,
                                #gamma = discount_factor_gamma,
                                #loss_function=loss,
                                #output_activation=output_activation,
                                game_balance_max=GAME_BALANCE_MAX)
#adapting_agent.load_models(playing_index)

env.set_opponent(adapting_agent)

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
            # give agent information about game ending
            adapting_agent.add_game_balance_information([r])
            break

        player = int(not player)