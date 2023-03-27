from env_wrapper2 import SelfPLayWrapper

# decide env
from tiktaktoe_env import TikTakToeEnv as GameEnv

import numpy as np
import datetime
import tensorflow as tf
import random as rnd


from agent import DQNAgent
from buffer import Buffer
from training import train_self_play_best

# seeds
seed = 42
#np.random.seed(seed)
#rnd.seed(seed) # otherwise always the same player starts
#tf.random.set_seed(seed)

#Subfolder from model
config_name = "TikTakToe_dicount_nosquare"
time_string = "20230324-154801"
agent = 1

model_path_best = f"model/{config_name}/{time_string}/best{agent}"

# Hyperparameter
iterations = 5000
INNER_ITS = 50
BATCH_SIZE = 256
#reward_function_adapting_agent = lambda d,r: tf.where(d, tf.where(r==0.0,tf.constant(1.0),tf.constant(0.0)), r)
epsilon = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.998
POLYAK = 0.9
dropout_rate =0 
normalisation = True

# playing hyperparameter
#index = 4260 #3720 #2840 # one model
#index = 780 # three models playing # "3-Agents_TikTakToe" "20230324-005528"
index = 2080 #1760 # "TikTakToe_dicount_nosquare" "20230324-154801"

# create buffer
best_buffer = Buffer(capacity = 100000,min_size = 5000)

# create agent
env = SelfPLayWrapper(GameEnv)
best_agent = DQNAgent(env,best_buffer, batch = BATCH_SIZE, model_path = model_path_best, polyak_update = POLYAK, inner_iterations = INNER_ITS, dropout_rate = dropout_rate, normalisation = normalisation)

best_agent.load_models(index)
env.set_opponent(best_agent)

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
        end = "won" if r==1 else "lost"
        print("You ", end) if r != 0 else print("Draw")
        break

    player = int(not player)