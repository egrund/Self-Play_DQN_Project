from keras_gym_env import ConnectFourEnv
import numpy as np
import datetime
import tensorflow as tf
import random as rnd

from env_wrapper2 import SelfPLayWrapper
from tiktaktoe_env import TikTakToeEnv 

from agent import DQNAgent
from buffer import Buffer
from training import train_self_play_best
from testing import testing, testing_dif_agents

### from env_wrapper import ConnectFourSelfPLay
# seeds
seed = 42
np.random.seed(seed)
rnd.seed(seed)
tf.random.set_seed(seed)

#Subfolder from model
config_list = ["3agents_linear/20230328-212250/best1","3agents_linear/20230328-212250/best2","3agents_linear/20230328-212250/best3", "agent_linear_decay099/20230327-185908/best1", "AgentTanH/20230327-192241/best1", "best_agent_tiktaktoe_0998/20230327-191103/best1"]
config_indices = [[900,1200,1150],[1050, 1250],[800,1400],[5250,5550,5300,5400,5800], [250,1150,1300],[3300,3400,3600,3700]]
#config_name = "best_agent_tiktaktoe_0998"
#time_string = "20230327-191103"
agent = 1
#model_path_best = f"model/{config_name}/{time_string}/best{agent}"

model_start_path = f"model/"

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
#BEST_INDEX = 3400

# create buffer
best_buffer = Buffer(capacity = 100000,min_size = 5000)

# create agent
#env = ConnectFourEnv()
env = SelfPLayWrapper(TikTakToeEnv)
agent = DQNAgent(env,
        best_buffer, 
        batch = BATCH_SIZE, 
        model_path = "", #model_path_best, 
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
#agent.load_models(BEST_INDEX)

# Testing Hyperparameter
AV = 10000 # how many games to play for each model to test
# from which iterations to load the models
LOAD = (0,500+1,100) # start,stop,step

rewards = testing_dif_agents(agent,
                             TikTakToeEnv, 
                             size=AV,
                             load = (config_list, config_indices),
                             plot=True)

print("done")