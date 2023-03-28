from env_wrapper2 import SelfPLayWrapper

# Choose the env at the top
#**************************

# from keras_gym_env import ConnectFourEnv as GameEnv
# from keras_gym_env_2wins import ConnectFourEnv2Wins as GameEnv
# from keras_gym_env_novertical import ConnectFourEnvNoVertical as GameEnv
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
np.random.seed(seed)
rnd.seed(seed)
tf.random.set_seed(seed)

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
unavailable_action_reward = False # only TikTakToe
D = 20 # how often to save and test the agent

# Model architecture
#********************
CONV_KERNEL = [4,4]
FILTERS = 128
HIDDEN_UNITS = [64,64]
loss = tf.keras.losses.MeanSquaredError()
output_activation = None

#Subfolder for Logs
config_name = "test"
#createsummary writer for vusalization in tensorboard    
time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# time_string = ""

agents = []
writers = []
env = SelfPLayWrapper(GameEnv)

for agent in range(1,AGENT_NUMBER+1):

    best_writer_path = f"logs/{config_name}/{time_string}/best_train_{config_name}_{time_string}_{agent}"
    writers.append(tf.summary.create_file_writer(best_writer_path))

    model_path_best = f"model/{config_name}/{time_string}/best{agent}"

    # create buffer
    best_buffer = Buffer(capacity = 100000,min_size = 5000)

    # create agent
    agents.append(DQNAgent(env,
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
                           output_activation=output_activation))

train_self_play_best(agents, 
                     GameEnv, 
                     BATCH_SIZE_SAMPLING, 
                     iterations, 
                     writers,
                     epsilon= epsilon, 
                     epsilon_decay = EPSILON_DECAY,
                     epsilon_min = EPSILON_MIN, 
                     sampling = SAMPLING, 
                     unavailable_in=unavailable_action_reward,
                     opponent_epsilon=opponent_epsilon_function,
                     d = D)

print("done")
