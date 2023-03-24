from env_wrapper2 import SelfPLayWrapper

# Choose the env at the top
#**************************

#from keras_gym_env import ConnectFourEnv as GameEnv
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
iterations = 10001
INNER_ITS = 50 *8
BATCH_SIZE = 64 #512
#reward_function_adapting_agent = lambda d,r: tf.where(d, tf.where(r==0.0,tf.constant(1.0),tf.constant(0.0)), r)
epsilon = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.998
POLYAK = 0.9
dropout_rate = 0.2
normalisation = True
BATCH_SIZE_SAMPLING = 512
SAMPLING = 2
AGENT_NUMBER = 1 # how many agents will play against each other while training

#Subfolder for Logs
config_name = "test"
#createsummary writer for vusalization in tensorboard    
time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# time_string = ""

agents = []
train_writer = []
test_writer = []
env = SelfPLayWrapper(GameEnv)

for agent in range(1,AGENT_NUMBER+1):

    best_train_path = f"logs/{config_name}/{time_string}/best_train{agent}"
    train_writer.append(tf.summary.create_file_writer(best_train_path))

    best_test_path = f"logs/{config_name}/{time_string}/best_test{agent}"
    test_writer.append(tf.summary.create_file_writer(best_test_path))

    model_path_best = f"model/{config_name}/{time_string}/best{agent}"

    # create buffer
    best_buffer = Buffer(capacity = 100000,min_size = 5000)

    # create agent
    agents.append(DQNAgent(env,best_buffer, batch = BATCH_SIZE, model_path = model_path_best, polyak_update = POLYAK, inner_iterations = INNER_ITS, dropout_rate = dropout_rate, normalisation = normalisation))

train_self_play_best(agents, GameEnv, BATCH_SIZE_SAMPLING, iterations, train_writer, test_writer, epsilon= epsilon, epsilon_decay = EPSILON_DECAY,epsilon_min = EPSILON_MIN, sampling = SAMPLING)

print("done")
