""" This file trains an adapting DQN agent by letting it play against pretrained normal DQN agents. """

from envs.envwrapper2 import SelfPLayWrapper
from agentmodule.agent import DQNAgent, AdaptingDQNAgent
from agentmodule.buffer import Buffer
from agentmodule.training import train_adapting

import numpy as np
import datetime
import tensorflow as tf
import random as rnd

# Choose the env at the top
#**************************

# from envs.keras_gym_env import ConnectFourEnv as GameEnv
# from envs.keras_gym_env_2wins import ConnectFourEnv2Wins as GameEnv
# from envs.keras_gym_env_novertical import ConnectFourEnvNoVertical as GameEnv
from envs.tiktaktoe_env import TikTakToeEnv as GameEnv

# seeds
seed = 42
np.random.seed(seed)
rnd.seed(seed)
tf.random.set_seed(seed)

# Hyperparameter
#*****************
iterations = 10001
INNER_ITS = 50 *2
BATCH_SIZE = 256

epsilon = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.998
opponent_epsilon_function = lambda x: np.random.uniform((x/2 if x > 0.01 else 0), 1)

POLYAK = 0.9
dropout_rate = 0
normalisation = True

BATCH_SIZE_SAMPLING = 80*6 # has to fit with the opponents size, because they will be broadcasted
SAMPLING = 2
discount_factor_gamma = tf.constant(0.9)
unavailable_action_reward = False
D = 20 # how often to save and test the agent

# we will load models range(step, max, step) make sure these are all existing
# also adapt BATCH_SIZE_SAMPLING so BATCH_SIZE_SAMPLING % (OPPONENT_MAX_INDEX / OPPONENT_STEP) == 0
# because the two values have to be broadcasted together
OPPONENT_MAX_INDEX = 5300 #4000
OPPONENT_STEP = 5300 #50
TESTING_SIZE = 100
TESTING_SAMPLING = 10 # how often to sample testing_size many games

# best Model architecture
#********************
CONV_KERNEL = [3]
FILTERS = 128
HIDDEN_UNITS_BEST = [64]
loss_best = tf.keras.losses.MeanSquaredError()
output_activation_best = None
BEST_INDEX = 5400
best_model_path = f"model/agent_linear_decay099/20230327-185908/best1"

# adapting Model architecture
HIDDEN_UNITS = [64]
loss = tf.keras.losses.MeanSquaredError()
output_activation = None
GAME_BALANCE_MAX = 25

#Subfolder for Logs
config_name = "adapting_test"
#createsummary writer for vusalization in tensorboard    
time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

writer_path = f"logs/{config_name}/{time_string}/adapting_{config_name}_{time_string}_training"
writer_train = tf.summary.create_file_writer(writer_path)
writer_path = f"logs/{config_name}/{time_string}/adapting_{config_name}_{time_string}_testing"
writer_test = tf.summary.create_file_writer(writer_path)

model_path = f"model/{config_name}/{time_string}/adapting"

# create buffer
adapting_buffer = Buffer(capacity = 100000,min_size = 5000)

# create agent
env = SelfPLayWrapper(GameEnv)
best_agent = DQNAgent(env,
        None, 
        batch = BATCH_SIZE, 
        model_path = best_model_path, 
        polyak_update = POLYAK, 
        inner_iterations = INNER_ITS, 
        conv_kernel = CONV_KERNEL,
        filters = FILTERS,
        hidden_units = HIDDEN_UNITS_BEST,
        dropout_rate = dropout_rate, 
        normalisation = normalisation, 
        gamma = discount_factor_gamma,
        loss_function=loss_best,
        output_activation=output_activation_best)
best_agent.load_models(BEST_INDEX)
best_agent.model_path = f"model/{config_name}/{time_string}/best"
best_agent.save_models(0)

adapting_agent = AdaptingDQNAgent(best_agent=best_agent,
                                  env = env, 
                                  buffer = adapting_buffer,
                                  batch = BATCH_SIZE,
                                  model_path=model_path,
                                  polyak_update=POLYAK,
                                  inner_iterations=INNER_ITS,
                                  hidden_units=HIDDEN_UNITS,
                                  gamma = discount_factor_gamma,
                                  loss_function=loss,
                                  output_activation=output_activation,
                                  game_balance_max=GAME_BALANCE_MAX)

opponents = [DQNAgent(env,
        None, 
        batch = BATCH_SIZE, 
        model_path = best_model_path, 
        polyak_update = POLYAK, 
        inner_iterations = INNER_ITS, 
        conv_kernel = CONV_KERNEL,
        filters = FILTERS,
        hidden_units = HIDDEN_UNITS_BEST,
        dropout_rate = dropout_rate, 
        normalisation = normalisation, 
        gamma = discount_factor_gamma,
        loss_function=loss_best,
        output_activation=output_activation_best) for i in range(OPPONENT_STEP,OPPONENT_MAX_INDEX +1,OPPONENT_STEP)]
[agent.load_models(i) for i,agent in zip(range(OPPONENT_STEP,OPPONENT_MAX_INDEX +1,OPPONENT_STEP),opponents)]

train_adapting(adapting_agent, 
        opponents,
        GameEnv, 
        BATCH_SIZE_SAMPLING, 
        iterations, 
        writer_train,
        writer_test,
        epsilon= epsilon, 
        epsilon_decay = EPSILON_DECAY,
        epsilon_min = EPSILON_MIN, 
        sampling = SAMPLING, 
        unavailable_in=unavailable_action_reward,
        opponent_epsilon=opponent_epsilon_function,
        testing_size = TESTING_SIZE,
        testing_sampling = TESTING_SAMPLING)

print("done")