from keras_gym_env import ConnectFourEnv
import numpy as np
import datetime
import tensorflow as tf
import random as rnd

from env_wrapper2 import SelfPLayWrapper
from tiktaktoe_env import TikTakToeEnv 

from agent import DQNAgent, AdaptingDQNAgent, AdaptingAgent
from buffer import Buffer
from training import train_self_play_best
from testing import testing_adapting_dif_epsilon_opponents

### from env_wrapper import ConnectFourSelfPLay
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
opponent_epsilon_function = lambda x: np.random.uniform((x/2 if x > 0.01 else 0), 1)

POLYAK = 0.9
dropout_rate = 0
normalisation = True

BATCH_SIZE_SAMPLING = 512
SAMPLING = 2
AGENT_NUMBER = 1 # how many agents will play against each other while training
discount_factor_gamma = tf.constant(0.3)
unavailable_action_reward = False
D = 20 # how often to save and test the agent

# Model architecture
#********************
CONV_KERNEL = [3]
FILTERS = 128
HIDDEN_UNITS = [64,]
loss = tf.keras.losses.MeanSquaredError()
output_activation = None
BEST_INDEX = 3400

# adapting Model architecture
HIDDEN_UNITS = [64]
loss = tf.keras.losses.MeanSquaredError()
output_activation = None
GAME_BALANCE_MAX = 25

#Subfolder for Logs
config_name = "adapting_test_new"
#createsummary writer for vusalization in tensorboard    
time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# time_string = ""

best_model_path = f"model/best_agent_tiktaktoe_0998/20230327-191103/best1"
model_path = f"model/{config_name}/{time_string}/adapting"

# create agent
#env = ConnectFourEnv()
env = SelfPLayWrapper(TikTakToeEnv)
best_agent = DQNAgent(env,
        None, 
        batch = BATCH_SIZE, 
        model_path = best_model_path, 
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
best_agent.load_models(BEST_INDEX)

adapting_agent = AdaptingAgent(best_agent=best_agent,
                            #env = env, 
                            #buffer = None,
                            #batch = BATCH_SIZE,
                            #model_path=model_path,
                            #polyak_update=POLYAK,
                            #inner_iterations=INNER_ITS,
                            #hidden_units=HIDDEN_UNITS,
                            #gamma = discount_factor_gamma,
                            #loss_function=loss,
                            #output_activation=output_activation,
                            game_balance_max=GAME_BALANCE_MAX)

# Testing Hyperparameter
#**************************
TESTING_SIZE = 100
TESTING_SAMPLING = 10 # how often to sample testing_size many games
OPPONENT_SIZE = 20 # how many different epsilon values will be tested

rewards = testing_adapting_dif_epsilon_opponents(adapting_agent, TikTakToeEnv, best_agent, opponent_size = OPPONENT_SIZE, batch_size=TESTING_SIZE, sampling = TESTING_SAMPLING, plot=True)

print("done")