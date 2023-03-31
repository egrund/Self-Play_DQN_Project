""" This file test an agent against an opponent with different epsilon values for its policy. """
import numpy as np    
import tensorflow as tf
import random as rnd
from agentmodule.testing import testing_adapting_dif_epsilon_opponents
from envs.envwrapper2 import SelfPLayWrapper
from agentmodule.agent import DQNAgent

# Choose which Agent to use from the following: all are possible
# **************************
from agentmodule.agent import DQNAgent, AdaptingAgent, AdaptingAgent2, AdaptingAgent3, AdaptingAgent4, AdaptingAgent5, AdaptingDQNAgent
AdaptingAgentToUse = AdaptingAgent5
# is c, or B or cp depending on which agent
calculation_value = tf.constant(4.) # has to be a float
GAME_BALANCE_MAX = 25

# Choose which env to use
# *************************
from envs.tiktaktoe_env import TikTakToeEnv 
from envs.keras_gym_env import ConnectFourEnv
GameEnv = ConnectFourEnv

# seeds
if True:
    seed = 42
    np.random.seed(seed)
    rnd.seed(seed)
    tf.random.set_seed(seed)

if GameEnv == TikTakToeEnv:
    config_name = "agent_linear_decay099"
    time_string = "20230327-185908"
    agent = 1
    # playing hyperparameter
    index = 5300
    # Model architecture
    CONV_KERNEL = [3]
    FILTERS = 128
    HIDDEN_UNITS = [64,]
    output_activation = None

if GameEnv == ConnectFourEnv:
    config_name = "agent_ConnectFour_tanh"
    time_string = "20230328-094318"
    agent = 1
    # playing hyperparameter
    index = 400
    # Model architecture
    CONV_KERNEL = [4,4]
    FILTERS = 128
    HIDDEN_UNITS = [64,64]
    output_activation = tf.nn.tanh

best_model_path = f"model/{config_name}/{time_string}/best1"

# adapting Model architecture if adapting DQN agent
if AdaptingAgentToUse == AdaptingDQNAgent:
    HIDDEN_UNITS_ADAPTING = [64]
    output_activation_adapting = None
    config_name_adapting = "adapting_test_new"
    time_string_adapting = "20230329-152817"
    adapting_index = 0
    adapting_model_path = f"model/{config_name_adapting}/{time_string_adapting}/adapting"

# create agent
#env = ConnectFourEnv()
env = SelfPLayWrapper(GameEnv)
best_agent =  DQNAgent(env,
        None, 
        batch = 1, 
        model_path = best_model_path,
        conv_kernel = CONV_KERNEL,
        filters = FILTERS,
        hidden_units = HIDDEN_UNITS,
        output_activation=output_activation)
best_agent.load_models(index)

adapting = True
if AdaptingAgentToUse == DQNAgent:
    adapting_agent = best_agent
    adapting = False
elif AdaptingAgentToUse == AdaptingAgent:
    adapting_agent = AdaptingAgent(best_agent=best_agent, game_balance_max=GAME_BALANCE_MAX)
elif AdaptingAgentToUse == AdaptingDQNAgent:
    adapting_agent = AdaptingDQNAgent(best_agent=best_agent,
                            env = env, 
                            buffer = None,
                            model_path=adapting_model_path,
                            hidden_units=HIDDEN_UNITS_ADAPTING,
                            output_activation=output_activation_adapting,
                            game_balance_max=GAME_BALANCE_MAX)
    adapting_agent.load_models(adapting_index)
else:
    adapting_agent = AdaptingAgentToUse(best_agent=best_agent,
                                calculation_value = calculation_value,
                                game_balance_max=GAME_BALANCE_MAX)

# Testing Hyperparameter
#**************************
TESTING_SIZE = GAME_BALANCE_MAX # change at game balance max
TESTING_SAMPLING = 1 # how often to sample testing_size many games
OPPONENT_SIZE = 1 # how many different epsilon values will be tested

rewards = testing_adapting_dif_epsilon_opponents(adapting_agent, 
                                                 GameEnv, 
                                                 best_agent, 
                                                 opponent_size = OPPONENT_SIZE, 
                                                 batch_size=TESTING_SIZE, 
                                                 sampling = TESTING_SAMPLING, 
                                                 printing = False,
                                                 plot=True,
                                                 adapting = adapting) # False when we use a DQNAgent

print("done")