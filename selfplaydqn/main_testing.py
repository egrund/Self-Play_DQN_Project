""" 
This file is a main that tests one or several DQN Agents. If you want to switch between different usages, you might have to remove some parts or uncomment others. 

- use config_list, config_indices and testing_dif_agents if the agents are from different runs
- use config_name, time_string, agent, LOAD and testing if the agents are from the same run and different interations
- if only testing one agent use config_name, time_string, agent, BEST_INDEX and load_models!!! 
"""

import numpy as np
import tensorflow as tf
import random as rnd

from envs.envwrapper2 import SelfPLayWrapper
from envs.tiktaktoe_env import TikTakToeEnv 
from envs.keras_gym_env import ConnectFourEnv

from agentmodule.agent import DQNAgent
from agentmodule.buffer import Buffer
from agentmodule.testing import testing, testing_dif_agents

# seeds
seed = 42
np.random.seed(seed)
rnd.seed(seed)
tf.random.set_seed(seed)

#Subfolder from model
model_start_path = f"model/"
model_path_best = ""
config_list = ["3agents_linear/20230328-212250/best1","3agents_linear/20230328-212250/best2","3agents_linear/20230328-212250/best3", "agent_linear_decay099/20230327-185908/best1", "AgentTanH/20230327-192241/best1", "best_agent_tiktaktoe_0998/20230327-191103/best1"]
config_indices = [[900,1200,1150],[1050, 1250],[800,1400],[5250,5550,5300,5400,5800], [250,1150,1300],[3300,3400,3600,3700]]

#config_name = "best_agent_tiktaktoe_0998"
#time_string = "20230327-191103"
agent = 1
#model_path_best = f"model/{config_name}/{time_string}/best{agent}"

# Model architecture
#********************
CONV_KERNEL = [3]
FILTERS = 128
HIDDEN_UNITS = [64,]
loss = tf.keras.losses.MeanSquaredError()
output_activation = None
#BEST_INDEX = 3400 # if testing only one agent

# create buffer
best_buffer = Buffer(capacity = 100000,min_size = 5000)

# create agent
env = SelfPLayWrapper(TikTakToeEnv)
agent = DQNAgent(env,
        best_buffer, 
        batch = 1, 
        model_path = model_path_best, 
        conv_kernel = CONV_KERNEL,
        filters = FILTERS,
        hidden_units = HIDDEN_UNITS,
        output_activation=output_activation)
#agent.load_models(BEST_INDEX) # if testing only one agent

# Testing Hyperparameter
AV = 10000 # how many games to play for each model to test
# from which iterations to load the models, if testing models from the same run
LOAD = (0,500+1,100) # start,stop,step

rewards = testing_dif_agents(agent,
                             TikTakToeEnv, 
                             size=AV,
                             load = (config_list, config_indices),
                             plot=True)

print("done")