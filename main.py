import tensorflow as tf 
from keras_gym_env import ConnectFourEnv

from agent import DQNAgent

# Hyperparameter


# create environment
env = ConnectFourEnv()
actions = env.available_actions

# create buffer
# buffer = []

# create agent
agent = DQNAgent(env,buffer)