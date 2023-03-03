from keras_gym_env import ConnectFourEnv

from agent import DQNAgent
from buffer import Buffer

# Hyperparameter


# create environment
env = ConnectFourEnv()
actions = env.available_actions

# create buffer
buffer = Buffer()
buffer.fill()

# create agent
agent = DQNAgent(env,buffer)