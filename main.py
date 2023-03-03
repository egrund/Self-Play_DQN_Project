import tensorflow as tf 
# import keras_gym 

from model import MyMLP

# Parameter


# create environment
#env = keras_gym.envs.ConnectFourEnv()
actions = [1,2,3,4,5,6] # env.available_actions

# create model and agent
model = MyMLP(hidden_units = [128,64,32], output_units = len(actions))
