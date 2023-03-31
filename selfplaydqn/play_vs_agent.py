
""" This file lets you play against an agent, default is our best TikTakToe agent """
from envs.env_wrapper2 import SelfPLayWrapper
from agentmodule.agent import DQNAgent
import random as rnd
import tensorflow as tf

# Choose game env:
# ****************
from envs.tiktaktoe_env import TikTakToeEnv
from envs.keras_gym_env import ConnectFourEnv
GameEnv = TikTakToeEnv

# code
#*******


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

model_path_best = f"model/{config_name}/{time_string}/best{agent}"

# create agent
env = SelfPLayWrapper(GameEnv)
best_agent =  DQNAgent(env,
        None, 
        batch = 1, 
        model_path = model_path_best, 
        conv_kernel = CONV_KERNEL,
        filters = FILTERS,
        hidden_units = HIDDEN_UNITS,
        output_activation=output_activation)

best_agent.load_models(index)
env.set_opponent(best_agent)

# playing loop
while(True):

    print()
    print("New Game")

    env.reset()

    player = rnd.randint(0,1) 
    done=False

    print("Start Player ", player)
    if player: # if opponent starts
        env.opponent_starts()
    env.render()

    while(True):

        print("Your turn ")

        # choose action
        input_action = input()
        if input_action == "stop":
            print("The program will be stopped")
            exit()
        while(int(input_action) not in env.available_actions):
            print("This action is not valid. Please try again. ")
            input_action = input()
        
        # do step, opponent is done automatically inside
        state, r, done = env.step(int(input_action))
        env.render()

        if(done):
            end = "won" if r==env.win_reward else "lost"
            print("You ", end) if r != env.draw_reward else print("Draw")
            break

        player = int(not player)