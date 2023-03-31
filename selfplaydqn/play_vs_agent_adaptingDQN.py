""" This file lets you play against an agent, default is our best TikTakToe agent """

from envs.envwrapper2 import SelfPLayWrapper
from agentmodule.agent import DQNAgent, AdaptingDQNAgent
import tensorflow as tf
import random as rnd

from envs.tiktaktoe_env import TikTakToeEnv as GameEnv


# Model architecture
CONV_KERNEL = [3]
FILTERS = 128
HIDDEN_UNITS = [64,]
output_activation = None

# adapting Model architecture
HIDDEN_UNITS_ADAPTING = [64]
output_activation_adapting = None
GAME_BALANCE_MAX = 25


#Subfolder for Logs
config_name_adapting = "adapting_test_new"
time_string_adapting = "20230329-152817"
model_path = f"model/{config_name_adapting}/{time_string_adapting}"
index_adapting = 180


# create agent
env = SelfPLayWrapper(GameEnv)
best_agent =  DQNAgent(env,
        None, 
        batch = 1, 
        model_path = model_path + "/best",
        conv_kernel = CONV_KERNEL,
        filters = FILTERS,
        hidden_units = HIDDEN_UNITS,
        output_activation=output_activation)
best_agent.load_models(0)

adapting_agent = AdaptingDQNAgent(best_agent=best_agent,
                            calculation_value = tf.constant(0.5),
                            env = env, 
                            buffer = None,
                            batch = 1,
                            model_path=model_path + "/adapting",
                            hidden_units=HIDDEN_UNITS_ADAPTING,
                            output_activation=output_activation_adapting,
                            game_balance_max=GAME_BALANCE_MAX)
adapting_agent.load_models(index_adapting)

env.set_opponent(adapting_agent)

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
            # give agent information about game ending
            adapting_agent.add_game_balance_information([r])
            break

        player = int(not player)