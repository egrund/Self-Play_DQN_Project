""" Using this file, you can play TikTakToe (or any other env with similar structure) against yourself or another person. """

from envs.tiktaktoe_env import TikTakToeEnv
from envs.keras_gym_env import ConnectFourEnv

# change which environment here
env = ConnectFourEnv()

state = env.reset()
player = 0

while(True):

    print()
    print("New Game")
    env.reset()
    done=False

    while(True):

        print("Turn Player ", player)
        env.render()

        input_action = input()
        if input_action == "stop":
            print("The program will be stopped")
            exit()
        while(int(input_action) not in env.available_actions):
            print("This action is not valid. Please try again. ")
            input_action = input()

        state, r, done = env.step(int(input_action))
        
        if(done):
            print("Player ", player, " wins") if r == 1 else print("Draw")
            env.render()
            break

    player = int(not player)
