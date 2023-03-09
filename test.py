from keras_gym_env import ConnectFourEnv

env = ConnectFourEnv()

state = env.reset()
player = 0

while(True):

    print("Turn Player ", player)
    env.render()
    print(state)

    input_action = int(input())

    state, r, done, info = env.step(input_action)
    
    if(done):
        print("Player ", player, " wins") if r == 1 else print("Draw")
        env.render()
        print(state)
        break

    player = int(not player)
