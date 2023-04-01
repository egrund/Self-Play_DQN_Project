# Self-Play_DQN_Project
IANNWTF Project WS22/23

The project was created using Python 3.10.8. The package requirements can be found in [requirements.txt](requirements.txt). It is our final project for the course 'Implementing Artificial Neural Networks in Tensorflow' given in the winter semester 22/23 at Osnabrück University. 

We trained DQN agents to play TikTakToe and ConnectFour using self-play.

# Group:
* [Eosandra Grund](https://github.com/egrund/): egrund@uni-osnabrueck.de
* [Fabian Kirsch](https://github.com/Kirschberg32/): fkirsch@uni-osnabrueck.de

# Files: 

* [Plots](Plots): Folder with different Plots from runs
  * [best_agents_plots](Plots/best_agents_plots):
  * [connectfour_testing](Plots/connectfour_testing):
  * [tiktaktoe_testing](Plots/tiktaktoe_testing):
* [logs](logs): Log-Files created by different
  * [3agents_linear/20230328-212250](logs/3agents_linear/20230328-212250):
  * [AgentTanH/20230327-192241](logs/AgentTanH/20230327-192241):
  * [ConnectFour_linear/20230330-174353/](logs/ConnectFour_linear/20230330-174353):
  * [agent_ConnectFour_tanh/20230328-094318](logs/agent_ConnectFour_tanh/20230328-094318):
  * [agent_linear_decay099/20230327-185908](logs/agent_linear_decay099/20230327-185908):
  * [best_agent_tiktaktoe_0998/20230327-191103](logs/best_agent_tiktaktoe_0998/20230327-191103):
* [model](model): Saved models of different runs. 
  * [3agents_linear/20230328-212250](model/3agents_linear/20230328-212250):
  * [AgentTanH/20230327-192241](model/AgentTanH/20230327-192241):
  * [ConnectFour_linear/20230330-174353/](model/ConnectFour_linear/20230330-174353):
  * [agent_ConnectFour_tanh/20230328-094318](model/agent_ConnectFour_tanh/20230328-094318):
  * [agent_linear_decay099/20230327-185908](model/agent_linear_decay099/20230327-185908):
  * [best_agent_tiktaktoe_0998/20230327-191103](model/best_agent_tiktaktoe_0998/20230327-191103): 
* [selfplaydqn](selfplaydqn): Entails all Python files
  * [agentmodule](selfplaydqn/agentmodule)
    * [__init__.py](selfplaydqn/agentmodule/__init__.py)
    * [agent.py](selfplaydqn/agentmodule/__init__.py)
    * [buffer.py](selfplaydqn/agentmodule/__init__.py)
    * [model.py](selfplaydqn/agentmodule/__init__.py)
    * [testing.py](selfplaydqn/agentmodule/__init__.py)
    * [training.py](selfplaydqn/agentmodule/__init__.py)
  * [envs](selfplaydqn/envs)
    * [__init__.py](selfplaydqn/envs/__init__.py)
    * [envwrapper2.py](selfplaydqn/envs/envwrapper2.py)
    * [keras_gym_env.py](selfplaydqn/envs/keras_gym_env.py)
    * [keras_gym_env_2wins.py ](selfplaydqn/envs/keras_gym_env_2wins.py )
    * [keras_gym_env_novertical.py](selfplaydqn/envs/keras_gym_env_novertical.py)
    * [sampler.py](selfplaydqn/envs/sampler.py)
    * [tiktaktoe_env.py](selfplaydqn/envs/tiktaktoe_env.py)
  * [__init__.py](selfplaydqn/__init__.py):
  * [main_testing.py](selfplaydqn/main_testing.py):
  * [main_testing_adapting.py](selfplaydqn/main_testing_adapting.py):
  * [main_train_best.py](selfplaydqn/main_train_best.py):
  * [play_vs_agent.py](selfplaydqn/play_vs_agent.py):
  * [play_vs_agent_adapting.py](selfplaydqn/play_vs_agent_adapting.py):
  * [play_vs_person.py](selfplaydqn/play_vs_person.py):
  * [plot_tensorboard_data.py](selfplaydqn/plot_tensorboard_data.py):
* [text_info_runs](text_info_runs): data created by testing different agents
  * [ConnectFour.txt](text_info_runs/ConnectFour.txt):
  * [best_agent_ConncetFour_comparison.txt](text_info_runs/best_agent_ConncetFour_comparison.txt):
  * [best_agent_comparison.txt](text_info_runs/best_agent_comparison.txt):
  * [best_model_tiktaktoe2703.txt](text_info_runs/best_model_tiktaktoe2703.txt):
  * [train_adapting_agent.txt](text_info_runs/train_adapting_agent.txt):
* [requirements.txt](requirements.txt): Required packages for the project
