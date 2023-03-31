"""
This file was created to make plots from csv files exported from tensoboard. The possible values are loss, or if importing 3 files rewards.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Change hyperparameters
#*************************
path = "model/agent_ConnectFour_tanh/20230328-094318/tensorboard_data/"
game  = "ConnectFour"
which = "reward"

if which == "loss":
    files = ["loss"]
    plot_title = "Loss " + game
    y_value = "Loss"
    hue_value = None
    point = None

if which == "reward":
    files = ["reward1", "reward0", "reward-1"]
    plot_title = "Rewards " + game
    y_value = "Percentage"
    hue_value = "Reward"
    point =  400 # The agent you chose to be shown in the plot

# Code
#*****
dataframe = None
# combine dataframes
if len(files) == 1:
    dataframe = pd.read_csv(path + files[0] + ".csv")
    dataframe.rename(columns={"Value":y_value}, inplace = True)
else:
    df = [pd.read_csv(path + f + ".csv") for f in files]
    [df[i].rename(columns={"Value":y_value}, inplace = True) for i,f in enumerate(files)]
    [df[i].insert(2,hue_value, int(f[len(hue_value):])) for i,f in enumerate(files)]
    dataframe = pd.concat(df)


axes = sns.lineplot(dataframe, linewidth=2, palette= "tab10",x="Step", y=y_value, hue = hue_value)
axes.set_title(plot_title, size="xx-large")
axes.set_xlabel("Step",size="xx-large")
axes.set_ylabel(y_value,size="xx-large")
if hue_value != None:
    axes.legend(prop={'size':15}, title = hue_value, title_fontsize = "xx-large")
if y_value == "Percentage":
    axes.set_ylim(0,100)
    axes.set_yticks(range(0,101,10))
axes.grid(True,color = 'black', linestyle="--",linewidth=0.5, alpha = 0.5)

# add the chosen point to plot
if point != None:
    plt.vlines(point,axes.get_ylim()[0],axes.get_ylim()[1],colors="red")

plt.show()