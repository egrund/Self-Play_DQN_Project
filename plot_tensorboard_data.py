import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mat

# Change hyperparameters
#*************************
path = "model/agent_linear_decay099/20230327-185908/tensorboard_data/"
game  = "TikTakToe"

# for creating a loss plot
"""
files = ["loss"]
plot_title = "Loss " + game
y_value = "Loss"
hue_value = None
point = None
"""

# for creating a reward plot
# """
files = ["reward1", "reward0", "reward-1"]
plot_title = "Rewards " + game
y_value = "Percentage"
hue_value = "Reward"
point =  5300 # The agent you chose to be shown in the plot
# """

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
axes.set(title=plot_title)
if y_value == "Percentage":
    axes.set_ylim(0,100)
    axes.set_yticks(range(0,101,10))
axes.grid(True,color = 'black', linestyle="--",linewidth=0.5, alpha = 0.5)

# add the chosen point to plot
if point != None:
    plt.vlines(5300,axes.get_ylim()[0],axes.get_ylim()[1],colors="red")

plt.show()