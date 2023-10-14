import numpy as np
import matplotlib.pyplot as plt
import class_maze as cm
import seaborn as sns

# Carrega dados
data = np.load('RL_RobotControl/sarsa_qtable.npy')
rewards = np.load('RL_RobotControl/reward_file.npy')
avg_rewards = np.load('RL_RobotControl/avg_rewards_file.npy')
Q = data['Q']
episodes = data['episodes']

# Carrega classe para plotar
env = cm.Maze()
env.reset()

# plot rewards
plt.rcParams['figure.figsize'] = (16,8)
plt.subplot(1, 2, 2)
plt.gca().clear()
plt.gca().set_box_aspect(.5)
plt.title('Recompensa por episódios')
plt.plot(avg_rewards, 'b', linewidth=2)
plt.plot(rewards, 'r', alpha=0.3)
plt.xlabel('Episódios')
plt.ylabel('Recompensa')

# Plot ambiente final
plt.subplot(1, 2, 1)
plt.gca().clear()
env.render(Q)