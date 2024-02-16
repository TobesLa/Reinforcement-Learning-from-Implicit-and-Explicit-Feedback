import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# This code plots the results from the exp_agents_rewards experiment

env_names = ['CliffWalking-v0', 'FrozenLake-v1']
feedback_variants = ['var1', 'var2', 'var3', 'var4', 'standard']
colors = ['blue', 'darkgreen', 'orange', 'magenta', 'black']
optimal_color = 'cyan'
optimal_froze = 1
optimal_cliff = -13
plots_directory = 'Plots'
results_directory = 'Results'
num_episodes_cliff = 1000
num_episodes_froze = 10000
cliff_array = np.arange(0, num_episodes_cliff)
froze_array = np.arange(0, num_episodes_froze)

if not os.path.exists(plots_directory):
    print(f"Creating directory: {plots_directory}")
    os.makedirs(plots_directory)


cliff_rewards = []
froze_rewards = []
for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]
    file_path = os.path.join(results_directory, f"{env_name}_agents_rewards_result.pkl")

    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    # Access the loaded data using dynamic keys
    for feedback_var in feedback_variants:
        loaded_all_rewards = loaded_data.get(f'all_true_rewards_{env_name}', {}).get(feedback_var, None)
        # Print or analyze the loaded data
        print(f'Loaded all_rewards for {env_name} and {feedback_var}: {loaded_all_rewards}')
        if env_name == env_names[0]:
            cliff_rewards.append(loaded_all_rewards[0])
        else:
            froze_rewards.append(loaded_all_rewards[0])

# plot of one run for all agents
# during training

plt.figure(figsize=(12, 6))
plt.plot(cliff_array, cliff_rewards[0], label='Feedback Variant 1', color=colors[0], linewidth=2, alpha=0.8)
plt.plot(cliff_array, cliff_rewards[1], label='Feedback Variant 2', color=colors[1], linewidth=2, alpha=0.8)
plt.plot(cliff_array, cliff_rewards[2], label='Feedback Variant 3', color=colors[2], linewidth=2, alpha=0.8)
plt.plot(cliff_array, cliff_rewards[3], label='Feedback Variant 4', color=colors[3], linewidth=2, alpha=0.8)
plt.plot(cliff_array, cliff_rewards[4], label='RL_Agent', color=colors[4], linestyle='-', linewidth=2, alpha=0.8)
plt.xlabel('Number of Episodes')
plt.ylabel('Rewards')
plt.axhline(y=optimal_cliff, color=optimal_color, linestyle='--', label='Optimal Value', zorder=6)
plt.title(f'Rewards received during training in Cliffwalking')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_directory, 'rewards_rec_during_training_cliff.pdf'))
plt.show()
plt.close()

# on trained

plt.figure(figsize=(12, 6))
plt.plot(froze_array, froze_rewards[0], label='Feedback Variant 1', color=colors[0], linestyle='-', linewidth=2, alpha=0.8)
plt.plot(froze_array, froze_rewards[1], label='Feedback Variant 2', color=colors[1], linestyle='-', linewidth=2, alpha=0.8)
plt.plot(froze_array, froze_rewards[2], label='Feedback Variant 3', color=colors[2], linestyle='-', linewidth=2, alpha=0.8)
plt.plot(froze_array, froze_rewards[3], label='Feedback Variant 4', color=colors[3], linestyle='-', linewidth=2, alpha=0.8)
plt.plot(froze_array, froze_rewards[4], label='RL_Agent', color=colors[4], linestyle='-', linewidth=2, alpha=0.8)
plt.xlabel('Number of Episodes')
plt.ylabel('Rewards')
plt.axhline(y=optimal_froze, color=optimal_color, linestyle='--', label='Optimal Value', zorder=6)
plt.title(f'Rewards received during training in Frozenlake')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plots_directory, 'rewards_rec_during_training_froze.pdf'))
plt.show()
plt.close()

