import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# This code plots the results from the exp_rl_agent and exp_feedback_agent experiment

env_names = ['CliffWalking-v0', 'FrozenLake-v1']
feedback_variants = ['IBF', 'ABF', 'ERF', 'SRF'] # var1 to var4
colors = ['blue', 'darkgreen', 'orange', 'magenta', 'black']
optimal_color = 'cyan'
optimal_froze = 1
optimal_cliff = -13
num_runs = 10
num_run_array = np.arange(0, num_runs)
num_episodes_array_cliff = np.arange(0, 1000)
num_episodes_array_froze = np.arange(0, 10000)
num_train_results = 100
results_directory = 'Results'
plots_directory = 'Plots'
if not os.path.exists(plots_directory):
    print(f"Creating directory: {plots_directory}")
    os.makedirs(plots_directory)

loaded_rew_during_learn_cliff = []
loaded_all_avg_rewards_cliff = []
loaded_all_std_devs_cliff = []

loaded_rew_during_learn_froze = []
loaded_all_avg_rewards_froze = []
loaded_all_std_devs_froze = []

# Load Data
# feedback agents results
for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]
    file_path = os.path.join(results_directory, f"{env_name}_feedback_agent_results.pkl")

    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    for feedback_var in feedback_variants:
        loaded_all_rewards = loaded_data.get(f'all_rewards_{env_name}', {}).get(feedback_var, None)
        loaded_all_avg_rewards = loaded_data.get(f'all_avg_rewards_{env_name}', {}).get(feedback_var, None)
        loaded_all_std_devs = loaded_data.get(f'all_std_devs_{env_name}', {}).get(feedback_var, None)
        loaded_rew_during_learn = []
        loaded_std_during_learn = []
        if chosen_env == 0:
            loaded_rew_during_learn_cliff.append(loaded_all_rewards)
            loaded_all_avg_rewards_cliff.append(loaded_all_avg_rewards)
            loaded_all_std_devs_cliff.append(loaded_all_std_devs)

        else:
            loaded_rew_during_learn_froze.append(loaded_all_rewards)
            loaded_all_avg_rewards_froze.append(loaded_all_avg_rewards)
            loaded_all_std_devs_froze.append(loaded_all_std_devs)

# rl agent results
loaded_rl_during_learn_cliff = []
loaded_rl_all_avg_rewards_cliff = []
loaded_rl_all_std_devs_cliff = []

loaded_rl_during_learn_froze = []
loaded_rl_all_avg_rewards_froze = []
loaded_rl_all_std_devs_froze = []
for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]
    file_path = os.path.join(results_directory, f"{env_name}_rl_agent_results.pkl")

    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    loaded_all_rewards = loaded_data['all_rewards']
    if chosen_env == 0:
        loaded_rl_during_learn_cliff = loaded_all_rewards
        loaded_rl_all_avg_rewards_cliff = loaded_data["all_avg_rewards"]
        loaded_rl_all_std_devs_cliff = loaded_data["all_std_devs"]
    else:
        loaded_rl_during_learn_froze = loaded_all_rewards
        loaded_rl_all_avg_rewards_froze = loaded_data["all_avg_rewards"]
        loaded_rl_all_std_devs_froze = loaded_data["all_std_devs"]


plot_title = 'during_training_rewards_cliff.pdf'
name = 'Cliffwalking'
plot_type = 'during Training'


loaded_rew = np.array(loaded_rew_during_learn_cliff)
loaded_rl_reward = np.array(loaded_rl_during_learn_cliff)

column_averages_var1 = np.mean(loaded_rew[0], axis=0)
column_std_devs_var1 = np.std(loaded_rew[0], axis=0)

column_averages_var2 = np.mean(loaded_rew[1], axis=0)
column_std_devs_var2 = np.std(loaded_rew[1], axis=0)

column_averages_var3 = np.mean(loaded_rew[2], axis=0)
column_std_devs_var3 = np.std(loaded_rew[2], axis=0)

column_averages_var4 = np.mean(loaded_rew[3], axis=0)
column_std_devs_var4 = np.std(loaded_rew[3], axis=0)

column_averages_standard = np.mean(loaded_rl_reward, axis=0)
column_std_devs_standard = np.std(loaded_rl_reward, axis=0)

# Plotting rewards during training Cliffwalking


fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

# Loop through subplots to plot data
for i, (ax, averages, std_devs) in enumerate(zip(axs, [column_averages_var1, column_averages_var2, column_averages_var3, column_averages_var4, column_averages_standard],
                                                  [column_std_devs_var1, column_std_devs_var2, column_std_devs_var3, column_std_devs_var4, column_std_devs_standard])):
    if i == 4:
        ax.plot(num_episodes_array_cliff, averages, label=f'Standard RL', color=colors[i])
    else:
        ax.plot(num_episodes_array_cliff, averages, label=feedback_variants[i], color=colors[i])
    ax.fill_between(num_episodes_array_cliff,
                    np.minimum(averages - std_devs, optimal_cliff),
                    np.minimum(averages + std_devs, optimal_cliff),
                    alpha=0.25, color=colors[i])
    ax.axhline(y=optimal_cliff, color=optimal_color, linestyle='--', label='Optimal Value', zorder=6)
    ax.set_xlabel('Number of Episodes')
    ax.set_ylabel(f'Avg Rewards')
    ax.legend()

# Set common y-axis range for all subplots after the loop
y_min = min(np.min(averages - std_devs) for averages, std_devs in zip([column_averages_var1, column_averages_var2, column_averages_var3, column_averages_var4, column_averages_standard],
                                                                      [column_std_devs_var1, column_std_devs_var2, column_std_devs_var3, column_std_devs_var4, column_std_devs_standard]))
y_max = optimal_cliff + 500
# Apply the common y-axis range to all subplots
for ax in axs:
    ax.set_ylim(y_min, y_max)

# Remove overlapping labels and set layout
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save and display the plot
plt.savefig(os.path.join(plots_directory, plot_title))
plt.show()
plt.close()


plot_title = 'during_training_rewards_froze.pdf'
name = 'Frozenlake'
plot_type = 'during Training'


loaded_rew = np.array(loaded_rew_during_learn_froze)
loaded_rl_reward = np.array(loaded_rl_during_learn_froze)



column_averages_var1 = np.mean(loaded_rew[0], axis=0)
column_std_devs_var1 = np.std(loaded_rew[0], axis=0)

column_averages_var2 = np.mean(loaded_rew[1], axis=0)
column_std_devs_var2 = np.std(loaded_rew[1], axis=0)

column_averages_var3 = np.mean(loaded_rew[2], axis=0)
column_std_devs_var3 = np.std(loaded_rew[2], axis=0)

column_averages_var4 = np.mean(loaded_rew[3], axis=0)
column_std_devs_var4 = np.std(loaded_rew[3], axis=0)

column_averages_standard = np.mean(loaded_rl_reward, axis=0)
column_std_devs_standard = np.std(loaded_rl_reward, axis=0)

# Plotting rewards during training Frozenlake
fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

# Loop through subplots to plot data
for i, (ax, averages, std_devs) in enumerate(zip(axs, [column_averages_var1, column_averages_var2, column_averages_var3, column_averages_var4, column_averages_standard],
                                                  [column_std_devs_var1, column_std_devs_var2, column_std_devs_var3, column_std_devs_var4, column_std_devs_standard])):
    if i == 4:
        ax.plot(num_episodes_array_froze, averages, label=f'Standard RL', color=colors[i])
    else:
        ax.plot(num_episodes_array_froze, averages, label=feedback_variants[i], color=colors[i])

    ax.fill_between(num_episodes_array_froze,
                    np.minimum(averages - std_devs, optimal_froze).clip(0),
                    np.minimum(averages + std_devs, optimal_froze).clip(0),
                    alpha=0.25, color=colors[i])
    ax.axhline(y=optimal_froze, color=optimal_color, linestyle='--', label='Optimal Value', zorder=6)
    ax.set_xlabel('Number of Episodes')
    ax.set_ylabel(f'Avg Rewards')
    ax.legend()

# Set common y-axis range for all subplots after the loop
y_min = 0
y_max = optimal_froze+0.1
# Apply the common y-axis range to all subplots
for ax in axs:
    ax.set_ylim(y_min, y_max)

# Remove overlapping labels and set layout
plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save and display the plot
plt.savefig(os.path.join(plots_directory, plot_title))
plt.show()
plt.close()


#trained agent froze
plot_title = 'trained_agent_rewards_froze.pdf'
name = 'Frozenlake'
plot_type = 'on trained agent'


avg_rewards_variant1 = np.array(loaded_all_avg_rewards_froze[0])
std_dev_variant1 = np.array(loaded_all_std_devs_froze[0])

avg_rewards_variant2 = np.array(loaded_all_avg_rewards_froze[1])
std_dev_variant2 = np.array(loaded_all_std_devs_froze[1])

avg_rewards_variant3 = np.array(loaded_all_avg_rewards_froze[2])
std_dev_variant3 = np.array(loaded_all_std_devs_froze[2])

avg_rewards_variant4 = np.array(loaded_all_avg_rewards_froze[3])
std_dev_variant4 = np.array(loaded_all_std_devs_froze[3])

avg_rewards_rl_agent = np.array(loaded_rl_all_avg_rewards_froze)
std_dev_rl_agent = np.array(loaded_rl_all_std_devs_froze)

# Scatter plot
plt.figure(figsize=(10, 6))

plt.scatter(avg_rewards_variant1, std_dev_variant1, label=feedback_variants[0], marker='o', color = colors[0])
plt.scatter(avg_rewards_variant2, std_dev_variant2, label=feedback_variants[1], marker='o', color = colors[1])
plt.scatter(avg_rewards_variant3, std_dev_variant3, label=feedback_variants[2], marker='o', color = colors[2])
plt.scatter(avg_rewards_variant4, std_dev_variant4, label=feedback_variants[3], marker='o', color = colors[3])
plt.scatter(avg_rewards_rl_agent, std_dev_rl_agent, label='RL Agent', marker='o', color = colors[4])

plt.xlabel('Average Rewards')
plt.ylabel('Standard Deviation')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(plots_directory, plot_title))
plt.show()
plt.close()

failed_variants = []
for i, sublist in enumerate(loaded_all_avg_rewards_cliff):
    count = sum(1 for value in sublist if value != -13)
    if count > 0:
        failed_variants.append((feedback_variants[i], count))

print("Feedback variants that not found optimal policies in the evaluation of cliffwalking and how often:", failed_variants)
