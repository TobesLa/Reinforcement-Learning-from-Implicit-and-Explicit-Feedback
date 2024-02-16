import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import statistics

# This code plots the results from the exp_delay experiment

env_names = ['CliffWalking-v0', 'FrozenLake-v1']
feedback_variants = ['IBF', 'ABF', 'ERF', 'SRF'] # var1 to var4
colors = ['blue', 'darkgreen', 'orange', 'magenta', 'black']
optimal_color = 'cyan'
optimal_froze = 1
optimal_cliff = -13
plots_directory = 'Plots'
results_directory = 'Results'

delay_levels = np.arange(0, 500, 50)
num_runs = 3

if not os.path.exists(plots_directory):
    print(f"Creating directory: {plots_directory}")
    os.makedirs(plots_directory)

num_train_results = 100

cliff_true_rewards = {}
cliff_all_avg_rewards = {}
cliff_all_std_devs = {}

froze_true_rewards = {}
froze_all_avg_rewards = {}
froze_all_std_devs = {}
# Load Data
# delay results
# Loading and printing the data
for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]

    for feedback_var in feedback_variants:
        file_path = os.path.join(results_directory, f"{env_name}_{feedback_var}_agents_delay_result.pkl")

        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)

        loaded_all_true_rewards = loaded_data.get(f'all_true_rewards_{env_name}_{feedback_var}', [])
        loaded_all_avg_rewards = loaded_data.get(f'all_avg_rewards_{env_name}_{feedback_var}', [])
        loaded_all_std_devs = loaded_data.get(f'all_std_devs_{env_name}_{feedback_var}', [])
        if chosen_env == 0:
            cliff_true_rewards[feedback_var] = loaded_all_true_rewards
            cliff_all_avg_rewards[feedback_var] = loaded_all_avg_rewards
            cliff_all_std_devs[feedback_var] = loaded_all_std_devs
        else:
            froze_true_rewards[feedback_var] = loaded_all_true_rewards
            froze_all_avg_rewards[feedback_var] = loaded_all_avg_rewards
            froze_all_std_devs[feedback_var] = loaded_all_std_devs

plot_title = 'sample_IBF_cliff_delay.pdf'
#plot one sample
plt.plot(np.arange(1000), cliff_true_rewards.get('IBF')[0][5], label='IBF',
         color=colors[0], zorder=3)
plt.axhline(y=optimal_cliff, color=optimal_color, linestyle='--', label='Optimal Value', zorder=1)
plt.xlabel('Episode')
plt.ylabel(f'Total Reward with delay of {delay_levels[5]}')
plt.legend()
plt.savefig(os.path.join(plots_directory, plot_title))
plt.show()

plot_title = 'sample_IBF_froze_delay.pdf'
#plot one sample
plt.plot(np.arange(10000), froze_true_rewards.get('IBF')[0][5], label='IBF',
         color=colors[0], zorder=3)
plt.axhline(y=optimal_froze, color=optimal_color, linestyle='--', label='Optimal Value', zorder=1)
plt.xlabel('Episode')
plt.ylabel(f'Total Reward with delay of {delay_levels[5]}')
plt.legend()
plt.savefig(os.path.join(plots_directory, plot_title))
plt.show()



# convert so that we save for each variant the mean instead all the rewards
for feedback_var in feedback_variants:
    for i in range(num_runs):
        for d in range(len(delay_levels)):
            curr_rewards_cliff = np.array(cliff_true_rewards.get(feedback_var)[i][d])
            cliff_true_rewards.get(feedback_var)[i][d] = curr_rewards_cliff.mean()
            curr_rewards_froze = np.array(froze_true_rewards.get(feedback_var)[i][d])
            froze_true_rewards.get(feedback_var)[i][d] = curr_rewards_froze.mean()

plot_title = 'during_training_rewards_delay_cliff.pdf'
name = 'Cliffwalking'
plot_type = 'during Training'

fig, axs = plt.subplots(len(feedback_variants), 1, figsize=(10, 15), sharex=True)

# Loop through subplots to plot data
for i, (ax, feedback_var) in enumerate(zip(axs, feedback_variants)):
    all_runs = np.array(cliff_true_rewards[feedback_var])
    avg_values = np.mean(all_runs, axis=0)
    std_dev_values = np.std(all_runs, axis=0)

    ax.plot(delay_levels, avg_values, label=feedback_variants[i], color=colors[i])
    ax.fill_between(delay_levels, avg_values - std_dev_values, avg_values + std_dev_values, alpha=0.25, color=colors[i])
    ax.set_xlabel('Delay')
    ax.set_ylabel(f'Avg Rewards during Training')
    ax.axhline(y=optimal_cliff, color=optimal_color, linestyle='--', label='Optimal Value', zorder=6)
    ax.legend()

# Set common y-axis range for all subplots after the loop
y_min = min(np.min(avg_values - std_dev_values) for feedback_var in feedback_variants for avg_values, std_dev_values in
            zip(cliff_true_rewards[feedback_var], np.std(cliff_true_rewards[feedback_var], axis=0)))
y_max = optimal_cliff + 100
for ax in axs:
    ax.set_ylim(y_min, y_max)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(plots_directory, plot_title))
plt.show()

print(cliff_true_rewards)


plot_title = 'during_training_rewards_delay_froze.pdf'
name = 'Frozenlake'
plot_type = 'during Training'


fig, axs = plt.subplots(len(feedback_variants), 1, figsize=(10, 15), sharex=True)

# Loop through subplots to plot data
for i, (ax, feedback_var) in enumerate(zip(axs, feedback_variants)):
    all_runs = np.array(froze_true_rewards[feedback_var])
    avg_values = np.mean(all_runs, axis=0)
    std_dev_values = np.std(all_runs, axis=0)

    ax.plot(delay_levels, avg_values, label=feedback_variants[i], color=colors[i])
    ax.fill_between(delay_levels, avg_values - std_dev_values, avg_values + std_dev_values, alpha=0.25, color=colors[i])
    ax.set_xlabel('Delay')
    ax.set_ylabel(f'Avg Rewards during Training')
    #ax.axhline(y=optimal_froze, color=optimal_color, linestyle='--', label='Optimal Value', zorder=6)
    ax.legend()

# Set common y-axis range for all subplots after the loop
y_min = 0
y_max = max(np.max(avg_values + std_dev_values) for feedback_var in feedback_variants for avg_values, std_dev_values in
            zip(froze_true_rewards[feedback_var], np.std(froze_true_rewards[feedback_var], axis=0)))

for ax in axs:
    ax.set_ylim(y_min, y_max)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(plots_directory, plot_title))
plt.show()


#trained agent cliff
plot_title = 'trained_agent_rewards_delay_cliff.pdf'
name = 'Cliffwalking'
plot_type = 'on trained agent'


print(cliff_all_avg_rewards)
print(cliff_all_std_devs)

fig, axs = plt.subplots(len(feedback_variants), 1, figsize=(10, 15), sharex=True)

# Loop through subplots to plot data
for i, (ax, feedback_var) in enumerate(zip(axs, feedback_variants)):
    all_runs = np.array(cliff_all_avg_rewards[feedback_var])
    avg_values = np.mean(all_runs, axis=0)
    std_dev_values = np.std(all_runs, axis=0)

    ax.plot(delay_levels, avg_values, label=feedback_variants[i], color=colors[i])
    ax.fill_between(delay_levels, avg_values - std_dev_values, avg_values + std_dev_values, alpha=0.25, color=colors[i])
    ax.set_xlabel('Delay')
    ax.set_ylabel(f'Avg Rewards on trained agent')
    ax.axhline(y=optimal_cliff, color=optimal_color, linestyle='--', label='Optimal Value', zorder=6)
    ax.legend()

# Set common y-axis range for all subplots after the loop
y_min = min(np.min(avg_values) for feedback_var in feedback_variants for avg_values in
            np.mean(np.array(cliff_all_avg_rewards[feedback_var]), axis=0))

y_max = optimal_cliff +100
for ax in axs:
    ax.set_ylim(y_min, y_max)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(plots_directory, plot_title))
plt.show()

#trained agent froze
plot_title = 'trained_agent_rewards_delay_froze.pdf'
name = 'Frozenlake'
plot_type = 'on trained agent'

fig, axs = plt.subplots(len(feedback_variants), 1, figsize=(10, 15), sharex=True)

# Loop through subplots to plot data
for i, (ax, feedback_var) in enumerate(zip(axs, feedback_variants)):
    all_runs = np.array(froze_all_avg_rewards[feedback_var])
    avg_values = np.mean(all_runs, axis=0)
    std_dev_values = np.std(all_runs, axis=0)

    ax.plot(delay_levels, avg_values, label=feedback_variants[i], color=colors[i])
    ax.fill_between(delay_levels, avg_values - std_dev_values, avg_values + std_dev_values, alpha=0.25, color=colors[i])
    ax.set_xlabel('Delay')
    ax.set_ylabel(f'Avg Rewards on trained agent')
    ax.axhline(y=optimal_froze, color=optimal_color, linestyle='--', label='Optimal Value', zorder=6)
    ax.legend()
y_min = 0
y_max = optimal_froze+0.1
for ax in axs:
    ax.set_ylim(y_min, y_max)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(plots_directory, plot_title))
plt.show()


