from Agents.Feedback_Agent import FeedbackAgent
import gymnasium as gym
import os
from Feedback.Feedback_FileHandler import FeedbackFileHandler
import pickle
import statistics
import numpy as np

env_names = ['CliffWalking-v0', 'FrozenLake-v1']
feedback_variants = ['IBF', 'ABF', 'ERF', 'SRF'] # var1 to var4
results_directory = 'Results'
goal_states = [(3, 11), (3, 3)]
num_q_updates = 5
max_steps_per_episode = 50

num_runs = 3 # 5 takes too long

noise = 0.1  # constant
batches = 5
epsilon = 1
num_train_results = 1000

learning_rates_cliff = [0.1, 0.01, 0.1, 0.01]
learning_rates_froze = [0.01, 0.1, 0.1, 0.01]

discounts_cliff = [0.4, 0.9, 0.95, 0.95]
discounts_froze = [0.7, 0.6, 0.8, 0.9]

directory = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory
parent_directory = os.path.dirname(directory)

delay_levels = np.arange(0, 500, 50)
print('delay levels: ', len(delay_levels))
for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]

    results_data = {}

    for f_index in range(len(feedback_variants)):
        feedback_var = feedback_variants[f_index]

        all_true_rewards = []
        all_avg_rewards = []
        all_std_devs = []

        file_path = os.path.join(parent_directory, "Feedback", f"{env_name}_feedback.txt")

        feedback_filehandler = FeedbackFileHandler(file_path)
        feedback = feedback_filehandler.load_dict_from_file()
        print(feedback)

        for run in range(num_runs):
            print(f'Currently at feedback variant {feedback_var}, run {run + 1}/{num_runs}')

            if chosen_env == 0:
                # cliffwalking
                learning_rate = learning_rates_cliff[f_index]
                discount = discounts_cliff[f_index]
                num_episodes = 1000
                exploration_decay_rate = 0.01
            else:
                # frozenlake
                learning_rate = learning_rates_froze[f_index]
                discount = discounts_froze[f_index]
                num_episodes = 10000
                # frozenlake needs more exploration
                exploration_decay_rate = 0.0005

            true_rewards_list = []
            avg_rewards_list = []
            std_devs_list = []

            for delay in delay_levels:
                print('at delay level: ', int(delay))
                agent = FeedbackAgent(env_name, feedback, learning_rate, discount, epsilon, exploration_decay_rate,
                                      max_steps_per_episode,
                                      num_episodes, num_q_updates, noise, int(delay), batches, feedback_var)
                feedback_q_table, true_rewards_all_episodes, rm_rewards_all_episodes = agent.feedback_learning()

                true_rewards_list.append(true_rewards_all_episodes)

                rewards_per_run = [agent.get_training_result() for _ in range(num_train_results)]
                avg_reward = sum(rewards_per_run) / num_train_results
                std_dev_reward = statistics.stdev(rewards_per_run)
                avg_rewards_list.append(avg_reward)
                std_devs_list.append(std_dev_reward)

            all_true_rewards.append(true_rewards_list)
            all_avg_rewards.append(avg_rewards_list)
            all_std_devs.append(std_devs_list)

        print('Write to file')
        results_data[f'all_true_rewards_{env_name}_{feedback_var}'] = all_true_rewards
        results_data[f'all_avg_rewards_{env_name}_{feedback_var}'] = all_avg_rewards
        results_data[f'all_std_devs_{env_name}_{feedback_var}'] = all_std_devs

        # Ensure the 'Results' directory exists
        os.makedirs(results_directory, exist_ok=True)

        file_path = os.path.join(results_directory, f"{env_name}_{feedback_var}_agents_delay_result.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(results_data, file)

        print(f'Results saved to {file_path}')

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
        if feedback_var == 'var1':
            print(f'HELLOOOOOO, loaded all true rewards is of size {len(loaded_all_true_rewards)} showing the number of runs \n'
                  f' while a component is a list of size {len(loaded_all_true_rewards[0])} showing the number of delay levels')
        print(f'Loaded all_true_rewards for {env_name} and {feedback_var}: {loaded_all_true_rewards}')
        print(f'Loaded all_avg_rewards for {env_name} and {feedback_var}: {loaded_all_avg_rewards}')
        print(f'Loaded all_std_devs for {env_name} and {feedback_var}: {loaded_all_std_devs}')
