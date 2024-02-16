from Agents.Feedback_Agent import FeedbackAgent
from Feedback.Feedback_FileHandler import FeedbackFileHandler
import os
import statistics
import pickle

env_names = ['CliffWalking-v0', 'FrozenLake-v1']
feedback_variants = ['IBF', 'ABF', 'ERF', 'SRF'] # var1 to var4
num_runs = 10
num_train_results = 1000
results_directory = 'Results'

num_q_updates = 5
max_steps_per_episode = 50
noise = 0.1
delay = 5
batches = 5
epsilon = 1

learning_rates_cliff = [0.1, 0.01, 0.1, 0.01]
learning_rates_froze = [0.01, 0.1, 0.1, 0.01]

discounts_cliff = [0.4, 0.9, 0.95, 0.95]
discounts_froze = [0.7, 0.6, 0.8, 0.9]

directory = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory
parent_directory = os.path.dirname(directory)

for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]
    results_data = {}

    all_rewards = {feedback_var: [] for feedback_var in feedback_variants}
    all_avg_rewards = {feedback_var: [] for feedback_var in feedback_variants}
    all_std_devs = {feedback_var: [] for feedback_var in feedback_variants}

    file_path = os.path.join(parent_directory, "Feedback", f"{env_name}_feedback.txt")

    feedback_filehandler = FeedbackFileHandler(file_path)
    feedback = feedback_filehandler.load_dict_from_file()
    print(feedback)

    for i in range(num_runs):
        print('Learning for the', i, 'time')
        for f_index in range(len(feedback_variants)):
            feedback_var = feedback_variants[f_index]
            print(f'Currently at feedback variant {feedback_var}')
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
            agent = FeedbackAgent(env_name, feedback, learning_rate, discount, epsilon, exploration_decay_rate,
                                  max_steps_per_episode,
                                  num_episodes, num_q_updates, noise, delay, batches, feedback_var)
            feedback_q_table, true_rewards_all_episodes, rm_rewards_all_episodes = agent.feedback_learning()
            all_rewards[feedback_var].append(true_rewards_all_episodes)
            print('Done learning')

            rewards_per_run = [agent.get_training_result() for _ in range(num_train_results)]
            avg_reward = sum(rewards_per_run) / num_train_results
            std_dev_reward = statistics.stdev(rewards_per_run)

            print(f'Average Reward: {avg_reward}, Standard Deviation: {std_dev_reward}')
            all_avg_rewards[feedback_var].append(avg_reward)
            all_std_devs[feedback_var].append(std_dev_reward)

        print('Write to file')
        results_data[f'all_rewards_{env_name}'] = all_rewards.copy()
        results_data[f'all_avg_rewards_{env_name}'] = all_avg_rewards.copy()
        results_data[f'all_std_devs_{env_name}'] = all_std_devs.copy()

        file_path = os.path.join(results_directory, f"{env_name}_feedback_agent_results.pkl")

        # Ensure the 'Results' directory exists
        os.makedirs(results_directory, exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(results_data, file)

        print(f'Results saved to {file_path}')

for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]
    file_path = os.path.join(results_directory, f"{env_name}_feedback_agent_results.pkl")

    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    for feedback_var in feedback_variants:
        loaded_all_rewards = loaded_data.get(f'all_rewards_{env_name}', {}).get(feedback_var, None)
        loaded_all_avg_rewards = loaded_data.get(f'all_avg_rewards_{env_name}', {}).get(feedback_var, None)
        loaded_all_std_devs = loaded_data.get(f'all_std_devs_{env_name}', {}).get(feedback_var, None)

        print(f'Loaded all_rewards for {env_name} and {feedback_var}: {loaded_all_rewards}')
        print(f'Loaded all_avg_rewards for {env_name} and {feedback_var}: {loaded_all_avg_rewards}')
        print(f'Loaded all_std_devs for {env_name} and {feedback_var}: {loaded_all_std_devs}')
