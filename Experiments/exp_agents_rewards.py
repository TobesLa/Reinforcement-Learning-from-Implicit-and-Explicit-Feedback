from Agents.Feedback_Agent import FeedbackAgent
import gymnasium as gym
from Agents.RL_Agent import RLAgent
import os
from Feedback.Feedback_FileHandler import FeedbackFileHandler
import pickle

env_names = ['CliffWalking-v0', 'FrozenLake-v1']
feedback_variants = ['IBF', 'ABF', 'ERF', 'SRF', 'standard'] # var1 to var4
results_directory = 'Results'
goal_states = [(3, 11), (3, 3)]
num_q_updates = 5
max_steps_per_episode = 50
noise = 0.1#1
delay = 5
batches = 5
epsilon = 1

learning_rates_cliff = [0.1, 0.01, 0.1, 0.01,0.15]
learning_rates_froze = [0.01, 0.1, 0.1, 0.01,0.15]

discounts_cliff = [0.4, 0.9, 0.95, 0.95,0.99]
discounts_froze = [0.7, 0.6, 0.8, 0.9,0.99]

directory = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the parent directory
parent_directory = os.path.dirname(directory)
all_true_rewards = []

for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]
    results_data = {}

    all_true_rewards = {feedback_var: [] for feedback_var in feedback_variants}

    file_path = os.path.join(parent_directory, "Feedback", f"{env_name}_feedback.txt")

    feedback_filehandler = FeedbackFileHandler(file_path)
    feedback = feedback_filehandler.load_dict_from_file()
    print(feedback)
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
        if feedback_var != 'standard':
            agent = FeedbackAgent(env_name, feedback, learning_rate, discount, epsilon, exploration_decay_rate,
                                  max_steps_per_episode,
                                  num_episodes, num_q_updates, noise, delay, batches, feedback_var)
            feedback_q_table, true_rewards_all_episodes, rm_rewards_all_episodes = agent.feedback_learning()
            agent.visualize_policy(feedback_q_table, gym.make(env_name), goal_states[chosen_env]) # see if policy is alright
        else:
            agent = RLAgent(env_name, learning_rate, discount, epsilon, num_episodes, max_steps_per_episode)
            q_table, true_rewards_all_episodes = agent.q_learning()
        all_true_rewards[feedback_var].append(true_rewards_all_episodes)

    results_data[f'all_true_rewards_{env_name}'] = all_true_rewards.copy()
    file_path = os.path.join(results_directory, f"{env_name}_agents_rewards_result.pkl")

    # Ensure the 'Results' directory exists
    os.makedirs(results_directory, exist_ok=True)

    with open(file_path, 'wb') as file:
        pickle.dump(results_data, file)

    print(f'Results saved to {file_path}')

for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]
    file_path = os.path.join(results_directory, f"{env_name}_agents_rewards_result.pkl")

    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    for feedback_var in feedback_variants:
        loaded_all_rewards = loaded_data.get(f'all_true_rewards_{env_name}', {}).get(feedback_var, None)

        print(f'Loaded all_rewards for {env_name} and {feedback_var}: {loaded_all_rewards}')

