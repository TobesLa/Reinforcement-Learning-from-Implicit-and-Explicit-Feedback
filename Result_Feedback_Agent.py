from Agents.Feedback_Agent import FeedbackAgent
import gymnasium as gym
import os
from Feedback.Feedback_FileHandler import FeedbackFileHandler


# this file can be used to run our feedback agent

def print_q_vals_and_reward():
    for s, a in feedback:
        for a_prime in range(4):
            print()
            if a_prime == a:
                print('should be optimal!')
            print('q-value for state action pair: ', (s, a_prime), 'q: ', feedback_q_table[s][a_prime])
            print('reward for state action pair ', (s, a_prime), 'r: ', reward_model.get_reward(s, a_prime))


env_names = ['CliffWalking-v0', "FrozenLake-v1"]
start_states = [36, 0]
optimal_vals = [-13, 1]
goal_states = [(3, 11), (3, 3)]
feedback_variants = ['IBF', 'ABF', 'ERF', 'SRF']  # var1 to var4
learning_rates_cliff = [0.1, 0.01, 0.1, 0.01]
learning_rates_froze = [0.01, 0.1, 0.1, 0.01]

discounts_cliff = [0.4, 0.9, 0.95, 0.95]
discounts_froze = [0.7, 0.6, 0.8, 0.9]

# change index 0 = cliffwalk, 1 = frozenlake
chosen_env = 0

# index of feedback variants
f_index = 2

feedback_var = feedback_variants[f_index]

env_name = env_names[chosen_env]
epsilon = 1
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

num_q_updates = 5
max_steps_per_episode = 50
noise = 0.1  # 1 should not work!
delay = 5
batches = 5  # 10

env = gym.make(env_name)
goal_state = goal_states[chosen_env]

directory = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(directory, rf"Feedback\{env_name}_feedback.txt")

feedback_filehandler = FeedbackFileHandler(file_path)
feedback = feedback_filehandler.load_dict_from_file()
print(feedback)
agent = FeedbackAgent(env_name, feedback, learning_rate, discount, epsilon, exploration_decay_rate,
                      max_steps_per_episode,
                      num_episodes, num_q_updates, noise, delay, batches, feedback_var)
feedback_q_table, true_rewards_all_episodes, rm_rewards_all_episodes = agent.feedback_learning()
agent.visualize_policy(feedback_q_table, env, goal_state)

print('avg rewards: ', sum(true_rewards_all_episodes) / len(true_rewards_all_episodes))
start_state = start_states[chosen_env]

optimal_val = optimal_vals[chosen_env]
agent.plot_rewards(optimal_val)
reward_model = agent.agent.reward_model

print_q_vals_and_reward()

print(f'decayed epsilon: {agent.agent.epsilon}')
# agent.visualize_results() #for gui
