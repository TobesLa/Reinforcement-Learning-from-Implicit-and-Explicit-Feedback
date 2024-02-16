from Agents.RL_Agent import RLAgent
import gymnasium as gym
import pickle
import statistics
import os


env_names = ['CliffWalking-v0', "FrozenLake-v1"]
start_states = [36, 0]
optimal_vals = [-13, 1]
goal_states = [(3, 11), (3, 3)]
num_runs = 10
num_train_results = 1000

learning_rate = 0.15
discount = 0.99
epsilon = 1
max_steps_per_episode = 50

for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]
    num_episodes = 1000 if chosen_env == 0 else 10000
    env = gym.make(env_name)
    goal_state = goal_states[chosen_env]
    results_data = {}
    all_rewards = []
    all_avg_rewards = []
    all_std_devs = []

    for i in range(num_runs):
        print('learning for the :', i, 's time')
        rl_agent = RLAgent(env_name, learning_rate, discount, epsilon, num_episodes, max_steps_per_episode)
        q_table, rl_rewards = rl_agent.q_learning()
        all_rewards.append(rl_rewards)
        print('done learning')
        sum_rewards = 0
        rewards_per_run = []
        for j in range(num_train_results):
            training_result = rl_agent.get_training_result()
            rewards_per_run.append(training_result)
            sum_rewards += training_result

        avg_reward = sum_rewards / num_train_results
        std_dev_reward = statistics.stdev(rewards_per_run) # Calculate standard deviation
        print(f'Average Reward: {avg_reward}, Standard Deviation: {std_dev_reward}')
        all_avg_rewards.append(avg_reward)
        all_std_devs.append(std_dev_reward)


    print('write to file')

    # Save the results to a file
    results_data = {'all_rewards': all_rewards, 'all_avg_rewards': all_avg_rewards, 'all_std_devs': all_std_devs}
    results_directory = 'Results'
    file_path = os.path.join(results_directory, f"{env_name}_rl_agent_results.pkl")

    # Ensure the 'Results' directory exists
    os.makedirs(results_directory, exist_ok=True)

    with open(file_path, 'wb') as file:
        pickle.dump(results_data, file)

    print(f'Results saved to {file_path}')


for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]
    file_path = os.path.join(results_directory, f"{env_name}_rl_agent_results.pkl")

    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    print(f'Loaded all_avg_rewards: {loaded_data["all_avg_rewards"]}')
    print(f'Loaded all_std_devs: {loaded_data["all_std_devs"]}')