from Agents.RL_Agent import RLAgent
import gymnasium as gym
from Feedback.Feedback_FileHandler import FeedbackFileHandler

# This code generates the simulated feedback and saves it to a .txt

env_names = ['CliffWalking-v0', "FrozenLake-v1"]
start_states = [36, 0]
optimal_vals = [-13, 1]
goal_states = [(3, 11), (3, 3)]

for chosen_env in range(len(env_names)):
    env_name = env_names[chosen_env]
    learning_rate = 0.15
    discount = 0.99
    epsilon = 1  # 1 if epsilon should decay over time as well
    num_episodes = 1000 if chosen_env == 0 else 10000
    max_steps_per_episode = 100

    env = gym.make(env_name)
    goal_state = goal_states[chosen_env]
    max_reward = 0  # Track the maximum percentage
    best_feedback = None  # Track the feedback corresponding to the maximum percentage
    best_agent = None
    for i in range(20):
        print('learning for the :', i, 's time')
        print(num_episodes)
        print(env_name)
        rl_agent = RLAgent(env_name, learning_rate, discount, epsilon, num_episodes, max_steps_per_episode)
        q_table, rl_rewards = rl_agent.q_learning()
        print('done learning')
        print('feedback extracted')
        sum_rewards = 0
        for j in range(100):
            # print('get sum rewards')
            sum_rewards += rl_agent.get_training_result()
        avg_reward = sum_rewards / 100
        print(avg_reward)

        # Update best_agent if a higher percentage is found
        if avg_reward > max_reward or best_feedback is None:
            max_reward = avg_reward
            best_agent = rl_agent
            best_reward = 0
            # ensure feedback found a trajectory that leads to optima
            while best_reward != optimal_vals[chosen_env]:
                best_feedback, best_reward = best_agent.extract_feedback()

    print('write to file')
    # Save feedback dictionary to a file
    file_path = f"{env_name}_feedback.txt"
    feedback_filehandler = FeedbackFileHandler(file_path)
    print(best_feedback)
    feedback_filehandler.save_dict_to_file(best_feedback)

    print('load feedback', feedback_filehandler.load_dict_from_file())
    best_agent.save_policy(goal_state)
