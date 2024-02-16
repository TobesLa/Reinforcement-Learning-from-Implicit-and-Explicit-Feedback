import unittest

from Agents.Feedback_Agent import FeedbackAgent

feedback_variants = ['IBF', 'ABF', 'ERF', 'SRF'] # var1 to var4
class TestFeedbackAgent(unittest.TestCase):
    def test_batches_invalid(self):
        agent = FeedbackAgent(env_name='CliffWalking-v0', feedback='some_feedback', learning_rate=0.01, discount=0.99,
                              epsilon=1, exploration_decay_rate=0.01, max_steps_per_episode=100,
                              num_episodes=1000, num_q_updates=5, noise=0.1, delay=5, batches=10,
                              feedback_variant= feedback_variants[0])
        self.assertIsNotNone(agent)

    def test_env_invalid(self):
        with self.assertRaises(ValueError):
            agent = FeedbackAgent(env_name='env', feedback='some_feedback', learning_rate=0.1, discount=0.5,
                                  epsilon=0.5, exploration_decay_rate=0.01, max_steps_per_episode=100,
                                  num_episodes=1000, num_q_updates=5, noise=0.1, delay=5, batches=5,
                                  feedback_variant= feedback_variants[0])

    def test_lr_invalid(self):
        with self.assertRaises(ValueError):
            FeedbackAgent(env_name='CliffWalking-v0', feedback='some_feedback', learning_rate=-0.1, discount=0.5,
                          epsilon=0.5, exploration_decay_rate=0.01, max_steps_per_episode=100,
                          num_episodes=1000, num_q_updates=5, noise=0.1, delay=5, batches=5,
                          feedback_variant= feedback_variants[0])

    def test_noise_invalid(self):
        with self.assertRaises(ValueError):
            FeedbackAgent(env_name='CliffWalking-v0', feedback='some_feedback', learning_rate=0.1, discount=0.5,
                          epsilon=0.5, exploration_decay_rate=0.01, max_steps_per_episode=100,
                          num_episodes=1000, num_q_updates=5, noise=100, delay=5, batches=5,
                          feedback_variant= feedback_variants[0])

    def test_batches_invalid(self):
        with self.assertRaises(ValueError):
            FeedbackAgent(env_name='CliffWalking-v0', feedback='some_feedback', learning_rate=0.1, discount=0.5,
                          epsilon=0.5, exploration_decay_rate=0.01, max_steps_per_episode=100,
                          num_episodes=1000, num_q_updates=5, noise=0.1, delay=5, batches=-10,
                          feedback_variant= feedback_variants[0])
    def test_batches_invalid(self):
        with self.assertRaises(ValueError):
            FeedbackAgent(env_name='CliffWalking-v0', feedback='some_feedback', learning_rate=0.1, discount=0.5,
                          epsilon=0.5, exploration_decay_rate=0.01, max_steps_per_episode=100,
                          num_episodes=1000, num_q_updates=5, noise=0.1, delay=5, batches=10,
                          feedback_variant='not a Feedback Variant')


if __name__ == '__main__':
    unittest.main()
