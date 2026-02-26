import gymnasium as gym
import logging 

logging.basicConfig(level=logging.INFO, filename='log.txt', format='%(message)s')

env_name = "Taxi-v3"

env = gym.make(env_name).env

episodes = 1000

for episode in range(episodes):
    state, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # save the episode data to a log file
        logging.info(f"Episode: {episode}, State: {state}, Action: {action}, Reward: {reward}, done: {done}")
