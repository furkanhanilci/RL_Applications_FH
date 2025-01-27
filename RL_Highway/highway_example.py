from matplotlib import pyplot as plt
import gymnasium as gym
import highway_env
highway_env.register_highway_envs()

import numpy as np
import torch
import copy
import random
import statistics as stat
import warnings
warnings.filterwarnings("ignore")


import networks as network
import utils as utils
import ddpg as ddpg




env = gym.make('highway-v0', render_mode='rgb_array')

# Environment configuration
env.configure(
    {"observation": {
        "type": "Kinematics",
        "vehicles_count": 7,
        "features": ["presence", "x", "y", "vx", "vy"],
    },

        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True
        },
        "absolute": False,
        "lanes_count": 4,
        "reward_speed_range": [40, 60], #  [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
        "simulation_frequency": 15,
        "vehicles_count": 50,
        "policy_frequency": 10,
        "initial_spacing": 5,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "duration": 20, # [s]
        "collision_reward": -2, #  The reward received when colliding with a vehicle.
        "action_reward": -0.3, # penalty
        "screen_width": 600,
        "screen_height": 300,
        "centering_position": [0.3, 0.5],
        "scaling": 7,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    })


# Start the training process
state = env.reset()

# Get state dimensions
# Number of vehicles X features, features= Vehicle, x, y, vx, vy (https://highway-env.farama.org/observations/)
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
state_dim_a = [env.observation_space.shape[0], env.observation_space.shape[1]]

# Get action dimension
#  throttle and steering angle (https://highway-env.farama.org/actions/)
action_dim = env.action_space.shape[0]



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Run on:", device)



# Initialize Actor and Critic network
actor = network.Actor_network(state_dim_a, action_dim).to(device)
critic = network.Critic_network(state_dim_a, action_dim, 256).to(device)

# Initialize target networks
target_actor = copy.deepcopy(actor).to(device)
target_critic = copy.deepcopy(critic).to(device)

# Initialize agent
ddpg_agent = ddpg.DDPG_agent(env, actor, critic, target_actor, target_critic, device)

# Load the model parameters (To continue training on the previous trained model only)
#ddpg_agent.load("model", "model_final.pt")

# Train the agent
ddpg_agent.train()


# Model Evaluation
eval_reward_list = []
avg_training = []

# Evaluate the model 5 trials
for i in range (5):
    eval_reward = utils.eval_agent(ddpg_agent, env, fname="model_final.pt", device=device, load=True)
    print(f"Avg test reward over episodes for trial {i+1}: {eval_reward[1][-1]:.3f}")
    eval_reward_list.append(eval_reward)
    avg_training.append(eval_reward[1][-1])

# Show the training result plots
utils.plots(ddpg_agent)

# Show the test reward plot for the last trial
utils.plot_eval(eval_reward)

# Show the mean and standard deviation of training reward
mean_avg_reward = stat.mean(avg_training)
stdev_avg_reward = stat.stdev(avg_training)

print(f"Mean of Avg test reward: {mean_avg_reward:.3f}")
print(f"Stdev of Avg test reward: {stdev_avg_reward:.3f}")


