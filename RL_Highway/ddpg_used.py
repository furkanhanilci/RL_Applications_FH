import os
import torch
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import numpy as np
import math
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import math
import numpy as np
import highway_env
highway_env.register_highway_envs()

from matplotlib import pyplot as plt
import copy




# About Conv2d, why in_channel is 1
# https://datascience.stackexchange.com/questions/64278/what-is-a-channel-in-a-cnn


# About Convolution layer output dimension
# https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer


class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=2, stride=1)
        
        out_height_1, out_width_1 = conv_output(self.obs_dim, kernel=3, padding=0, stride=1)
        out_height_2, out_width_2 = conv_output([out_height_1, out_width_1], kernel=2, padding=0, stride=1)
        out_height_3, out_width_3 = conv_output([out_height_2, out_width_2], kernel=2, padding=0, stride=1)

        linear_input_size= out_height_3*out_width_3*64


        self.linear1 = nn.Linear(linear_input_size, 1024)
        self.linear2 = nn.Linear(1024 + self.action_dim, 512)
        self.linear3 = nn.Linear(512, 300)
        self.linear4 = nn.Linear(300, 1)

    def forward(self, obs, act):
        # Output: (batch_size, num_channels, height, width)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Output: (batch_size, num_channels * height * width)
        x = torch.flatten(x, 1)

        
        x = F.relu(self.linear1(x))
        xa_cat = torch.cat([x,act], 1)
        xa = F.relu(self.linear2(xa_cat))
        xa = F.relu(self.linear3(xa))
        qval = self.linear4(xa)

        return qval
    

# Deterministic policy
class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1) # Output: (batch_size, 64, height, width)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=2, stride=1) # Output: (batch_size, 128, height, width)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=2, stride=1) # Output: (batch_size, 64, height, width)
        
        out_height_1, out_width_1 = conv_output(self.obs_dim, kernel=3, padding=0, stride=1)
        out_height_2, out_width_2 = conv_output([out_height_1, out_width_1], kernel=2, padding=0, stride=1)
        out_height_3, out_width_3 = conv_output([out_height_2, out_width_2], kernel=2, padding=0, stride=1)
        
        linear_input_size= out_height_3*out_width_3*64

        
        self.linear1 = nn.Linear(linear_input_size, 512) # Output: (batch_size, 512)
        self.linear2 = nn.Linear(512, 128) # Output: (batch_size, 128)
        self.linear3 = nn.Linear(128, self.action_dim) # Output: (batch_size, action_dim)

    def forward(self, obs):
        # Output: (batch_size, num_channels, height, width)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
      
        # Output: (batch_size, num_channels * height * width)
        x = torch.flatten(x, 1)
        
        # Output: (batch_size, output dimension)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x  # return action


def conv_output(in_shape, kernel, padding, stride):
    
    # [(W−K+2P)/S]+1
    out_height = int(((in_shape[0] - kernel+2*padding) / stride) + 1)
    out_width = int(((in_shape[1] - kernel+ 2*padding ) / stride) + 1)
    return out_height, out_width


# Ornstein-Ulhenbeck Noise
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)
    
    def __len__(self):
        return len(self.buffer)



class DDPGAgent:
    
    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate,state_dim,action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Run on:", self.device)
        
        self.env = env
        self.obs_dim = state_dim
        self.action_dim = action_dim
        self.total_rewards = []
        self.avg_reward = []
        self.actor_loss = []
        self.critic_loss = []
        
        # hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        
        # initialize actor and critic networks
        self.critic = Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.obs_dim, self.action_dim).to(self.device)
        
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim).to(self.device)
    
        # Copy critic target parameters
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
    
        self.replay_buffer = BasicBuffer(buffer_maxlen)        
        self.noise = OUNoise(self.env.action_space)
        
    def get_action(self, obs): # From Actor network
        state = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0).to(self.device)
        #state = torch.FloatTensor(obs).to(self.device)
        action = self.actor.forward(state)
        action = action.squeeze(0).cpu().detach().numpy()

        return action
    
    def save(self, path, fname):
     
        if not os.path.exists(path):
            os.makedirs(path)
        path_file = os.path.join(path, fname)

        save_dict = {
            'model_actor_state': self.actor.state_dict(),
            'model_critic_state': self.critic.state_dict(),
            'model_actor_op_state': self.actor_optimizer.state_dict(),
            'model_critic_op_state': self.critic_optimizer.state_dict(),
            

            'total_rewards': self.total_rewards,
            'avg_rewards': self.avg_reward,
            'actor_loss': self.actor_loss,
            'critic_loss': self.critic_loss,

        }
        torch.save(save_dict, path_file)

    def load(self, path, fname):
     

        path_file = os.path.join(path, fname)
        load_dict = torch.load(path_file, map_location=self.device)

        # Load weights and optimizer states
        self.actor.load_state_dict(load_dict['model_actor_state'])
        self.critic.load_state_dict(load_dict['model_critic_state'])
        self.actor_optimizer.load_state_dict(load_dict['model_actor_op_state'])
        self.critic_optimizer.load_state_dict(load_dict['model_critic_op_state'])
        
        # Load other variables
        self.total_rewards = load_dict['total_rewards']
        self.avg_reward = load_dict['avg_rewards']
        self.actor_loss = load_dict['actor_loss']
        self.critic_loss = load_dict['critic_loss']


        # Creating the target networks
        self.target_actor_net = copy.deepcopy(self.actor).to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic).to(self.device)
        print("Successfully load the model parameters.")
    
    
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)

        # Sample a random minibatch of N batch size transitions from Replay Buffer
        state_batch, action_batch, reward_batch, next_state_batch, masks = self.replay_buffer.sample(batch_size)
       
        
   
        # Training Actor (Calculation of current Q)
        # state_batch = torch.FloatTensor(state_batch).to(self.device)
        state_batch = torch.FloatTensor(state_batch).unsqueeze(1).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device) # Deterministic action selection
        curr_Q = self.critic.forward(state_batch, action_batch) 
        
        # Training target Critic (Calculation of next Q)
        next_state_batch = torch.FloatTensor(next_state_batch).unsqueeze(1).to(self.device)
        next_actions = self.actor_target.forward(next_state_batch)
        next_Q = self.critic_target.forward(next_state_batch, next_actions.detach())
        
        
        # Calculation of expected Q (y value)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device)
        expected_Q = reward_batch + self.gamma * next_Q # self.gamma-->discount factor
        
        # Main loss or critic loss (mean-squared Bellman error-MSBE, Temporal difference)
        q_loss = F.mse_loss(curr_Q, expected_Q.detach())
        self.critic_loss.append(q_loss)

        # Update the critic network by minimizing loss
        self.critic_optimizer.zero_grad()
        q_loss.backward() 
        self.critic_optimizer.step()

        
        
        # Calculate actor policy or actor loss (positive for maximizing Q, negative for minimizing loss)
        # In the below code, the goal is to minimize a loss function that measures the difference between predicted output and true label. 
        policy_loss = -self.critic.forward(state_batch, self.actor.forward(state_batch)).mean() # mean of the Q values for all state-action pairs
        self.actor_loss.append(policy_loss.item())
        # Update actor network using backward method 
        # This code block adjusts the actors's policy to produce actions that result in higher-Q values by minimizing loss.
        self.actor_optimizer.zero_grad() # gradients of the actor optimizer to zero for clean optimization
        policy_loss.backward() # backpropagation to compute gradients of "policy_loss"
        self.actor_optimizer.step() # update actor network parameters to based on computed gradients

        # soft update target networks using Polyak averaging 
        # For actor target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        # For critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


def mini_batch_train(env, agent, max_episodes, max_steps, batch_size,exploration_noise):
    episode_rewards = []
    file_name = "model_final_saved.pt"
    
    for episode in range(max_episodes):
        s= env.reset()
        state=s[0]
        episode_reward = 0
        
        for step in range(max_steps):
            
            action = agent.get_action(state)[0]
            action= action + exploration_noise.get_action(action)
            next_state, reward, done, truncated, info  = env.step(action)
            next_state_tensor = torch.tensor(next_state)
            reward, done=reward_modified(reward,next_state_tensor,done)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)
                


            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                exploration_noise.reset() # Re-initialize the process when episode ends
                print("Episode " + str(episode) + ": " + str(episode_reward))
                agent.total_rewards.append(reward)
                agent.avg_reward.append(np.mean(agent.total_rewards))
                break

            state = next_state
            env.render() 
    agent.save("model", fname=file_name)  # Save the model parameters

def reward_modified(reward,next_state_tensor,done):

    num_vehicles=next_state_tensor.shape[0]
    front_v = False

    if reward == 0:
                reward = -3
                done = True
    # Set done condition and giving a penalty if the ego vehicle is moving very slowly in the x-axis
    # The ego-vehicle is always described in the first row (http://highway-env.farama.org/observations/)
    elif next_state_tensor[0][3].item() < 0.15:
                done = True
                reward -= 0.7
    else:
            for veh in range(1, num_vehicles):
                if abs(next_state_tensor[veh][2].item()) < 0.17:  # Check if there is any vehicle in the same lane
                        if 0.09 < next_state_tensor[veh][1].item() < 0.15:  # Reward for maintaining appropriate distance from the front vehicle
                                reward += 0.2
                        if next_state_tensor[veh][3].item() < 0.07:  # Reward for maintaining relative speed to the front vehicle
                                reward += 0.1

                        elif next_state_tensor[veh][1].item() < 0.075:  # Penalize if the ego vehicle is getting too close to the front vehicle
                                reward -= 0.3

                        if abs(next_state_tensor[veh][1].item()) < 0.20:  # Check if the front vehicle is in a safe distance
                                front_v = True

    # Reward for moving faster if there is no vehicle within the safe distance
    if front_v == False and 0.28 < next_state_tensor[0][3].item() < 0.31:
                    reward += 0.4

    # Reward for moving with appropriate x-axis speed and not making a sharp y-axis movement
    if abs(next_state_tensor[0][4].item()) < 0.05 and 0.24 < next_state_tensor[0][3].item() < 0.31:
                    reward += 0.4

    # Penalize for moving too slow but still above the threshold
    elif next_state_tensor[0][3].item() < 0.2:
                    reward -= 0.4

    # Penalize for making a very quick movement in the y-axis
    if next_state_tensor[0][4].item() > 0.2:
                    reward -= 0.4
                    done = True


    return reward, done
                

env = gym.make('highway-v0', render_mode='rgb_array')


# Environment configuration
configuration={"observation": {
        "type": "Kinematics",
        "vehicles_count": 7, #rows of observation
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
    }

env.configure(configuration)

max_episodes = 1000
max_steps = 500
batch_size = 32

gamma = 0.99
tau = 1e-2
buffer_maxlen = 100000
critic_lr = 1e-3
actor_lr = 1e-3
num_of_test_episodes=200

state = env.reset()

# Get state dimensions
# Number of vehicles X features, features= Vehicle, x, y, vx, vy (https://highway-env.farama.org/observations/)
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
state_dim_a = [env.observation_space.shape[0], env.observation_space.shape[1]]

# Get action dimension
# throttle and steering angle (https://highway-env.farama.org/actions/)
action_dim = env.action_space.shape[0]



exploration_noise = OUNoise(env.action_space)

# Training
agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr,state_dim_a,action_dim)
episode_rewards = mini_batch_train(env, agent, max_episodes, max_steps, batch_size,exploration_noise)


# Testing 
reward_test=[]
avg_reward_test=[]
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
new_env=gym.make('highway-v0', render_mode='rgb_array')
new_env.configure(configuration)
agent.load("model", "model_final_saved.pt") # Agent degisecek mi?
agent.actor.to(agent.device)
for i in range(num_of_test_episodes):
    s = new_env.reset()
    state=s[0]
    local_reward = 0
    done = False
    while not done:
        state =  torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(agent.device)
        #state =  torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)
        action = agent.actor.forward(state)      
        action = action.squeeze(0).cpu().detach().numpy()
        next_state, reward, done, truncated, info= new_env.step(action)
        next_state_tensor = torch.tensor(next_state)
        reward, done=reward_modified(reward,next_state_tensor,done)
        state=next_state
        local_reward += reward
        new_env.render()
    reward_test.append(local_reward)
    avg_reward_test.append(np.mean(reward_test))

plt.plot(agent.critic_loss)
plt.ylabel("Critic Loss")
plt.xlabel("Updated Steps")
plt.title("Critic Loss")
plt.show()

plt.plot(agent.actor_loss)
plt.ylabel("Actor Loss")
plt.xlabel("Updated Steps")
plt.title("Actor Loss")
plt.show()

plt.plot(agent.total_rewards, label='Total Reward in the episode')
plt.plot(agent.avg_reward, label='Average Total Reward over episodes')
plt.ylabel("Rewards")
plt.title("Training Reward")
plt.xlabel("Episodes")
plt.legend()
plt.show()

print('stop')