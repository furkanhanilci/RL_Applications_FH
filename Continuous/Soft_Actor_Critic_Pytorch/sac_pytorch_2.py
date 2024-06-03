import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import time
from pyvirtualdisplay import Display
import imageio

# Start virtual display
display = Display(visible=1, size=(1400, 900))
display.start()

# Device selection: Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))
        log_std = torch.ones_like(mean) * -1
        std = torch.exp(log_std)
        return mean, std

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(done)).unsqueeze(1).to(device)
        )

class SACAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, std = self.policy(state)
        if deterministic:
            action = mean
        else:
            action = mean + std * torch.randn_like(mean)
        action = torch.clamp(action, -1, 1)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, step):
        if len(replay_buffer.buffer) < 64:
            return

        state, action, reward, next_state, done = replay_buffer.sample()

        with torch.no_grad():
            next_action_mean, next_action_std = self.policy(next_state)
            next_action = next_action_mean + next_action_std * torch.randn_like(next_action_mean)
            next_action = torch.clamp(next_action, -1, 1)
            next_q1 = self.q1_target(next_state, next_action)
            next_q2 = self.q2_target(next_state, next_action)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_action_std
            target_q = reward + (1 - done) * self.gamma * next_q

        q1_loss = nn.MSELoss()(self.q1(state, action), target_q)
        q2_loss = nn.MSELoss()(self.q2(state, action), target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        action_mean, action_std = self.policy(state)
        action = action_mean + action_std * torch.randn_like(action_mean)
        action = torch.clamp(action, -1, 1)
        q1_new = self.q1(state, action)
        q2_new = self.q2(state, action)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * action_std - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Mode: 0 for training, 1 for testing
mode = 1

if mode == 0:  # Training mode
    # Training parameters
    num_episodes = 1000
    max_steps = 1000
    success_threshold = 200
    successful_episodes = 0
    start_time = time.time()

    # Lists to store metrics
    rewards = []
    success_rates = []
    average_rewards = []
    running_times = []
    convergence_times = []

    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    replay_buffer = ReplayBuffer()
    agent = SACAgent(state_dim, action_dim)

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        for t in range(max_steps):
            env.render()  # Render the environment for each step
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)

            agent.update(replay_buffer, episode * max_steps + t)
            state = next_state
            episode_reward += reward

            if done:
                break

        rewards.append(episode_reward)
        if episode_reward >= success_threshold:
            successful_episodes += 1

        success_rate = successful_episodes / (episode + 1)
        success_rates.append(success_rate)

        average_reward = np.mean(rewards)
        average_rewards.append(average_reward)

        current_time = time.time()
        running_time = current_time - start_time
        running_times.append(running_time)

        convergence_time = (num_episodes - successful_episodes) / num_episodes * running_time
        convergence_times.append(convergence_time)

        print(f"Episode {episode}, Reward: {episode_reward}, Success Rate: {success_rate}, Average Reward: {average_reward}")

    env.close()

    # Save model
    torch.save(agent.q1.state_dict(), 'q1_model.pth')
    torch.save(agent.q2.state_dict(), 'q2_model.pth')
    torch.save(agent.policy.state_dict(), 'policy_model.pth')

    # Plotting the graphs
    fig, axs = plt.subplots(4, 1, figsize=(12, 20))

    axs[0].plot(range(num_episodes), rewards, label='Reward')
    axs[0].set_title('Reward per Episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()

    axs[1].plot(range(num_episodes), success_rates, label='Success Rate')
    axs[1].set_title('Success Rate per Episode')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Success Rate')
    axs[1].legend()

    axs[2].plot(range(num_episodes), average_rewards, label='Average Reward')
    axs[2].set_title('Average Reward per Episode')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Average Reward')
    axs[2].legend()

    axs[3].plot(range(num_episodes), running_times, label='Running Time')
    axs[3].plot(range(num_episodes), convergence_times, label='Convergence Time')
    axs[3].set_title('Running and Convergence Time')
    axs[3].set_xlabel('Episode')
    axs[3].set_ylabel('Time (seconds)')
    axs[3].legend()

    plt.tight_layout()
    plt.show()

elif mode == 1:  # Test mode
    # Load model
    env = gym.make('LunarLanderContinuous-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim)
    agent.q1.load_state_dict(torch.load('q1_model.pth'))
    agent.q2.load_state_dict(torch.load('q2_model.pth'))
    agent.policy.load_state_dict(torch.load('policy_model.pth'))

    # Testing process
    test_episodes = 1000  # Number of episodes to test
    test_rewards = []
    frames = []  # List to store frames

    best_reward = -float('inf')
    best_episode_frames = []

    for episode in range(test_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        episode_frames = []

        while not done:
            frame = env.render(mode='rgb_array')  # Render the environment
            episode_frames.append(frame)  # Save frames
            action = agent.select_action(state, deterministic=True)  # Deterministic action selection
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward

        test_rewards.append(episode_reward)
        print(f"Test Episode {episode}, Reward: {episode_reward}")

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_episode_frames = episode_frames

    env.close()

    # Save the best episode as a GIF
    imageio.mimsave('best_episode.gif', best_episode_frames, fps=30)

    # Visualize test rewards
    plt.figure(figsize=(10, 5))
    plt.plot(range(test_episodes), test_rewards, label='Test Reward')
    plt.title('Test Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
