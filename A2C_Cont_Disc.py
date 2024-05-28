import gym
import torch
import numpy as np
from torch.distributions.categorical import Categorical
from torch import nn
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.rewards = []
        self.action_prob = []
        self.state_values = []
        self.entropy = []

    def calculate_data_dr(self, gamma):
        disc_rewards = []
        R = 0
        for reward in self.rewards[::-1]:
            R = reward + gamma * R
            disc_rewards.insert(0, R)

        disc_rewards = torch.Tensor(disc_rewards)
        min_reward = disc_rewards.min()
        max_reward = disc_rewards.max()
        disc_rewards = (disc_rewards - min_reward) / (max_reward - min_reward)

        return torch.stack(self.action_prob), torch.stack(self.state_values), disc_rewards.to(device), torch.stack(self.entropy)

    def calculate_data_ar(self):
        average_reward = sum(self.rewards) / len(self.rewards)
        average_reward_value_function = [r - average_reward for r in self.rewards]
        average_reward_value_function = torch.Tensor(average_reward_value_function)
        return torch.stack(self.action_prob), torch.stack(self.state_values), average_reward_value_function.to(device), torch.stack(self.entropy)

    def update(self, reward, entropy, log_prob, state_value):
        self.entropy.append(entropy)
        self.action_prob.append(log_prob)
        self.state_values.append(state_value)
        self.rewards.append(reward)

    def reset(self):
        del self.rewards[:]
        del self.action_prob[:]
        del self.state_values[:]
        del self.entropy[:]

class ActorCriticContinuous(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(ActorCriticContinuous, self).__init__()

        self.fc_1_critic = nn.Linear(state_dim, hidden_dim)
        self.fc_2_critic = nn.Linear(hidden_dim, int(hidden_dim / 2))

        self.fc_1_actor = nn.Linear(state_dim, hidden_dim)
        self.fc_2_actor = nn.Linear(hidden_dim, int(hidden_dim / 2))

        self.critic_head = nn.Linear(int(hidden_dim / 2), 1)
        self.actor_head_mean = nn.Linear(int(hidden_dim / 2), action_dim)
        self.actor_head_sigma = nn.Linear(int(hidden_dim / 2), action_dim)

    def forward_critic(self, inp):
        x = F.leaky_relu(self.fc_1_critic(inp))
        x = F.leaky_relu(self.fc_2_critic(x))
        state_value = self.critic_head(x)
        return state_value

    def forward_actor(self, inp):
        x = F.leaky_relu(self.fc_1_actor(inp))
        x = F.leaky_relu(self.fc_2_actor(x))
        action_mean = self.actor_head_mean(x)
        action_sigma = F.softplus(self.actor_head_sigma(x))
        return action_mean, action_sigma

class ActorCriticDiscrete(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(ActorCriticDiscrete, self).__init__()

        self.fc_1 = nn.Linear(state_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, int(hidden_dim / 2))

        self.critic_head = nn.Linear(int(hidden_dim / 2), 1)
        self.actor_head = nn.Linear(int(hidden_dim / 2), action_dim)

    def forward(self, inp):
        x = F.leaky_relu(self.fc_1(inp))
        x = F.leaky_relu(self.fc_2(x))
        state_value = self.critic_head(x)
        action_prob = F.softmax(self.actor_head(x), dim=-1)
        return action_prob, state_value

class A2C:
    def __init__(self, parameters, learning_rate):
        self.lr = learning_rate
        self.optimize = torch.optim.Adam(parameters, learning_rate)

    def train(self, memory, gamma, eps):
        dis_count_reward_option = 0
        if dis_count_reward_option == 1:
            action_prob, values, disc_rewards, entropy = memory.calculate_data_dr(gamma)
            advantage = disc_rewards.detach() - values
        else:
            action_prob, values, average_reward_value, entropy = memory.calculate_data_ar()
            advantage = average_reward_value.detach() - values

        policy_loss = (-action_prob * advantage.detach()).mean() - eps * entropy.mean()
        value_loss = 0.5 * advantage.pow(2).mean()
        loss = policy_loss + value_loss

        self.optimize.zero_grad()
        loss.backward()
        self.optimize.step()

def select_action(model, state, mode):
    state = torch.Tensor(state).to(device)
    if mode == "continuous":
        V_value = model.forward_critic(state)
        mean, sigma = model.forward_actor(state)
        s = torch.distributions.MultivariateNormal(mean, torch.diag(sigma))
    else:
        probs, state_value = model(state)
        s = Categorical(probs)

    action = s.sample()
    entropy = s.entropy()
    log_prob_action = s.log_prob(action)

    return action.cpu().numpy(), entropy, log_prob_action, V_value

def main(gamma=0.99, lr=5e-3, num_episodes=400, eps=0.001, seed=42, lr_step=100, lr_gamma=0.9, horizon=200,
         hidden_dim=64, env_name='Pendulum-v0', render=True, save_model=True, model_path='actor_critic.pth'):
    env = gym.make(env_name)
    torch.manual_seed(seed)
    env.seed(seed)

    if type(env.action_space) == gym.spaces.Discrete:
        action_mode = "discrete"
    elif type(env.action_space) == gym.spaces.Box:
        action_mode = "continuous"
    else:
        raise Exception("action space is not known")

    if action_mode == "continuous":
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    state_dim = env.observation_space.shape[0]

    if action_mode == "continuous":
        actor_critic = ActorCriticContinuous(action_dim=action_dim, state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    else:
        actor_critic = ActorCriticDiscrete(action_dim=action_dim, state_dim=state_dim, hidden_dim=hidden_dim).to(device)

    print(f'size of each action = {action_dim}')
    print(f'size of state = {state_dim}')
    print(f'low of each action = {env.action_space.low}')
    print(f'high of each action = {env.action_space.high}')

    a2c = A2C(actor_critic.parameters(), lr)
    reward_list = []
    avg_score_deque = deque(maxlen=100)
    avg_scores_list = []

    for episode in range(num_episodes):
        memory = Memory()
        state = env.reset()
        total_reward = 0
        done = False
        count = 0
        while not done and count < horizon:
            count += 1
            action, entropy, log_prob, state_value = select_action(actor_critic, state, action_mode)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            memory.update(reward, entropy, log_prob, state_value)
            if render:
                env.render()

            print(f"episode: {episode + 1}, steps: {count}, current reward: {total_reward}")

        a2c.train(memory, gamma, eps)
        memory.reset()

        reward_list.append(total_reward)
        avg_score_deque.append(total_reward)
        mean = np.mean(avg_score_deque)
        avg_scores_list.append(mean)

    plt.plot(range(num_episodes), reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Training Performance')
    plt.show()

    if save_model:
        torch.save(actor_critic.state_dict(), model_path)

    return actor_critic, reward_list, avg_scores_list

if __name__ == '__main__':
    actor_critic, reward_list, avg_scores_list = main(render=True)

    # Testing
    env = gym.make('Pendulum-v0')
    best_actor_critic = actor_critic
    best_actor_critic.load_state_dict(torch.load('actor_critic.pth'))
    best_actor_critic.eval()
    total_rewards = []
    for _ in range(200):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            state = torch.Tensor(state).to(device)
            with torch.no_grad():
                mean, sigma = best_actor_critic.forward_actor(state)
                m = torch.distributions.Normal(mean, sigma)
                action = m.sample().cpu().numpy()
            state, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward
        total_rewards.append(total_reward)

    import plotly.graph_objects as go

    x = np.array(range(len(total_rewards)))
    m = np.mean(total_rewards)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=total_rewards, name='test reward',
                             line=dict(color="green", width=1)))

    fig.add_trace(go.Scatter(x=x, y=[m] * len(total_rewards), name='average reward',
                             line=dict(color="red", width=1)))

    fig.update_layout(title="A2C Test Performance",
                      xaxis_title="test episodes",
                      yaxis_title="reward")
    fig.show()

    print("average test reward:", m)
