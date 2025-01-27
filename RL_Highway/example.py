from matplotlib import pyplot as plt

import gymnasium as gym


import highway_env
highway_env.register_highway_envs()


env = gym.make('highway-v0', render_mode='rgb_array')
env.configure({
    "action": {
        "type": "ContinuousAction"
    }
})

env.reset()

for _ in range(300):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()


plt.imshow(env.render())
plt.show()