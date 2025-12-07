import gymnasium as gym
import gymnasium_env  # This registers the GridWorld environment
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt


def run(episodes, is_training = True, render = False):
# Create the environment with human rendering
    env = gym.make("gymnasium_env/GridWorld-v0", size=4, render_mode="human" if render else None)
    
    print("GridWorld Environment Loaded Successfully!")

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))# init a 64 x 4 array
    else:
        f = open('grid_30x30.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    
    

    print("Starting random agent... (Close the window to stop)")

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()
    
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        while not (terminated or truncated):
            if is_training and rng.random() < epsilon:

                action = env.action_space.sample()  # Random action: 0=up, 1=down, 2=left, 3=right
            else:
                action = np.argmax(q[state,:])
            new_state, reward, terminated, truncated, _ = env.step(action)
    
            if is_training:
                q[state,action] = q[state,action] + learning_rate_a*(reward + discount_factor_g*np.max(q[new_state,:]) - q[state,action])

            state = new_state
        
        epsilon = max(epsilon - epsilon_decay_rate,0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()
    # Small delay so you can see movement smoothly
        
    
    # Optional: print info
    # print(f"Action: {action}, Reward: {reward}, Position: {observation}")

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0,t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('grid_30x30.png')

    if is_training:
        f = open("grid_30x30.pkl", "wb")
        pickle.dump(q,f)
        f.close()
    

if __name__ == '__main__':
    run(150,is_training=False,render=True)