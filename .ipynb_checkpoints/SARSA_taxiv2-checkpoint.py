import gym
import numpy as np
import matplotlib.pyplot as plt

def eps_greedy(Q, s, eps=0.1):
    '''
    Epsilon Greedy Policy, dealing with EE Tradeoff. 
    '''
    if np.random.uniform(0, 1)< eps:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[s])
    
    
def run_episodes(env, Q, num_episodes=100, to_print=False):
    """
    Executes Episodic Tasks untill termination and returns the total reward at the end. 
    env: OpenAI env.
    Q :  Action Value Table 
    num_episodes: Episodes of Games to play. 
    to_print: flag for printing stats.
    """
    state = env.reset()
    total_rewards = []
    
    for ii in range(num_episodes):
        done = False
        curr_reward = 0
        while not done:
            s_, r, done, info = env.step(np.argmax(Q[state]))
            
            state = s_
            curr_reward += r
            if done:
                state = env.reset()
                total_rewards.append(curr_reward)
                
    if to_print:
        print(f'Mean Score {np.mean(total_rewards)} of {num_episodes} Episodes')
        
    return np.mean(total_rewards)


def Q_learning(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    """
    In case of Q Learning we are updating the Action Table with the Maximum action value 
    for the next state. 
    Q[s, a] = Q[s, a] + lr * (r + gamma * max(Q[s_]) - Q[s, a])
    
    """
    action_range = env.action_space.n
    observation_range = env.observation_space.n
    
    Q = np.zeros((observation_range, action_range))
    
    current_reward = []
    test_rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        
        if eps > 0.01:
            eps -= eps_decay
            
        while not done:
            action = eps_greedy(Q, state, eps)
            s_, r, done, info = env.step(action)
            
            Q[state][action] += lr * (r + gamma * np.max(Q[s_]) - Q[state][action]) # Selecting the next action from the maximum action value. 
            state = s_
            total_rewards += r
            if done:
                current_reward.append(total_rewards)
        
        if (ep % 300) == 0:
            test_reward = run_episodes(env, Q, 1000)
            print("Episode:{:5d}  Epsilon:{:2.4f}  Reward:{:2.4f}".format(ep, eps, test_reward))
            test_rewards.append(test_reward)
            
    return Q

def SARSA(env, lr=0.01, num_episodes=10000, eps=0.3, gamma=0.95, eps_decay=0.00005):
    """
    In case of SARSA, we are selecting the next set of actions for the next state 
    based on the Epsilon Greedy Policy, thereby dealing with Exploration-Exploitation
    Tradeoff. 
    Q[s, a] = Q[s, a] + lr * (r + gamma * Q[next_state][next_action] - Q[s, a])
    
    """
    action_range = env.action_space.n
    observation_range = env.observation_space.n
    Q = np.zeros((observation_range, action_range))
    games_rewards = []
    test_rewards = []
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_rewards = 0
        
        if eps > 0.01:
            eps -= eps_decay
            
        action = eps_greedy(Q, state, eps)
        
        while not done:
            s_, r, done, info = env.step(action)
            next_action = eps_greedy(Q, s_, eps)  #Selecting the Next Action based on Epsilon Greedy Policy. 
            
            Q[state][action] += lr * (r + gamma * Q[s_][next_action] - Q[state][action])
            state = s_
            action = next_action
            total_rewards += r
            if done:
                games_rewards.append(total_rewards)
        
        if ep % 300 == 0:
            test_reward = run_episodes(env, Q, 1000)
            print("Episode:{:5d}  Eps:{:2.4f}  Rew:{:2.4f}".format(ep, eps, test_reward))
            test_rewards.append(test_reward)
            
    return Q
if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    q_learning = Q_learning(env, lr=0.1, num_episodes=5000, eps=0.4, gamma=.95, eps_decay=.001)
    q_sarsa = SARSA(env, lr=.1, num_episodes=5000, eps=.4, gamma=.95, eps_decay=.001)
        