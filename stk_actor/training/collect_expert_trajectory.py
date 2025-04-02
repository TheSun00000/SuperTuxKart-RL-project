import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from tqdm import tqdm
import numpy as np


if __name__ == "__main__":

    # num_envs = 32  # Number of parallel environments
    # env = gym.vector.SyncVectorEnv([
    #     lambda: gym.make("supertuxkart/flattened_multidiscrete-v0", render_mode=None, agent=AgentSpec(use_ai=True))
    #     for _ in range(num_envs)
    # ])
    
    num_envs = 1
    env = gym.make("supertuxkart/flattened_multidiscrete-v0", render_mode=None, agent=AgentSpec(use_ai=True))




    state, _ = env.reset()
    steps = [0]*num_envs
    rounds = 0

    states = []
    actions = []
    rewards = []
    terminateds = []
    truncateds = []

    tqdm_steps = tqdm(range(1, 1000001))
    for step in tqdm_steps:

        # action = env.action_space.sample()
        # print(actions)
        # exit()
        next_state, reward, terminated, truncated, _ = env.step(
            env.action_space.sample()
        )
        action = next_state['action']
        done = terminated | truncated
        done = [done]
        # dones = [terminated or truncated for terminated, truncated in zip(terminateds, truncateds)]
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        terminateds.append(terminated)
        truncateds.append(truncated)

        
        for i in range(len(done)):
            steps[i] += 1
            if True:
                # print(steps, steps[i])
                new_state, _ = env.reset()
                next_state['continuous'] = new_state['continuous']
                next_state['discrete'] = new_state['discrete']
                steps[i] = 0
                rounds += 1
        
        # if any(done):
        #     break
        
        state = next_state
        
        tqdm_steps.set_description(f'size={len(states)*num_envs} rounds={rounds}')
            
        # print(steps)
        
        
        # # print(to_save.shape)
        # if step % 10000 == 0:            
            
        #     to_save = np.concat([
        #         np.concat([state['continuous'].reshape(1, -1) for state in states]),
        #         np.concat([state['discrete'].reshape(1, -1) for state in states]),
        #         np.concat([action.reshape(1, -1) for action in actions]),
        #         np.array(terminateds).reshape(-1, 1),
        #         np.array(truncateds).reshape(-1, 1),
        #     ], axis=1)
        #     np.save('flattened_multidiscrete_1.npy', to_save)

    env.close()