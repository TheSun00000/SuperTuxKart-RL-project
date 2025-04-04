import gymnasium as gym
from bbrl.agents import Agent
import torch
import torch.nn as nn
from .utils import CustomAgent


class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action


class Actor(Agent):
    """Computes probabilities over action"""
    def __init__(self, observation_space, action_space):
        super().__init__(name="NaximActor")
        
        input_dim, hidden_dim = 64, 512
        discrete_dims = [2, 2, 2, 2, 2]
        continuous_dims = [1, 1]
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.discrete_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            for out_dim in discrete_dims
        ])
        self.continuous_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                nn.Sigmoid() if i == 0 else nn.Identity()
            )
            for i, out_dim in enumerate(continuous_dims)
        ])
        
        self.agent = CustomAgent(self.shared_layers, self.discrete_heads, self.continuous_heads)
    
    
    
    def state_to_tensor(self, state):

        # continuous_vars = ['is_stuck', 'obstacle_ahead', 'obstacle_position', 'target_position', 'target_distance', 'target_angle']
        # continuous_vars = ['obstacle_ahead', 'obstacle_position', 'powerup', 'previous_actions', 'start_race', 'target_angle', 'target_distance', 'target_position', 'velocity']
        continuous_vars = ['karts_position', 'previous_actions', 'velocity', 'start_race', 'obstacle_ahead', 'obstacle_position', 'target_position', 'target_angle', 'target_distance', 'powerup']
        # print(state.keys())
        continuous_state = []
        for key in continuous_vars:
            value = state[key]
            if value.shape:
                num_envs = value.shape[0]
                if key in ['items_position', 'paths_end', 'paths_start', 'paths_width']:
                    normalized_value = value[..., :10, :]
                else:    
                    # normalized_value = (value - self.means[key]) / self.stds[key]
                    normalized_value = value
                    
                continuous_state.append(normalized_value.reshape(num_envs, -1))

        continuous_state = torch.concatenate(continuous_state, axis=-1)
        # state = torch.tensor(continuous_state, dtype=torch.float32)
        state = continuous_state.float()

        return state
    
    
    
    def forward(self, t: int):
        # print(self.workspace.keys())
        keys = ['start_race', 'powerup', 'attachment', 'attachment_time_left', 'karts_position', 'previous_actions', 'velocity', 'obstacle_ahead', 'obstacle_position', 'target_position', 'target_distance', 'target_angle']
        state = {}
        for key in keys:
            value = self.get((f'env/env_obs/{key}', t))
            # state[key] = value
            state[key] = value.squeeze(0)
                    


        action = self.agent(state)
        continuous_action = torch.tensor(action['continuous']).unsqueeze(0).float()
        discrete_action   = torch.tensor(action['discrete']).unsqueeze(0).long()
        
        
        self.set(("action/continuous", t), continuous_action)
        self.set(("action/discrete", t), discrete_action)


class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        # Selects the best actions according to the policy
        pass


class SamplingActor(Agent):
    """Samples random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        action = self.action_space.sample()

        self.set(("action/continuous", t), torch.tensor([action['continuous']]))
        self.set(("action/discrete", t), torch.tensor([action['discrete']]))

