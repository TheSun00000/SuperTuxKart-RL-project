import gymnasium as gym
from pystk2_gymnasium.wrappers import SpaceFlattener
from pystk2_gymnasium.definitions import ActionObservationWrapper
import numpy as np


# Obstacles
# {'BONUS_BOX': <Type.BONUS_BOX: 0>,
#  'BANANA': <Type.BANANA: 1>,
#  'NITRO_BIG': <Type.NITRO_BIG: 2>,
#  'NITRO_SMALL': <Type.NITRO_SMALL: 3>,
#  'BUBBLEGUM': <Type.BUBBLEGUM: 4>,
#  'EASTER_EGG': <Type.EASTER_EGG: 6>}

# Powerups
# {'NOTHING': <Type.NOTHING: 0>,
#  'BUBBLEGUM': <Type.BUBBLEGUM: 1>,
#  'CAKE': <Type.CAKE: 2>,
#  'BOWLING': <Type.BOWLING: 3>,
#  'ZIPPER': <Type.ZIPPER: 4>,
#  'PLUNGER': <Type.PLUNGER: 5>,
#  'SWITCH': <Type.SWITCH: 6>,
#  'SWATTER': <Type.SWATTER: 7>,
#  'RUBBERBALL': <Type.RUBBERBALL: 8>,
#  'PARACHUTE': <Type.PARACHUTE: 9>,
#  'ANVIL': <Type.ANVIL: 10>}

# Attachment
# {'NOTHING': <Type.NOTHING: 9>,
#  'PARACHUTE': <Type.PARACHUTE: 0>,
#  'ANVIL': <Type.ANVIL: 1>,
#  'BOMB': <Type.BOMB: 2>,
#  'SWATTER': <Type.SWATTER: 3>,
#  'BUBBLEGUM_SHIELD': <Type.BUBBLEGUM_SHIELD: 6>}

class ActionOnlyFlattenerWrapper(ActionObservationWrapper):
    """Flattens actions and observations."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.action_flattener = SpaceFlattener(env.action_space)
        self.action_space = self.action_flattener.space
        
    def observation(self, observation):
        return observation


    def action(self, action):
        discrete_actions = {}
        if not self.action_flattener.only_continuous:
            actions = (
                action if self.action_flattener.only_discrete else action["discrete"]
            )
            # print(actions)
            # print('XXXX', len(self.action_flattener.discrete_keys), len(actions))
            assert len(self.action_flattener.discrete_keys) == len(actions), (
                "Not enough discrete values: "
                f"""expected {len(self.action_flattener.discrete_keys)}, """
                f"""got {len(action)}"""
            )
            discrete_actions = {
                key: key_action
                for key, key_action in zip(self.action_flattener.discrete_keys, actions)
            }

        continuous_actions = {}
        if not self.action_flattener.only_discrete:
            actions = (
                action
                if self.action_flattener.only_continuous
                else action["continuous"]
            )
            continuous_actions = {
                key: actions[
                    self.action_flattener.indices[ix] : self.action_flattener.indices[
                        ix + 1
                    ]
                ].reshape(shape)
                for ix, (key, shape) in enumerate(
                    zip(
                        self.action_flattener.continuous_keys,
                        self.action_flattener.shapes,
                    )
                )
            }

        return {**discrete_actions, **continuous_actions}




class EscapeStuckObservationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        env.observation_space['is_stuck'] = gym.spaces.Box(0, float("inf"), shape=(1,), dtype=np.float32)
        
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        
        self.is_stuck = np.array([0.0])
        observation['is_stuck'] = self.is_stuck
        
        return observation, info
        
    def step(self, action):
        
        observation, reward, terminated, truncated, info = self.env.step(action)


        if self.is_stuck > 0:
            self.is_stuck -= 1.0
        else:
            # if (np.linalg.norm(observation['velocity']) < 0.15):
            if abs(observation['velocity'][2]) < 0.15:
                self.is_stuck = np.array([5.0])
            else:
                self.is_stuck = np.array([0.0])
        observation['is_stuck'] = self.is_stuck
        
        return observation, reward, terminated, truncated, info
    
    
    
class ExpertObservationWrapper(ActionObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        env.observation_space['start_race'] = gym.spaces.Box(0, 1, (1,), dtype=np.float32)
        
        env.observation_space['obstacle_ahead'] = gym.spaces.Box(0, 1, (1,), dtype=np.float32)
        env.observation_space['obstacle_position'] = gym.spaces.Box(-float('inf'), float('inf'), (3,), dtype=np.float32)
        env.observation_space['target_position'] = gym.spaces.Box(-float('inf'), float('inf'), (3,), dtype=np.float32)
        env.observation_space['target_distance'] = gym.spaces.Box(-float('inf'), float('inf'), (1,), dtype=np.float32)
        env.observation_space['target_angle'] = gym.spaces.Box(-float('inf'), float('inf'), (1,), dtype=np.float32)
                
        self.env = env 
    
    
    def angle_between_vectors(self, v1, v2):
        cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))


    def orthogonal_distance(self, P1, vector, P2):
        w = P2 - P1
        cross_product = abs(w[0] * vector[1] - w[1] * vector[0])
        distance = cross_product / np.linalg.norm(vector)    
        return distance
    
    def observation(self, state):
        
        items_position = state['items_position']
        items_type = state['items_type']
        items_distances = [float(np.linalg.norm(item)) for item in items_position]

        items_is_obstacle  = [(item in [1, 4])*1 for item in items_type]
        
        
        obstacle_ahead = np.array([0.0]) #######################
        if any(items_distances[i] < 10 for i in range(5)) and any(items_is_obstacle[:5]):
            obstacle_ahead = np.array([1.0])
            for i in range(5):
                if items_is_obstacle[i]:break
                
            obstacle_position = items_position[i] #######################


        def is_on_circuit(ix):
            path_distance = state['paths_distance']
            return (ix == 0) or (path_distance[ix-1][1] == path_distance[ix][0])

        def get_target_ix(current_ix, segments):

            p1, p2 = segments[current_ix]
            for j in range(1, len(segments)-current_ix):
                if is_on_circuit(current_ix + j):
                    _, p3 = segments[current_ix + j]
                    deviation = float(self.orthogonal_distance(p1, p2 - p1, p3))
                    if deviation > 1:
                        target_ix = current_ix+j
                        return target_ix
                else:
                    break
                        
            if (current_ix != len(segments)-1) and is_on_circuit(current_ix + 1):
                return current_ix+1
            else:
                return current_ix


        # segments = state['path_nodes'][..., [0, 2]]
        segments = np.concatenate((np.expand_dims(state['paths_start'], 1), np.expand_dims(state['paths_end'], 1)), axis=1)[..., [0, 2]]
        target_ix = get_target_ix(0, segments)


        target = state['paths_end'][target_ix] #######################
        target_distance = np.linalg.norm(target).reshape(1) #######################
        
        
            

        # Steer:
        sign = (1 if target[0] >= 0 else -1)
        angle = self.angle_between_vectors(np.array([0, 1]), target[[0, 2]]) * sign  #######################
        angle = angle.reshape(1)

        karts = self.env.unwrapped._env.unwrapped.world.karts
        players = self.env.unwrapped._env.config.players
        for my_ix in range(len(players)):
            if players[my_ix].name == 'Naxim':
                break
        kart = karts[my_ix]

        new_obseration = state
        
        new_obseration['start_race'] = (state['distance_down_track'] < 50)
        
        new_obseration['powerup'] = kart.powerup.type.value
        
        new_obseration['obstacle_ahead'] = obstacle_ahead
        new_obseration['obstacle_position'] = obstacle_position if obstacle_ahead else np.full((3,), -1., dtype=np.float32)
        new_obseration['target_position'] = target
        new_obseration['target_distance'] = target_distance
        new_obseration['target_angle'] = angle
        
        return new_obseration

    
    def action(self, action):
        return action

    
    
    
class ActionTimeExtentionWrapper(ActionObservationWrapper):
    def __init__(self, env: gym.Env, n=5):
        super().__init__(env)
        
        self.n = n
        env.observation_space['previous_actions'] = gym.spaces.Box(-np.inf, np.inf, (self.n, 7), dtype=np.float32)
        self.previous_actions = np.full((self.n, 7), -1, dtype=np.float32)
        self.last_action = None
    
    def action(self, action):
        self.last_action = action
        return action
    
    def observation(self, observation):
        action = self.last_action
        if action:
            flat_action = np.concatenate([action[key] for key in action], axis=-1) if isinstance(action, dict) else action
            self.previous_actions[:-1] = self.previous_actions[1:]
            self.previous_actions[-1] = flat_action
            
        observation['previous_actions'] = self.previous_actions
        return observation