import gymnasium as gym
from pystk2_gymnasium.wrappers import SpaceFlattener
from pystk2_gymnasium.definitions import ActionObservationWrapper
import numpy as np



class ExpertAgent1:
    def __init__(self):
        
        self.actions_buffer = []
        # self.actions_buffer += [np.array([0, 0, 0, 0, 0, 0, 3])]*19
        # self.actions_buffer += [np.array([4, 0, 0, 0, 0, 0, 3])]*5
            
    
    def angle_between_vectors(self, v1, v2):
        cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))
    
    
    def orthogonal_distance(self, P1, vector, P2):
        w = P2 - P1
        cross_product = abs(w[0] * vector[1] - w[1] * vector[0])
        distance = cross_product / np.linalg.norm(vector)    
        return distance
    
    
    def get_action(self, state):
        
        velocity = np.linalg.norm(state['velocity'])
        
        items_position = state['items_position']
        items_type = state['items_type']
        items_distances = [float(np.linalg.norm(item)) for item in items_position]

        items_is_obstacle  = [(item in [1, 4])*1 for item in items_type]

        
        
        #### Action: ############################################################################vvv
        
        is_stuck = (velocity < 0.15)
        
        
        if len(self.actions_buffer) == 0: 
            
            # Case when the car is stuck
            if is_stuck:
                for _ in range(5):
                    self.actions_buffer.append(np.array([0, 1, 0, 0, 0, 0, 0]))
            
            
            # Case when thre are some objects that are obstacles ahead
            if len(self.actions_buffer) == 0:
                if any(items_distances[i] < 10 for i in range(5)) and any(items_is_obstacle[:5]):
                    for i in range(5):
                        if items_is_obstacle[i]:break
                        
                    obstacle_position = items_position[i]
                    
                    sign = (1 if obstacle_position[0] >= 0 else -1)
                    angle = self.angle_between_vectors(np.array([0, 1]), obstacle_position[[0, 2]]) * sign
                    
                            
                    if 0 < obstacle_position[0] < 2 and obstacle_position[2] > 2:
                        action = np.array([2, 0, 0, 0, 0, 0, 2])
                        self.actions_buffer.append(action)
                        
                    elif -2 < obstacle_position[0] <= 0 and obstacle_position[2] > 2:
                        action = np.array([2, 0, 0, 0, 0, 0, 4])
                        self.actions_buffer.append(action)
                    
                

            if len(self.actions_buffer) == 0:
                
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
                
                
                target = state['paths_end'][target_ix]
                target_distnace = np.linalg.norm(target)
                    
                
                # Steer:
                sign = (1 if target[0] >= 0 else -1)
                angle = self.angle_between_vectors(np.array([0, 1]), target[[0, 2]]) * sign
                
                if angle > 5:
                    steer = 6
                elif 3 < angle <= 5:
                    steer = 5
                elif -5 < angle <= -3:
                    steer = 1
                elif angle < -5:
                    steer = 0
                else:
                    steer = 3
                
                # Speed:
                speed = 4
                if abs(target[0]) > 5:
                    speed = 2

                brake = 0
                drift = 0
                
                
                fire = 1
                # Nitro:
                nitro = 0
                if target_distnace > 30:
                    nitro = 1
                    
                
                
                    
                action = np.array([speed, brake, drift, fire, nitro, 1, steer])
                self.actions_buffer.append(action)        
        
        return self.actions_buffer.pop(0)



class ExpertAgent2:
    def __init__(self):
        pass
    
    def get_action(self, state):
        
        action = None
        if state['is_stuck']:
            action = np.array([0, 1, 0, 0, 0, 0, 0])
        
        if action is None and state['obstacle_ahead']:
            obstacle_position = state['obstacle_position']
            if 0 < obstacle_position[0] < 2 and obstacle_position[2] > 2:
                action = np.array([2, 0, 0, 0, 0, 0, 2]) 
                
            elif -2 < obstacle_position[0] <= 0 and obstacle_position[2] > 2:
                action = np.array([2, 0, 0, 0, 0, 0, 4])


        if action is None:
            target_position = state['target_position']
            angle = state['target_angle']
            target_distance = state['target_distance']
            
            if angle > 5:
                steer = 6
            elif 3 < angle <= 5:
                steer = 5
            elif -5 < angle <= -3:
                steer = 1
            elif angle < -5:
                steer = 0
            else:
                steer = 3
            
            # Speed:
            speed = 4
            if abs(target_position[0]) > 5:
                speed = 2

            brake = 0
            drift = 0
            fire = 1
            # Nitro:
            nitro = 0
            if target_distance > 30:
                nitro = 1
            
            action = np.array([speed, brake, drift, fire, nitro, 1, steer])
        
        return action 
    
    
    
def angle_between_vectors(v1, v2):
        cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

class ExpertAgent3:
    def __init__(self):
        pass
    
    def get_action(self, state):
        
        karts_position = state['karts_position']
        kart_distances = np.linalg.norm(karts_position, axis=1)
        kart_in_front = [kart[2] > 0 for kart in karts_position]
        kart_angles = [angle_between_vectors(np.array([0., 1.]), kart[[0, 2]]) for kart in karts_position]
        kart_is_ahead = [(kart_distances[i] < 30) and kart_in_front[i] and kart_angles[i] and kart_angles[i] < 30  for i in range(len(karts_position))]
        any_kart_is_ahead = any(kart_is_ahead)
        
        action = None
        
        not_finished_going_backward = not (state['previous_actions'][-5:, 2] == 1).all()
        is_going_backward = (state['previous_actions'][-1, 2] == 1).any() and not_finished_going_backward
        is_stuck = abs(state['velocity'][2]) < 0.15 and not_finished_going_backward
        # print(is_going_backward, is_stuck)
        
        # print(not_finished_going_backward, is_going_backward, is_stuck)
        
        if not state['start_race'] and (is_going_backward or is_stuck) :
            action = {
                'continuous': np.array([0.0, 0.0]),
                'discrete': np.array([1, 0, 0, 0, 1])
            }
        
        
        if action is None and state['obstacle_ahead']:
            obstacle_position = state['obstacle_position']
            if 0 < obstacle_position[0] < 2 and obstacle_position[2] > 2:
                action = {
                    'continuous': np.array([0.5, -0.5]),
                    'discrete': np.array([0, 0, 0, 0, 1])
                }
                
            elif -2 < obstacle_position[0] <= 0 and obstacle_position[2] > 2:                
                action = {
                    'continuous': np.array([0.5, 0.5]),
                    'discrete': np.array([0, 0, 0, 0, 1])
                }
                
                
        # if action is None and any(kart_is_close_ahead) :
            
        #     kart_position = karts_position[0]
            
        #     if 0 < kart_position[0] < 1 and kart_position[2] > 1:
        #         action = {
        #             'continuous': np.array([1, -0.5]),
        #             'discrete': np.array([0, 0, 0, 0, 1])
        #         }
                
        #     elif -1 < kart_position[0] <= 0 and kart_position[2] > 1:                
        #         action = {
        #             'continuous': np.array([1, 0.5]),
        #             'discrete': np.array([0, 0, 0, 0, 1])
        #         }
                

        if action is None:
            target_position = state['target_position']
            angle = state['target_angle']
            target_distance = state['target_distance']
            
            # Speed:
            steer = angle[0] / 20
            if abs(steer) < 0.05:
                steer = 0
        
            # Speed:
            speed = 1
            if abs(target_position[0]) > 5:
                speed = 0.5
            
            # Nitro
            nitro = 0
            if target_distance > 30:
                nitro = 1
            
            
            # Fire
            powerup = state['powerup']
            fire = 0
            if powerup != 0:
                # print(powerup)
                if powerup in [1, 9, 10] or powerup in [4, 5, 6, 7]: # BUBBLEGUM, PARACHUTE, ANVIL or # Idk what the hell do the other do
                    fire = 1
                elif powerup in [2, 3, 5, 8] and any_kart_is_ahead: # CAKE, BOWLING, PLUNGER, RUBBERBALL
                    fire = 1


            action = {
                'continuous': np.array([ speed, steer ]),
                'discrete': np.array([0, 0, fire, nitro, 1])
            }
        
        return action 