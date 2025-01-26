import numpy as np


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
        kart_is_close_ahead = [(kart_distances[i] < 5) and kart_in_front[i] and kart_angles[i] for i in range(len(karts_position))]
        
        
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
            
            # print(angle[0])
            steer = angle[0] / 20
            # print(steer)
            if abs(steer) < 0.05:
                steer = 0
        
            # Speed:
            speed = 1
            if abs(target_position[0]) > 5:
                speed = 0.5
            
            # if steer

            brake = 0
            drift = 0
            
            nitro = 0
            if target_distance > 30:
                nitro = 1
            
            
            
            
            
            powerup = state['powerup']
            fire = 0
            if powerup != 0:
                # print(powerup)
                if powerup in [1, 9, 10]: # BUBBLEGUM, PARACHUTE, ANVIL 
                    fire = 1
                    # print('BUBBLEGUM, PARACHUTE, ANVIL ')
                elif powerup in [2, 3, 5, 8] and any(kart_is_ahead): # CAKE, BOWLING, PLUNGER, RUBBERBALL
                # elif powerup in [2, 3, 5, 8]: # CAKE, BOWLING, PLUNGER, RUBBERBALL
                    fire = 1
                    # print('CAKE, BOWLING, PLUNGER, RUBBERBALL')
                elif powerup in [4, 5, 6, 7]: # Idk what the hell do the other do
                    fire = 1
                    # print('dk what the hell do the other do')
                
            # fire = 1
            action = {
                'continuous': np.array([ speed, steer ]),
                'discrete': np.array([brake, drift, fire, nitro, 1])
            }
        
        return action 