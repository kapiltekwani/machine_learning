# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

location_mapping = {0:'A',1:'B', 2:'C',3:'D',4:'E'}
day_mapping = {0:'mon',1:'tue',2:'wed',3:'thu',4:'fri',5:'sat',6:'sun'}

req_dis_poisson = [2, 12, 4, 7, 8]


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = self.get_action_space()
        #TODO need to check if we can start with location which has maximum probability of
        #getting passengers, although then it will always result in fix start point.
        self.location = np.random.choice(np.arange(m))
        #TODO atleast here I can start with random day and zero hour(marking start of the month)
        self.hour = np.random.choice(range(t))
        self.day = np.random.choice(range(d))
        #self.state = (self.location, self.hour, self.day)
        
        self.state_space =  self.get_state_space()
       
        self.state_init = (self.location, self.hour, self.day)

        # Start the first round
        self.reset()

    def get_action_space(self):
        list_of_actions = [(d1, d2) for d1 in range(m) for d2 in range(m) if d1 != d2 ]
        list_of_actions.append((0, 0))
        return list_of_actions

    def get_state_space(self):
        state_space = []
        for i in range(m):
            for j in range(7):
                for k in range(24):
                    state_space.append((i,j,k))
         return state_space
                

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""

        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        requests = np.random.poisson(req_dis_poisson[location])

        if requests > 15:
            requests = 15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_idx]

        
        actions.append([0,0])

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        #Reward per hour is R.
        if (action[0] == 0 and action[1] == 0):
            reward = -C
        else :
            #Need to check if start point is not same as the current position 
            if (action[0] != state[0]):
                time_to_reach_start_point = Time-matrix[state[0]][action[0][state[1]][state[2]]
                #now we need to update the time 
                total_hour = state[1] + time_to_reach_start_point
                if (total_hour > 24):
                    new_day =  state[2] + (total_hour/24)
                    new_hour = state[1] + (total_hour % 24)
                time_from_start_to_end = Time-matrix[action[0]][action[1]][new_hour][new_day]
            else:
                time_to_reach_start_point = 0
                time_from_start_to_end = Time-matrix[action[0]][action[1]][state[1]][state[2]]
        
            reward = (R*time_from_start_to_end) - (C * (time_from_start_to_end + time_to_reach_start_point))
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
