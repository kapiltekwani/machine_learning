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

location_mapping = {1:'A',2:'B', 3:'C',4:'D',5:'E'}
day_mapping = {0:'mon',1:'tue',2:'wed',3:'thu',4:'fri',5:'sat',6:'sun'}


req_dis_pos_A=2
req_dis_pos_B=12
req_dis_pos_C=4
req_dis_pos_D=7
req_dis_pos_E=8


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = self.get_action_space()
        print("action_space: %s"%self.action_space)
        
        #TODO need to check if we can start with location which has maximum probability of
        #getting passengers, although then it will always result in fix start point.
        self.location = np.random.choice(np.arange(1, m+1))
        #TODO atleast here I can start with random day and zero hour(marking start of the month)
        self.hour = np.random.choice(range(t))
        self.day = np.random.choice(range(d))
        self.state = np.array([self.location, self.hour, self.day])
        self.action_size = len(self.get_action_space())
        
        self.state_space = self.get_state_space()
        self.state_init =  np.array([self.location, self.hour, self.day])

        # Start the first round
        self.reset()
        
    def get_random_action(self, state):
        (possible_actions_index, actions) = self.requests(state)
        #return random.choice(actions)
        print("possible_actions_index: %s"%possible_actions_index)
        if len(possible_actions_index) == 0:
            idx_length = self.action_size - 1
            return idx_length
        return random.choice(possible_actions_index)
        
    
    def get_action_space(self):
        list_of_actions = [[d1, d2] for d1 in range(1, m+1) for d2 in range(1, m+1) if d1 != d2 ]
        list_of_actions.append([0, 0])
        list_of_actions = np.array(list_of_actions)
        ideal_action_space_len = ((m-1)*m + 1)
        print("Length of action space list_of_actions: %d, ideally it is expected to be ((m-1)*m + 1): %s"%(len(list_of_actions), ideal_action_space_len))
        return list_of_actions

    def get_state_space(self):
        return [(i, j , k) for i in range(1, m+1) for j in range(t) for k in range(d)]
        
                
    #def step(state, action):

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        state_encod = np.zeros(m + t + d)
        state_encod[(state[0] - 1)] = 1
        state_encod[(m + state[1])] = 1
        state_encod[(m + t + state[2])] = 1
       
        
        
        #state_encod = np.array([position, time, day])
        print("state: %s"%state)
        print("encoded state: %s"%state_encod)
        print("size of encoded state: %s"%state_encod.size)
        
        return state_encod
        
    
     ## Encoding state (or state-action) for NN input

    def state_encod_arch1_old(self, state):
        
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        position = np.zeros(m)
        time = np.zeros(t)
        day = np.zeros(d)
        
        position[(state[0] - 1)] = 1
        time[state[1]] = 1
        day[state[2]] = 1
        
        state_encod = np.array([position, time, day])
        print("state: %s"%state)
        print("encoded state: %s"%state_encod)
        print("size of encoded state: %s"%state_encod.size)
        
        return state_encod
        
        #state_encod = [0]*(m+t+d)
        #state_encod[state[0]] = 1
        #state_encod[state[1]] = 1
        #state_encod[state[2]] = 1
        #eturn state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        #TODO, as per the comments mentioned above poistions vary from 1 to m, whereas
        #in the dummy code they have checked it for  location 0 too, need to verify.
        if location == 1:
            requests = np.random.poisson(req_dis_pos_A)
        if location == 2:
            requests = np.random.poisson(req_dis_pos_B)
        if location == 3:
            requests = np.random.poisson(req_dis_pos_C)
        if location == 4:
            requests = np.random.poisson(req_dis_pos_D)
        if location == 5:
            requests = np.random.poisson(req_dis_pos_E)

        if requests > 15:
            requests = 15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append([0,0])

        return possible_actions_index,actions   



    def reward_func(self, state, action_idx, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        #Reward per hour is R.
        cost_for_pckup = 0
        reward = 0
        time_spend_for_pckup = 0
        current_location = state[0]
        current_time = state[1]
        current_day = state[2]

        action = self.action_space[action_idx]

        pickup_location = action[0]
        drop_location = action[1]

        if pickup_location == drop_location:
            return reward

        if current_location != pickup_location:
            time_spend_for_pckup = Time_matrix[current_location - 1][pickup_location - 1][current_time][current_day]
            cost_for_pckup = C*time_spend_for_pckup

        #now the pickup_location will be the current_location and current_time will be current_time + time_spend_for_pckup.
        total_time = current_time + time_spend_for_pckup
        if total_time >= 24:
            current_day += int(total_time/t)
            if current_day >= d:
                current_day = current_day % 7
            current_time = current_time % t

        time_for_drop = Time_matrix[pickup_location - 1][drop_location - 1][current_time][current_day]
        cost_for_drop = C*time_for_drop
        reward_for_drop = R*time_for_drop

        total_cost_for_drop = cost_for_pckup + cost_for_drop
        reward = reward_for_drop - total_cost_for_drop
        return reward

    def inc_time_and_date(self, current_time, current_day, time_duration):
        final_time = current_time + int(time_duration)
        final_day = current_day
        if final_time >= 24:
            final_day = current_day + int(final_time/t)
            if final_day >= d:
                final_day = int(final_day % 7)
            final_time = int(final_time % t)

        return (final_time, final_day)

        
        

    def next_state_func(self, state, action_idx, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        current_position = state[0]
        current_time = state[1]
        current_day = state[2]
    
        action = self.action_space[action_idx]
        
        pickup_location = action[0]
        drop_location = action[1]
        final_time = current_day
        next_state = None        

        if pickup_location == drop_location:
            (final_time, final_day) = self.inc_time_and_date(current_time, current_day, 1)
            next_state = np.array([current_position, final_time, final_day])
            return next_state

        time_spend_for_pckup = Time_matrix[current_position - 1][pickup_location - 1][current_time][current_day]
        #print(current_time, current_day)
        (pickup_time, pickup_day) = self.inc_time_and_date(current_time, current_day, time_spend_for_pckup)
        #print(pickup_time, pickup_day)
        
        time_for_drop = Time_matrix[pickup_location - 1][drop_location - 1][pickup_time][pickup_day]
        (final_drop_time, final_drop_day) = self.inc_time_and_date(pickup_time, pickup_day, time_for_drop)
       
        next_state = np.array([drop_location, final_drop_time, final_drop_day]) 
        return next_state

    def step(self, state, action, Time_matrix):
        reward = self.reward_func(state, action, Time_matrix)
        next_state = self.next_state_func(state, action, Time_matrix)

        return (next_state, reward)
        



    def reset(self):
        return self.action_space, self.state_space, self.state_init
