# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5  # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        from_loc = [0,1,2,3,4]
        to_loc   = [0,1,2,3,4]
        # action space consists of possible combinations of start and end locations.
        # possible combinations (m * m-1) + 1 locations
        # ****************** make the code dynamic
        self.action_space = [(a,b) for a in from_loc for b in to_loc if a!=b or a==0]
        self.loc = np.random.choice((0,1,2,3,4))
        self.time = np.random.choice((0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23))
        self.day = np.random.choice((0,1,2,3,4,5,6))
        self.state_space = [[i, j, k] for i in range(0, m) for j in range(t) for k in range(d)]
        self.state_init = [self.loc, self.time, self.day]
        self.state_size = m + t +d;

        # Start the first round
        self.reset()

    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = np.zeros(m+t+d)
        state_encod[state[0]] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        return state_encod

    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)

        if requests > 15:
            requests = 15
        # out of (m-1)*m + 1 options take 'x' number of random options where 'x' is the number of requests
        # (0,0) is not considered as customer request hence, starting the range from 1 and not 0
        possible_actions_index = random.sample(range(1, m*(m-1) +1), requests) 
        actions = [self.action_space[i] for i in possible_actions_index]
        # add the 'no-ride' option
        actions.append([0,0])

        return possible_actions_index,actions

    def inc_time_and_date(self, current_time, current_day, time_duration):
        final_time = current_time + int(time_duration)
        final_day = current_day
        if final_time >= 24:
            final_day = current_day + int(final_time/t)
            if final_day >= d:
                final_day = int(final_day % 7)
            final_time = int(final_time % t)

        return (final_time, final_day)

    def reward_func(self, state, action_index, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward""" 
        cost_for_pckup = 0
        reward = 0
        time_spend_for_pckup = 0
        current_location = state[0]
        current_time = state[1]
        current_day = state[2]

        action = self.action_space[action_index]

        pickup_location = action[0]
        drop_location = action[1]

        # action = (0,0) is when the user chooses the option of 'no_ride'
        if pickup_location == drop_location and pickup_location == 0:
            return C * (-1)

        # action has (p,q) and state has (i,d,h) where request is to go from p to q and current location of driver is i
        # R is the revenue per hour and C is the fuel and other cost per hour
        if current_location != pickup_location:
            time_spend_for_pckup = Time_matrix[current_location][pickup_location][current_time][current_day]
            cost_for_pckup = C*time_spend_for_pckup

        #now the pickup_location will be the current_location and current_time will be current_time + time_spend_for_pckup.
        total_time = current_time + time_spend_for_pckup
        if total_time >= 24:
            current_day += int(total_time/t)
            if current_day >= d:
                current_day = current_day % 7
            current_time = current_time % t

        time_for_drop = Time_matrix[pickup_location][drop_location][current_time][current_day]
        cost_for_drop = C*time_for_drop
        reward_for_drop = R*time_for_drop

        total_cost_for_drop = cost_for_pckup + cost_for_drop
        reward = reward_for_drop - total_cost_for_drop
        return reward

    def next_state_func(self, state, action_index, Time_matrix):
        """Takes state and action as input and returns next state"""

        current_position = state[0]
        current_time = state[1]
        current_day = state[2]
        action = self.action_space[action_index]

        pickup_location = action[0]
        drop_location = action[1]
        final_time = current_day
        next_state = None

        if pickup_location == drop_location:
            (final_time, final_day) = self.inc_time_and_date(current_time, current_day, 1)
            next_state = np.array([current_position, final_time, final_day])
            return next_state

        time_spend_for_pckup = Time_matrix[current_position][pickup_location][current_time][current_day]
        #print(current_time, current_day)
        (pickup_time, pickup_day) = self.inc_time_and_date(current_time, current_day, time_spend_for_pckup)
        #print(pickup_time, pickup_day)

        time_for_drop = Time_matrix[pickup_location][drop_location][pickup_time][pickup_day]
        (final_drop_time, final_drop_day) = self.inc_time_and_date(pickup_time, pickup_day, time_for_drop)

        next_state = np.array([drop_location, final_drop_time, final_drop_day])
        total_trip_time = time_spend_for_pckup + time_for_drop
        return next_state

    def step(self, state, action, Time_matrix):
        reward = self.reward_func(state, action, Time_matrix)
        next_state = self.next_state_func(state, action, Time_matrix)

        return [next_state, reward]

    def reset(self):
        return self.action_space, self.state_space, self.state_init

    #  if the cab driver does not choose any of the request and Termination condition - 30*24
