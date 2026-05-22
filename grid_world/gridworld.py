# import numpy and random
import numpy as np
import random

class GridWorld:

    def __init__(self,
                 size,
                 battery_pos,
                 obstacles):
        '''
        Initializes a GridWorld environment.

        size: int, the size of the grid (size x size)
        battery_pos: (x,y) tuple, the position of the battery
        obstacles: list of (x,y) tuples, the positions of the obstacles
        '''
        self.actions = ['Down','Up','Left','Right']
        self.num_actions = len(self.actions)
        self.battery_pos = battery_pos
        self.obstacles = set(obstacles)        
        self.size = size
        self.num_obstacles = len(obstacles)


    def states(self):
        '''
        Returns a set of all valid states in the gridworld (i.e., all positions that are not obstacles).
        '''
        states = set()
        for i in range(self.size):
            for j in range(self.size):
                if (i,j) not in self.obstacles:
                    states.add((i,j))
        return states

    
    def successors(self, state):
        '''
        Returns a set of valid successor states that can be reached from the given state by taking any of the possible actions.
        I.e., all states s' with sum_a p(s'|s,a) > 0.
        '''
        valid_states = self.states()
        successors = set()
        
        for valid_state in valid_states:
            sum_a = 0
            for action in range(self.num_actions):
                sum_a += self.p(valid_state, state, action)
            if sum_a > 0:
                successors.add(valid_state)
        
        return successors

    def p(self, successor, state, action):
        '''
        Returns the probability of transitioning from state to successor
        using the given action.
        '''
        if(self.actions[action] == 'Down'):
            if (state[0]+1, state[1]) == successor and (state[0]+1, state[1]) in self.states():
                return 1.0
            elif state == successor and (state[0]+1, state[1]) not in self.states():
                return 1.0
        elif(self.actions[action] == 'Up'):
            if (state[0]-1, state[1]) == successor and (state[0]-1, state[1]) in self.states():
                return 1.0
            elif state == successor and (state[0]-1, state[1]) not in self.states():
                return 1.0
        elif(self.actions[action] == 'Left'):
            if (state[0], state[1]-1) == successor and (state[0], state[1]-1) in self.states():
                return 1.0
            elif state == successor and (state[0], state[1]-1) not in self.states():
                return 1.0
        elif(self.actions[action] == 'Right'):
            if (state[0], state[1]+1) == successor and (state[0], state[1]+1) in self.states():
                return 1.0
            elif state == successor and (state[0], state[1]+1) not in self.states():
                return 1.0
        return 0 

    def step(self, state, action):
        '''
        Given a state and an action, returns the successor for the transition under action. 
        '''        
        possible_next_states = list(self.successors(state))
        
        if not possible_next_states:
            return state
            
        probs = [self.p(s_next, state, action) for s_next in possible_next_states]
        
        # random.choices erlaubt uns das Sampling mit Gewichten.
        # Da es eine Liste zurückgibt, nehmen wir das erste Element [0].
        return random.choices(possible_next_states, weights=probs, k=1)[0]
    
    def reward(self, successor, state, action):
        '''
        Returns the reward for transitioning from state to successor
        using the given action.
        '''

        if(state != self.battery_pos and successor == self.battery_pos):
            r = 1
        else:
            r = 0

        return r
    
    def render(self, state):
        '''
        Prints the gridworld to the console, with the agent's current position marked as 'X', 
        obstacles marked as '#', and the battery marked as 'B'.
        '''
        grid = np.full((self.size,self.size), '_')
        print("\n")
        for obs in self.obstacles: 
            grid[*obs] = '#'
        grid[*self.battery_pos] = 'B'
        grid[*state] = 'X'
        print("\n".join("".join(row) for row in grid))
        print("\n")