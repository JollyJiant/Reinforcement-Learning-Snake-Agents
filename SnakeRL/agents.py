import numpy as np
# import pickle
import random
from snake_game import Snake
import time

MAX_ITER = 10000
GAMMA = 0.9

class QLearningAgent():
    def __init__(self):
        # define initial parameters
        self.learning_rate = 0.01
        self.eps = 0.75
        self.eps_discount = 0.9992
        self.min_eps = 0.001
        self.num_episodes = MAX_ITER
        self.table = np.zeros((4, 8, 2, 2, 2, 2, 4))
        self.score = []
        self.survived = []
        
    # epsilon-greedy action choice
    def get_action(self, state):
        # select random action (exploration)
        if random.random() < self.eps:
            return random.choice([0, 1, 2, 3])
        
        # select best action (exploitation)
        return np.argmax(self.table[state])
    
    def train(self):
        print("Q-Learning")
        start = time.time()

        for i in range(1, MAX_ITER+1):
            game  = Snake(visualize=False)
            steps_without_food = 0
            length = game.snake_length
            
            # print updates
            # if i % 1000 == 0:
            #     print(f"Episodes: {i}, score: {np.mean(self.score)}, survived: {np.mean(self.survived)}, eps: {self.eps}, lr: {self.learning_rate}")
            #     self.score = []
            #     self.survived = []
            
            current_state = game.get_state()
            self.eps = max(self.eps * self.eps_discount, self.min_eps)
            done = False
            # TODO: change the way this runs to stop after Q-Table convergence
            while not done:
                Q_prev = self.table.copy()
                # choose action and take it, observing reward and change of state
                action = self.get_action(current_state)
                new_state, reward, done = game.step(action)
                
                delta = reward + GAMMA*self.table[new_state][action] - self.table[current_state][action]
                self.table[current_state][action] += self.learning_rate * delta
                
                current_state = new_state
                
                steps_without_food += 1
                if length != game.snake_length:
                    length = game.snake_length
                    steps_without_food = 0
                if steps_without_food == 1000:
                    done = True
            
            # keep track of validation metrics
            self.score.append(game.snake_length - 1)
            self.survived.append(game.survived)

            if np.allclose(Q_prev, self.table): break

        end = time.time()
        print(f'Took {end-start} seconds to train')
        
        return i, self.table, self.score, self.survived

class PolicyIterationAgent():
    def __init__(self):
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.action_indices = [0, 1, 2, 3]
        self.V = np.zeros((4, 8, 2, 2, 2, 2))
        # print('load all states')
        self.states = [tuple(x) for x in np.ndindex(self.V.shape)]
        # print('load initial policies')
        self.policy = {tuple(x): random.choice(self.action_indices) for x in self.states}

        self.evaulation = {tuple(x): {'g': 0, 'next_state': (0, 0, 0, 0, 0, 0)} for x in self.states}

        self.num_episodes = MAX_ITER
        self.epsilon = 0.001

    # TODO: need to change reward so that Snake is incentivized to move toward the food
    # OR
    # need to introduce noise (random exploration)
    def train(self):
        print("Policy Iteration")
        start = time.time()
        
        # generate candidate policy NOTE: done in __init__
        
        i = 1
        while i != MAX_ITER+1:
            if i % 500 == 0: print(i)
            V_prev = self.V.copy()

            # update V based on the policy
            # V_i(x_k) = E[g(x_k, π_i(x_k))] + γ * E[V_i(x_k+1)]
            for state in self.states:
                eval = self.evaulation[state]
                g = eval['g']
                next_state = eval['next_state']
                self.V[state] = g + GAMMA*self.V[next_state]

            # if V isn't 0 for all states (happens during the 0th run)
            if not np.all(self.V == 0): 
                if np.allclose(V_prev, self.V):
                    break

            # NOTE: trajectory based; plays N games of snake for X steps
            for n in range(20): # across 20 games
                game = Snake(visualize=False)
                for x in range(250): # determine the best action for 250 steps (or until Snake hits something)
                    state = game.get_state()
                    # figure out the best action for this state
                    best_value = -np.inf
                    best_action = None
                    best_new_state = None
                    best_reward = None
                    for action in self.action_indices:
                        # NOTE: need to do a simulated step here so I don't update the actual game until the optimal action
                        new_state, reward = game.simulated_step(action)
                        value = reward + GAMMA*self.V[new_state]
                        if value > best_value:
                            best_value = value
                            best_action = action
                            best_new_state = new_state
                            best_reward = reward
                    # load values for future evaluation
                    self.evaulation[state] = {'g': best_reward, 'next_state': best_new_state}
                    # take the best action
                    new_state, reward, done = game.step(best_action)
                    self.policy[state] = best_action
                    if done: break
            
            i += 1

        end = time.time()
        print(f'Took {end-start} seconds to train')
        
        return i, self.policy

class ActorCriticAgent():
    def __init__(self):
        # Initialize policy and value tables
        self.num_actions = 4
        self.value_table = np.zeros((4, 8, 2, 2, 2, 2))  # Initial value estimates
        self.policy_table = np.ones((4, 8, 2, 2, 2, 2, self.num_actions)) / self.num_actions  # Uniform policy

        # Parameters
        self.alpha = 0.01  # Learning rate for critic
        self.beta = 0.01   # Learning rate for actor

        self.score = []
        self.survived = []

    def train(self):
        print("Actor-Critic")
        start = time.time()

        for i in range(1, MAX_ITER+1):
            game = Snake(visualize=False)
            steps_without_food = 0
            length = game.snake_length
            done = False
            
            while not done:
                # get the current state of the game
                state = game.get_state()
                
                # Select action based on policy
                action_probs = self.policy_table[state]
                action = np.random.choice(self.num_actions, p=action_probs)
                
                # Take action in the environment (updates the environment too)
                next_state, reward, done = game.step(action)
                
                # Compute TD error
                td_error = reward + GAMMA * self.value_table[next_state] * (1 - done) - self.value_table[state]
                
                # Update value table (critic)
                self.value_table[state] += self.alpha * td_error
                
                # Update policy table (actor)
                self.policy_table[state + (action,)] += self.beta * td_error
                self.policy_table[state] = np.clip(self.policy_table[state], 0, 1)  # Ensure probabilities are valid
                self.policy_table[state] /= np.sum(self.policy_table[state])  # Normalize to form a valid probability distribution

                steps_without_food += 1
                if length != game.snake_length:
                    length = game.snake_length
                    steps_without_food = 0
                if steps_without_food == 1000:
                    done = True

            # keep track of validation metrics
            self.score.append(game.snake_length - 1)
            self.survived.append(game.survived)
        
        end = time.time()
        print(f'Took {end-start} seconds to train')

        return i, self.policy_table, self.score, self.survived