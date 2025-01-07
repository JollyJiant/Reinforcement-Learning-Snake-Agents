import random
import numpy as np
import pygame
# import pickle
import time

# color class
class Color:
    def __init__(self):
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.blue = (50, 150, 255)
        self.green = (200, 255, 0)

class Snake:
    def __init__(self, visualize=True):
        # whether to show episode number at the top
        self.show_episode = False
        self.episode = None
        
        # scale adjusts size of whole board (use 1.0 or 2.0)
        self.scale = 2
        self.game_width = int(600 * self.scale)
        self.game_height = int(400 * self.scale)
        
        # padding for score & episode
        self.padding = int(30 * self.scale)
        self.screen_width = self.game_width
        self.screen_height = self.game_height + self.padding
         
        self.snake_size = int(10 * self.scale)
        self.food_size = int(10 * self.scale)
        self.snake_speed = 40
                 
        self.snake_coords = []
        self.snake_length = 1
        self.dir = "right"
        self.board = np.zeros((self.game_height // self.snake_size, self.game_width // self.snake_size))
        
        self.game_close = False
        self.visualize = visualize
     
        
        # starting location for the snake
        self.x1 = self.game_width / 2
        self.y1 = self.game_height / 2 + self.padding
        
        self.r1, self.c1 = self.coords_to_index(self.x1, self.y1)
        self.board[self.r1][self.c1] = 1
             
        self.c_change = 1
        self.r_change = 0
          
        self.food_r, self.food_c = self.generate_food()
        self.board[self.food_r][self.food_c] = 2
        self.survived = 0
        if self.visualize:
            pygame.init()
            self.color = Color()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height)) 
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("bahnschrift", int(18 * self.scale))
                  
         
        self.last_dir = None
        self.step()
        
    def print_score(self, score):
        value = self.font.render(f"Score: {score}", True, self.color.white)
        self.screen.blit(value, [500 * self.scale, 10])
        
    def print_episode(self):
        if self.show_episode:
            value = self.font.render(f"Episode: {self.episode}", True, self.color.white)
            self.screen.blit(value, [10, 10])
        
    def draw_snake(self):
        for i in range(len(self.snake_coords) - 1, -1, -1):
            r, c = self.snake_coords[i]
            x, y = self.index_to_coords(r, c)
            if i == len(self.snake_coords) - 1:
                # head square color
                pygame.draw.rect(self.screen, self.color.blue, [x, y, self.snake_size, self.snake_size])
            else:
                pygame.draw.rect(self.screen, self.color.green, [x, y, self.snake_size, self.snake_size])
            
    def game_end_message(self):
        mesg = self.font.render("Game over!", True, self.color.red)
        self.screen.blit(mesg, [2 * self.game_width / 5, 2 * self.game_height / 5 + self.padding])
        
    def furthest_danger(self, r, c):
        left = self.board[r][:c]
        left = left[::-1]

        right = self.board[r][c+1:]

        up = self.board[:r, c]
        up = up[::-1]

        down = self.board[r+1:, c]

        directions = [left, right, up, down]
        distances = [np.where(direction == 1) for direction in directions]
        print(distances)

        return np.argmax(distances)

    # is there danger in this square (body or wall)
    def is_unsafe(self, r, c):
        if self.valid_index(r, c):
          if self.board[r][c] == 1:
              return 1
          return 0
        else:
          return 1

    def get_state_from_simulated_action(self, head, food, direction):
        head_c, head_r = head
        food_c, food_r = food
        state = []

        snake_direction = ['left', 'right', 'up', 'down'].index(direction)
        state.append(snake_direction)

        food_direction = ''
        if food_r < head_r: food_direction += 'n'
        if food_r > head_r: food_direction += 's'
        if food_c < head_c: food_direction += 'w'
        if food_c > head_c: food_direction += 'e'

        state.append(['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'].index(food_direction))

        state.append(self.is_unsafe(head_r + 1, head_c)) # down
        state.append(self.is_unsafe(head_r - 1, head_c)) # up
        state.append(self.is_unsafe(head_r, head_c + 1)) # right
        state.append(self.is_unsafe(head_r, head_c - 1)) # left

        return tuple(state)

    # returns tuple of 6 features
    def get_state(self):
        head_r, head_c = self.snake_coords[-1]
        state = []
        
        snake_direction = ['left', 'right', 'up', 'down'].index(self.dir)
        state.append(snake_direction)

        food_direction = ''
        if self.food_r < head_r: food_direction += 'n'
        if self.food_r > head_r: food_direction += 's'
        if self.food_c < head_c: food_direction += 'w'
        if self.food_c > head_c: food_direction += 'e'

        state.append(['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'].index(food_direction))

        # NOTE: previous order was down,up,left,right
        # (Changing it to what it currently is shouldn't change anything, BUT, I'm paranoid and keeping the log just in case)
        state.append(self.is_unsafe(head_r, head_c - 1)) # left
        state.append(self.is_unsafe(head_r, head_c + 1)) # right
        state.append(self.is_unsafe(head_r - 1, head_c)) # up
        state.append(self.is_unsafe(head_r + 1, head_c)) # down

        # TODO: Is there anyway I can have a feature that indicates which direction has the most open space?
        # (to prevent the snake from moving in a direction that has danger not immediately near, then gets stuck in a danger box)
        # state.append(self.furthest_danger(head_r, head_c))

        return tuple(state)
      
    def get_head_food_and_danger(self):
        head_r, head_c = self.snake_coords[-1]
        ["left", "right", "up", "down"]
        [0, 1, 2, 3]
        dangers = []
        if self.is_unsafe(head_r, head_c - 1): dangers.append(0) # left
        if self.is_unsafe(head_r, head_c + 1): dangers.append(1) # right
        if self.is_unsafe(head_r - 1, head_c): dangers.append(2) # up
        if self.is_unsafe(head_r + 1, head_c): dangers.append(3) # down
        return {'head': (head_r, head_c), 'food': (self.food_r, self.food_c), 'dangers': dangers}

    def valid_index(self, r, c):
        return 0 <= r < len(self.board) and 0 <= c < len(self.board[0])
      
    # board coordinates <==> row, column conversions
    def index_to_coords(self, r, c):
        x = c * self.snake_size
        y = r * self.snake_size + self.padding
        return (x, y)
    def coords_to_index(self, x, y):
        r = int((y - self.padding) // self.snake_size)
        c = int(x // self.snake_size)
        return (r, c)
    
    # randomly place food
    def generate_food(self):
        food_c = int(round(random.randrange(0, self.game_width - self.food_size) / self.food_size))
        food_r = int(round(random.randrange(0, self.game_height - self.food_size) / self.food_size))
        if self.board[food_r][food_c] != 0:
            food_r, food_c = self.generate_food()
        return food_r, food_c
    
    def game_over(self):
        return self.game_close
    
    def simulated_step(self, action_index):
        game_close = False # Assume we aren't done with the game
        reward = 0 # default reward of 0

        action = ["left", "right", "up", "down"][action_index]

        # determine the new head position
        c_change = None
        r_change = None
        if action == "left":
            c_change = -1
            r_change = 0
            dir = "left"
        elif action == "right":
            c_change = 1
            r_change = 0
            dir = "right"
        elif action == "up":
            r_change = -1
            c_change = 0
            dir = "up"
        elif action == "down":
            r_change = 1
            c_change = 0
            dir = "down"

        if self.c1 >= self.game_width // self.snake_size or self.c1 < 0 or self.r1 >= self.game_height // self.snake_size or self.r1 < 0:
            game_close = True

        c1 = self.c1 + c_change
        r1 = self.r1 + r_change
        
        for r, c in self.snake_coords[:-1]:
            if r == r1 and c == c1:
                game_close = True

        # determine the reward
        food_r, food_c = self.food_r, self.food_c
        # snake ate the food
        if c1 == self.food_c and r1 == self.food_r:
            food_r, food_c = self.generate_food()
            reward = 5 # food eaten, so +5 reward

        # death = -10 reward
        if game_close:
            reward = -10

        new_state = self.get_state_from_simulated_action((c1, r1), (food_c, food_r), dir)
        
        snake_direction_index = new_state[0]
        food_direction_index = new_state[1]
        snake_direction_cardinal = ['w', 'e', 'n', 's'][snake_direction_index]
        food_direction_cardinal = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'][food_direction_index]
        if snake_direction_cardinal in food_direction_cardinal:
            reward += 5 # minor reward for choosing to move toward the food

        return new_state, reward
        
    def step(self, action="None"):
        if action == "None":
            action = random.choice(["left", "right", "up", "down"])
        else:
            action = ["left", "right", "up", "down"][action]
        
        reward = 0

        if self.visualize:
            for event in pygame.event.get():
                pass
 
        # take action
        self.last_dir = self.dir
        if action == "left" and (self.dir != "right" or self.snake_length == 1):
            self.c_change = -1
            self.r_change = 0
            self.dir = "left"
        elif action == "right" and (self.dir != "left" or self.snake_length == 1):
            self.c_change = 1
            self.r_change = 0
            self.dir = "right"
        elif action == "up" and (self.dir != "down" or self.snake_length == 1):
            self.r_change = -1
            self.c_change = 0
            self.dir = "up"
        elif action == "down" and (self.dir != "up" or self.snake_length == 1):
            self.r_change = 1
            self.c_change = 0
            self.dir = "down"

 
        if self.c1 >= self.game_width // self.snake_size or self.c1 < 0 or self.r1 >= self.game_height // self.snake_size or self.r1 < 0:
            self.game_close = True
        self.c1 += self.c_change
        self.r1 += self.r_change
        
        food_x, food_y = self.index_to_coords(self.food_r, self.food_c)
        if self.visualize:
            self.screen.fill(self.color.black)
            pygame.draw.rect(self.screen, (255, 255, 255), (0, self.padding, self.game_width, self.game_height), 1)
            pygame.draw.rect(self.screen, self.color.red, [food_x, food_y, self.food_size, self.food_size])
        
        self.snake_coords.append((self.r1, self.c1))
        
        if self.valid_index(self.r1, self.c1):
            self.board[self.r1][self.c1] = 1
        
        if len(self.snake_coords) > self.snake_length:
            rd, cd = self.snake_coords[0]
            del self.snake_coords[0]
            if self.valid_index(rd, cd):
                self.board[rd][cd] = 0
 
        for r, c in self.snake_coords[:-1]:
            if r == self.r1 and c == self.c1:
                self.game_close = True
            
        if self.visualize:
            self.draw_snake()
            self.print_score(self.snake_length - 1)
            self.print_episode()
            pygame.display.update()
 
        # snake ate the food
        if self.c1 == self.food_c and self.r1 == self.food_r:
            self.food_r, self.food_c = self.generate_food()
            self.board[self.food_r][self.food_c] = 2
            self.snake_length += 1
            reward = 5 # food eaten, so +5 reward
        self.survived += 1

        # death = -10 reward
        if self.game_close:
            reward = -10
        self.survived += 1

        new_state = self.get_state()
        
        snake_direction_index = new_state[0]
        food_direction_index = new_state[1]
        snake_direction_cardinal = ['w', 'e', 'n', 's'][snake_direction_index]
        food_direction_cardinal = ['n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'][food_direction_index]
        if snake_direction_cardinal in food_direction_cardinal:
            reward += 1 # minor reward for choosing to move toward the food

        return new_state, reward, self.game_close
    
    def q_game(self, episode, table):
        # time.sleep(3)
        if self.visualize:
            self.show_episode = True
            self.episode = episode
            self.print_episode()
            pygame.display.update()
            time.sleep(5)

        # pass in pickle file with q table (stored in directory pickle with file name being episode #.pickle)
        # filename = f"pickle/{episode}.pickle"
        # with open(filename, 'rb') as file:
        #     table = pickle.load(file)
        current_length = 2
        steps_unchanged = 0
        while not self.game_over():
            if self.snake_length != current_length:
                steps_unchanged = 0
                current_length = self.snake_length
            else:
                steps_unchanged += 1
                

            state = self.get_state()
            action = np.argmax(table[state])
            if steps_unchanged == 1000:
                # stop if snake hasn't eaten anything in 1000 episodes (stuck in a loop)
                break
            self.step(action)
            if self.visualize: self.clock.tick(self.snake_speed)
        if self.game_over() and self.visualize == True:
            # snake dies
            self.screen.fill(self.color.black)
            pygame.draw.rect(self.screen, (255, 255, 255), (0, self.padding, self.game_width, self.game_height), 1)
            self.game_end_message()
            self.print_episode()
            self.print_score(self.snake_length - 1)
            pygame.display.update()
            time.sleep(5)
            pygame.quit()
        return self.snake_length
    
    def policy_game(self, episode, policy):
        # time.sleep(3)
        if self.visualize:
            self.show_episode = True
            self.episode = episode
            self.print_episode()
            pygame.display.update()
            time.sleep(5)

        current_length = 2
        steps_unchanged = 0
        while not self.game_over():
            if self.snake_length != current_length:
                steps_unchanged = 0
                current_length = self.snake_length
            else:
                steps_unchanged += 1
                

            state = self.get_state()
            action = policy[state]
            if steps_unchanged == 1000:
                # stop if snake hasn't eaten anything in 1000 episodes (stuck in a loop)
                break
            self.step(action)
            if self.visualize: self.clock.tick(self.snake_speed)
        if self.game_over() and self.visualize == True:
            # snake dies
            self.screen.fill(self.color.black)
            pygame.draw.rect(self.screen, (255, 255, 255), (0, self.padding, self.game_width, self.game_height), 1)
            self.game_end_message()
            self.print_episode()
            self.print_score(self.snake_length - 1)
            pygame.display.update()
            time.sleep(5)
            pygame.quit()
        return self.snake_length

    def ac_game(self, episode, policy_table):
        # time.sleep(3)
        if self.visualize:
            self.show_episode = True
            self.episode = episode
            self.print_episode()
            pygame.display.update()
            time.sleep(5)

        current_length = 2
        steps_unchanged = 0
        while not self.game_over():
            if self.snake_length != current_length:
                steps_unchanged = 0
                current_length = self.snake_length
            else:
                steps_unchanged += 1
                

            state = self.get_state()
            action_probs = policy_table[state]
            action = np.random.choice(4, p=action_probs)
            # action = np.argmax(action_probs) # greedy action choice, take the move with highest likelihood
            if steps_unchanged == 1000:
                # stop if snake hasn't eaten anything in 1000 episodes (stuck in a loop)
                break
            self.step(action)
            if self.visualize: self.clock.tick(self.snake_speed)
        if self.game_over() and self.visualize == True:
            # snake dies
            self.screen.fill(self.color.black)
            pygame.draw.rect(self.screen, (255, 255, 255), (0, self.padding, self.game_width, self.game_height), 1)
            self.game_end_message()
            self.print_episode()
            self.print_score(self.snake_length - 1)
            pygame.display.update()
            time.sleep(5)
            pygame.quit()
        return self.snake_length