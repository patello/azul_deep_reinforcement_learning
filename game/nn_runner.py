from game.azul import Azul

import math
import random
import copy
import numpy as np

random.seed()

# Class till will help the neural network to run properly
class NNRunner:
    class GameStatistics():
        def __init__(self,nr_of_points=1000):
            self.nr_of_points=nr_of_points
            self.statisticsBuffer = {key : np.empty(0) for key in ["player_score","opponent_score","rounds","percent_first_player","floor_penalty","max_combo","completed_rows","completed_columns","completed_colors"]}
            self.statistics =  {key : np.empty(0) for key in ["player_score","opponent_score","rounds","percent_first_player","floor_penalty","max_combo","completed_rows","completed_columns","completed_colors"]}
        def update(self,statistics):
            for stat in statistics:
                self.statisticsBuffer[stat]=np.append(self.statisticsBuffer[stat],statistics[stat])
                if self.statisticsBuffer[stat].size >= self.nr_of_points:
                    self.statistics[stat] = np.append(self.statistics[stat],self.statisticsBuffer[stat].mean())
                    self.statisticsBuffer[stat] = np.empty(0)
                
    def __init__(self):
        self.game = Azul()
        self.game_statistics = NNRunner.GameStatistics()
        # Start the game with a set board
        self.game.new_round()
        # NNRunner will keep track of the players relative score, in order to see how much is gained
        # or lost.
        self.player_score = 0
        self.move_counter = 0
    def step(self, i):
        self.game.step(*nn_deserialize(i))
        self.move_counter += 1
        while (self.game.current_player != 1 or np.count_nonzero(self.get_valid_moves()) < 2) and not self.game.is_end_of_game():
            self.game.step(*nn_deserialize(opponent_random(self.game)))
            self.move_counter += 1
        game_copy=copy.deepcopy(self.game)
        game_copy.count_score()
        new_player_score = game_copy.score[0]-game_copy.score[1]
        reward = new_player_score-self.player_score
        self.player_score = new_player_score
        return reward, self.game.is_end_of_game()
    def get_state_flat(self):
        return np.concatenate((self.game.game_board_displays.flatten(),self.game.game_board_center,self.game.pattern_lines.flatten(),self.game.walls.flatten(),self.game.floors,self.game.score,[self.game.next_first_player]))
    def get_valid_moves(self):
        #TODO: Write test for this
        return check_all_valid(self.game)
    def reset(self):
        #TODO: Write test for this
        #Changed this to only run parts of initialization, since I want to keep game statistics intact
        self.game = Azul()
        self.game.new_round()
        self.player_score = 0
        self.move_counter = 0
        
# Returns a integer between 1..180
def nn_serialize(display,color,pattern):
    return display+color*6+pattern*5*6

# Takes an integer between 1..180 (TODO: Implement out of bounds check) 
# and returns the corresponding display, color and pattern
def nn_deserialize(i):
    display = i%6
    color = math.floor(i/6)%5
    pattern = math.floor(i/(5*6))
    return (display, color, pattern)

def check_all_valid(game):
    all_valid = np.zeros(180,dtype="bool")
    for i in range(6*5*6):
        all_valid[i]=game.is_legal_move(*nn_deserialize(i))
    return all_valid

def opponent_random(game):
    weight_table=np.ones(180)
    #Weight of the straight to floor moves should be lower than the others
    for i in range(6):
        for j in range(5):
            weight_table[nn_serialize(i,j,0)]=0.01
    all_valid = check_all_valid(game)
    #Weight should be set to zero for invalid moves
    weight_table=np.multiply(weight_table,all_valid)
    return random.choices(range(180),weights=weight_table)[0]
