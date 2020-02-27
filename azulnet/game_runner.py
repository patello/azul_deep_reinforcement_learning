import numpy as np
import random
import math
import torch
import copy

from azulnet.azul import Azul

class GameRunner:
    class GameStatistics():
        def __init__(self):
            self.statisticsBuffer = {key : np.empty(0) for key in ["player_score","opponent_score","rounds","percent_first_player","floor_penalty","max_combo","completed_rows","completed_columns","completed_colors","win_percent"]}
            self.statistics =  {key : np.empty(0) for key in ["player_score","opponent_score","rounds","percent_first_player","floor_penalty","max_combo","completed_rows","completed_columns","completed_colors","win_percent"]}
        def update(self,statistics):
            for stat in statistics:
                self.statisticsBuffer[stat]=np.append(self.statisticsBuffer[stat],statistics[stat])
        def get_stats(self):
            for stat in self.statistics:
                if len(self.statisticsBuffer[stat]) > 0:
                    self.statistics[stat] = np.append(self.statistics[stat],self.statisticsBuffer[stat].mean())
                    self.statisticsBuffer[stat] = np.empty(0)
            return self.statistics
    def __init__(self,opponent=None,rules={"first_player":"Random","tile_pool":"Lid"}):
        self.game = Azul(rules=rules)
        self.rules = rules
        self.game_statistics = GameRunner.GameStatistics()
        if opponent is not None:
            self.opponent=opponent
        else:
            self.opponent=RandomAgent()
        # Start the game with a set board
        self.game.new_round()
        # NNRunner will keep track of the players relative score, in order to see how much is gained
        # or lost.
        self.player_score = 0
        self.move_counter = 0
    def opponent_move(self):
        state=self.get_state_flat(perspective=self.game.current_player-1)
        valid_moves = torch.from_numpy(self.get_valid_moves().reshape(1,180))
        action = self.opponent.get_a_output(state,valid_moves)
        self.game.step(*nn_deserialize(action))
        self.move_counter += 1
    def step(self, i):
        self.game.step(*nn_deserialize(i))
        self.move_counter += 1
        while (self.game.current_player != 1 or np.count_nonzero(self.get_valid_moves()) < 2) and not self.game.is_end_of_game():
            self.opponent_move()
        game_copy=copy.deepcopy(self.game)
        game_copy.count_score()
        new_player_score = game_copy.score[0]-game_copy.score[1]
        reward = new_player_score-self.player_score
        self.player_score = new_player_score
        return reward, self.game.is_end_of_game()
    def get_state_flat(self,perspective=0):
        order = [perspective] + list(set(range(self.game.players))-set([perspective]))
        if self.game.next_first_player > 0:
            perspective_next_first_player = ((self.game.next_first_player - 1 - perspective) % self.game.players) + 1
        else:
            perspective_next_first_player = 0
        #Need to make sure this works even for more players. The code below can be used.
        #print(order)
        #print("Next: "+str(self.game.next_first_player)+", Perspective: "+str(perspective)+", From perspective: "+str(perspective_next_first_player))
        return np.concatenate((
            self.game.game_board_displays.flatten(),
            self.game.game_board_center,
            self.game.pattern_lines[order].flatten(),
            self.game.walls[order].flatten(),
            self.game.floors[order],
            self.game.score[order],
            [perspective_next_first_player]))
    def get_valid_moves(self):
        #TODO: Write test for this
        return check_all_valid(self.game)
    def reset(self):
        #TODO: Write test for this
        #Changed this to only run parts of initialization, since I want to keep game statistics intact
        self.game = Azul(rules=self.rules)
        self.game.new_round()
        self.player_score = 0
        self.move_counter = 0

class RandomAgent():
    def __init__(self):
        self.weight_table=np.ones(180)
        #Weight of the straight to floor moves should be lower than the others
        for i in range(6):
            for j in range(5):
                self.weight_table[nn_serialize(i,j,0)]=0.01
    def get_a_output(self,state,valid_moves):
        #Weight should be set to zero for invalid moves
        valid_weight_table=np.multiply(self.weight_table,valid_moves.numpy()[0])
        return random.choices(range(180),weights=valid_weight_table)[0]

                      
        
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

