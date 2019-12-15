from game.azul import Azul

import math
import numpy as np

# Class till will help the neural network to run properly
class NNRunner:
    def __init__(self):
        self.game = Azul()
#       NNRunner will keep track of the players relative score, in order to see how much is gained
#       or lost.
        self.player_score = 0
        self.move_counter = 0
# Pseudo code for new functionality that will be added
#   def step(self, i):
#       make action i
#       generate action for opponent, repeate until it is player 1's turn again
#       count score by creating a deep copy of the game board and use the function count_score
#           if that is not possible, we will need to modify the original function not to modofy the table
#       calculate the reward
#       update the player_score
#       return the reward, new state and end_of_game

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
# Pseudo code for new functionality that will be added
#def opponent_random(game):
#   stores all valid moves, by using check_all_valid.
#   multiply all invalid moves with a weight table, to remove the invalid ones.
#   (possibly normalize the weights if needed)
#   return a random integer, as specified by the weight table
