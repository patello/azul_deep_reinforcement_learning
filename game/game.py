import numpy as np
import random
import sys

random.seed()

class Azul:
    def __init__(self,players=2):
        self.game_board_displays = np.zeros((5,5),dtype=np.int)
        self.game_board_center = np.zeros(6,dtype=np.int)
        pattern_lines_prototype = np.zeros((5,5),dtype=np.int)
        self.pattern_lines=np.repeat(pattern_lines_prototype[np.newaxis,:, :], players, axis=0)
        walls_prototype = np.zeros((5,5),dtype=np.bool)
        self.walls=np.repeat(walls_prototype[np.newaxis,:, :], players, axis=0)
        self.floors=np.zeros((players,7),dtype=bool)
        self.score=np.zeros(players,dtype=int)
        self.first_player=1
    def new_round(self):
        #Game board center starts empty except for the first player token (sixth index).
        self.game_board_center=[0,0,0,0,0,1]
        #Factory displays have 4 random tiles (TODO: respect total number of tiles)
        self.game_board_displays=np.zeros((5,5),dtype=np.int)
        for i in range(5):
            for j in range (4):
                self.game_board_displays[i,random.randrange(0,4,1)] += 1

if __name__ == "__main__":
    game=Azul()
    game.new_round()
    print(str(game.game_board_displays))

# TESTS Move these later

def test_azul_init():
    # game_board_displays should be of size (5,5).
    # TODO: Size should change depending on nr of players
    assert Azul().game_board_displays.shape == (5,5)
    # game_board_center should be of size 6
    assert Azul().game_board_center.shape == (6,)
    # pattern__lines should be of size (nr of players,5,5)
    for i in range(2,5):
        assert Azul(players=i).pattern_lines.shape == (i,5,5)
        assert np.count_nonzero(Azul(players=i).pattern_lines) == 0
    # walls should be of size (nr of players,5,5) and be empty
    for i in range(2,5):
        assert Azul(players=i).walls.shape == (i,5,5)
        assert np.count_nonzero(Azul(players=i).walls) == 0
    # floor should be of size (nr of players,7)
    for i in range(2,5):
        assert Azul(players=i).floors.shape == (i,7)
    # score should be of size (nr of players,) and be zero
    for i in range(2,5):
        assert Azul(players=i).score.shape == (i,)
        assert np.count_nonzero(Azul(players=i).score) == 0

def test_azul_new_round():
    game=Azul()
    game.new_round()
    #After new round, no display should have more than 4 tiles
    for tile in game.game_board_displays:
        assert np.sum(tile) == 4
    #After new round, game board center should have zero color tiles
    assert np.count_nonzero(game.game_board_center[:5]) == 0
    #After new round, game board center should have one white tile
    assert game.game_board_center[5] == 1