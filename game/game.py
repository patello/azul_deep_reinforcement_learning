import numpy as np
import random
import sys

random.seed()

class azul:
    def __init__(self,players=2):
        self.gameBoardDisplays = np.zeros((5,5),dtype=np.int)
        self.gameBoardCenter = np.zeros(6,dtype=np.int)
        patternLinesPrototype = np.zeros((5,5),dtype=np.int)
        self.patternLines=np.repeat(patternLinesPrototype[np.newaxis,:, :], players, axis=0)
        wallsPrototype = np.zeros((5,5),dtype=np.bool)
        self.walls=np.repeat(wallsPrototype[np.newaxis,:, :], players, axis=0)
        self.floors=np.zeros((players,7),dtype=bool)
        self.score=np.zeros(players,dtype=int)
        self.firstPlayer=1
    def new_round(self):
        #Game board center starts empty except for the first player token (sixth index).
        self.gameBoardCenter=[0,0,0,0,0,1]
        #Factory displays have 4 random tiles (TODO: respect total number of tiles)
        self.gameBoardDisplays=np.zeros((5,5),dtype=np.int)
        for i in range(5):
            for j in range (4):
                self.gameBoardDisplays[i,random.randrange(0,4,1)] += 1

if __name__ == "__main__":
    game=azul()
    game.new_round()
    print(str(game.gameBoardDisplays))

# TESTS Move these later

def test_azulInit():
    game=azul()
    game4Players=azul(players=4)
    # gameBoardDisplays should be of size (5,5).
    # TODO: Size should change depending on nr of players
    assert game.gameBoardDisplays.shape == (5,5)
    # gameBoardCenter should be of size 6
    assert game.gameBoardCenter.shape == (6,)
    # patternLines should be of size (nrOfPlayers,5,5)
    assert game.patternLines.shape == (2,5,5)
    assert np.count_nonzero(game4Players.patternLines) == 0
    assert game4Players.patternLines.shape == (4,5,5)
    # walls should be of size (nrOfPlayers,5,5) and be empty
    assert game.walls.shape == (2,5,5)
    assert np.count_nonzero(game4Players.walls) == 0
    assert game4Players.walls.shape == (4,5,5)
    # floor should be of size (nrOfPlayers,7)
    assert game.floors.shape == (2,7)
    assert game4Players.floors.shape == (4,7)
    # score should be of size (nrOfPlayers,) and be zero
    assert game.score.shape == (2,)
    assert np.count_nonzero(game4Players.score) == 0
    assert game4Players.score.shape == (4,)

def test_azul_new_round():
    game=azul()
    game.new_round()
    #After new round, no display should have more than 4 tiles
    for tile in game.gameBoardDisplays:
        assert np.sum(tile) == 4
    #After new round, game board center should have zero color tiles
    assert np.count_nonzero(game.gameBoardCenter[:5]) == 0
    #After new round, game board center should have one white tile
    assert game.gameBoardCenter[5] == 1