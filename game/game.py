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
    def newRound(self):
        #Game board center starts empty except for the first player token (sixth index).
        self.gameBoardCenter=[0,0,0,0,0,1]
        #Factory displays have 4 random tiles (TODO: respect total number of tiles)
        self.gameBoardDisplays=np.zeros((5,5),dtype=np.int)
        for i in range(5):
            for j in range (4):
                self.gameBoardDisplays[i,random.randrange(0,4,1)] += 1

if __name__ == "__main__":
    game=azul()
    game.newRound()
    print(str(game.gameBoardDisplays))