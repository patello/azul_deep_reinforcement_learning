import numpy as np
import sys

class azul:
    def __init__(self,players=2):
        self.gameBoard = np.zeros((5,5),dtype=np.int)
        workshopPrototype = np.array(list(map(lambda x: np.full(5,x),range(1,6))),dtype=np.int)
        self.playerWorkshops=np.repeat(workshopPrototype[np.newaxis,:, :], players, axis=0)
        tilesPrototype = np.zeros((5,5),dtype=np.bool)
        self.playerTiles=np.repeat(tilesPrototype[np.newaxis,:, :], players, axis=0)
    def start(self):
        print("started")

if __name__ == "__main__":
    game=azul()
    print(str(game.playerWorkshops[0]))