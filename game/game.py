import numpy as np
import random
import sys
import json

random.seed()

class Azul:
    def __init__(self,players=2):
        self.game_board_displays = np.zeros((5,5),dtype=np.int)
        self.game_board_center = np.zeros(6,dtype=np.int)
        pattern_lines_prototype = np.zeros((5,5),dtype=np.int)
        self.pattern_lines=np.repeat(pattern_lines_prototype[np.newaxis,:, :], players, axis=0)
        walls_prototype = np.zeros((5,5),dtype=np.bool)
        self.walls=np.repeat(walls_prototype[np.newaxis,:, :], players, axis=0)
        self.floors=np.zeros(players,dtype=np.int)
        self.score=np.zeros(players,dtype=np.int)
        self.current_player=0
        self.next_first_player=1
        self.players=players
    def __eq__(self, other):
        return np.array_equal(self.game_board_displays,other.game_board_displays) and np.array_equal(self.game_board_center,other.game_board_center) and np.array_equal(self.pattern_lines,other.pattern_lines) and np.array_equal(self.walls,other.walls) and np.array_equal(self.floors, other.floors) and np.array_equal(self.score,other.score) and self.current_player==other.current_player and self.next_first_player==other.next_first_player
    def new_round(self):
        #Set current player and reset next player
        self.current_player=self.next_first_player
        self.next_first_player=0
        #Game board center starts empty except for the first player token (sixth index).
        self.game_board_center=np.array([0,0,0,0,0,1],dtype=np.int)
        #Factory displays have 4 random tiles (TODO: respect total number of tiles)
        self.game_board_displays=np.zeros((5,5),dtype=np.int)
        for i in range(5):
            for j in range (4):
                self.game_board_displays[i,random.randrange(0,4,1)] += 1
    def import_JSON(self,path):
        with open(path) as json_file:
            data = json.load(json_file)
            self.game_board_displays = np.array(data["game_board_displays"],dtype=np.int)
            self.game_board_center = np.array(data["game_board_center"],dtype=np.int)
            self.pattern_lines = np.array(data["pattern_lines"],dtype=np.int)
            self.walls = np.array(data["walls"],dtype=np.bool)
            self.floors = np.array(data["floors"],dtype=np.int)
            self.score = np.array(data["score"],dtype=np.int)
            self.current_player = data["current_player"]
            self.next_first_player = data["next_first_player"]
            self.players = data["players"]
    def move(self, display, color, pattern):
        def add_to_floor(nr_tiles):
            if self.floors[self.current_player-1]+nr_tiles<7:
                self.floors[self.current_player-1]+=nr_tiles
            else:
                self.floors[self.current_player-1]=7
        #Display 0 indicates center display. 1..5 (for 2 players) are the displays
        if display != 0:
            #Save number of tiles of the given color before reseting
            nr_tiles=self.game_board_displays[display-1,color]
            #Remove the tiles of the corresponding tile from the display
            self.game_board_displays[display-1,color]=0
            #Add the remaining tiles to the center
            self.game_board_center[:5]+=self.game_board_displays[display-1]
            #Remove the remaining tiles from the display
            self.game_board_displays[display-1]=np.zeros(5)
        else:
            #Save number of tiles of the given color before removing
            nr_tiles=self.game_board_center[color]
            #Remove the tiles of the corresponding tile from the center
            self.game_board_center[color]=0
            #If first player tile is in the center, remove it and set next player to current player
            if self.game_board_center[5]==1:
                self.game_board_center[5]=0
                self.next_first_player=self.current_player
                add_to_floor(1)
        #If pattern is 1..5 then it corresponds to a pattern line. Otherwise it corresponds to the floor
        if pattern != 0:
            #Check how many spots are remaining on the pattern. It can hold its index (+1 because of the floor on index 0)
            tile_overflow = pattern-self.pattern_lines[self.current_player-1,pattern-1,color]-nr_tiles
            #If tiles are zero or positive, then the tiles can be added. Otherwise set it to the maximum and put the rest on the floor
            if tile_overflow >= 0:
                self.pattern_lines[self.current_player-1,pattern-1,color]+=nr_tiles
            else:
                self.pattern_lines[self.current_player-1,pattern-1,color]=pattern
                #Minus tile_overflow is the tiles that didn't fit in the pattern line
                add_to_floor(-tile_overflow)
        else:
            add_to_floor(nr_tiles)
    def is_legal_move(self, display, color, pattern):
        #Check if displays or center should be checked, then check if there exists tiles of the corresponding color in the display. Else return false
        if display > 0:
            if self.game_board_displays[display-1,color]<1:
                return False
        else:
            if self.game_board_center[color]<1:
                return False
        #Check if we are adding to a pattern line, otherwise testning pattern line and wall is not relevant
        if pattern > 0:
            if np.count_nonzero(self.pattern_lines[self.current_player-1,pattern-1,:color])+np.count_nonzero(self.pattern_lines[self.current_player-1,pattern-1,color+1:5])> 0:
                return False
            if self.walls[self.current_player-1,pattern-1,color]:
                return False
        return True
    def next_player(self):
        if self.current_player<self.players:
            self.current_player+=1
        else:
            self.current_player=1

if __name__ == "__main__":
    game=Azul()
    test_azul_is_legal_move()

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
        assert Azul(players=i).floors.shape == (i,)
    # score should be of size (nr of players,) and be zero
    for i in range(2,5):
        assert Azul(players=i).score.shape == (i,)
        assert np.count_nonzero(Azul(players=i).score) == 0

def test_azul_new_round():
    game=Azul()
    prev_next_first_player=game.next_first_player
    game.new_round()
    #After new round, no display should have more than 4 tiles
    for tile in game.game_board_displays:
        assert np.sum(tile) == 4
    #After new round, game board center should have zero color tiles
    assert np.count_nonzero(game.game_board_center[:5]) == 0
    #After new round, game board center should have one white tile
    assert game.game_board_center[5] == 1
    #Current player should be set to the next_first_player from last round
    assert game.current_player == prev_next_first_player
    #Next first player should be reset to 0
    game.next_first_player = 1
    game.new_round()
    assert game.next_first_player == 0

def test_azul_eq():
    game1=Azul()
    game2=Azul()
    #Test that two initialized boards are equal
    assert game1==game2
    #Test that two random boards are not equal 
    game1.new_round()
    game2.new_round()
    assert game1!=game2

def test_azul_import():
    #Test that importing the empty game reference is equal to a new game.
    game=Azul()
    imported_game=Azul()
    imported_game.import_JSON("./tests/resources/game_empty.json")
    assert game == imported_game
    #Test that importing first round reference is equal to the first game of a round with seed 1
    random.seed(1)
    game=Azul()
    game.new_round()
    imported_game=Azul()
    imported_game.import_JSON("./tests/resources/game_first_round.json")
    assert game == imported_game
    random.seed()

def test_azul_move():
    game = Azul()
    #When moving from a display with other colors, some end up in the center. The display shall be empty
    game.import_JSON("./tests/resources/game_first_round.json")
    game.move(5,0,2)
    assert np.array_equal(game.game_board_displays[4],np.zeros(5,))
    assert np.array_equal(game.game_board_center,np.array([0,1,2,0,0,1]))
    assert np.array_equal(game.pattern_lines[game.current_player-1,1],np.array([1,0,0,0,0]))
    #When moving from a display with only one color, nothing shall end up in the center
    game.import_JSON("./tests/resources/game_first_round.json")
    game.move(2,3,4)
    assert np.array_equal(game.game_board_displays[1],np.zeros(5,))
    assert np.array_equal(game.game_board_center,np.array([0,0,0,0,0,1]))
    assert np.array_equal(game.pattern_lines[game.current_player-1,3],np.array([0,0,0,4,0]))
    #When moving to a wall with not enough room, it will fill up and the rest will end up on the floor
    game.import_JSON("./tests/resources/game_first_round.json")
    game.move(2,3,2)
    assert np.array_equal(game.pattern_lines[game.current_player-1,1],np.array([0,0,0,2,0]))
    assert game.floors[game.current_player-1]==2
    #When taking from the center, also take the first player token, which takes one space on the floor
    game.import_JSON("./tests/resources/game_first_round.json")
    game.move(1,0,2)
    game.move(0,1,1)
    assert np.array_equal(game.game_board_center,np.array([0,0,1,0,0,0],dtype=int))
    assert np.array_equal(game.pattern_lines[game.current_player-1,0],np.array([0,1,0,0,0],dtype=int))
    assert game.next_first_player==game.current_player
    assert game.floors[game.current_player-1]==1
    #Tiles in the center stack. Also check that overfilling works when filling a non empty pattern
    game.import_JSON("./tests/resources/game_first_round.json")
    game.move(1,0,3)
    game.move(3,0,3)
    assert np.array_equal(game.game_board_center,np.array([0,2,1,1,0,1],dtype=int))
    assert np.array_equal(game.pattern_lines[game.current_player-1,2],np.array([3,0,0,0,0],dtype=int))
    assert game.floors[game.current_player-1]==1
    #Giving a pattern value of one makes the tiles go directly to the floor and adding to the floor stacks
    game.import_JSON("./tests/resources/game_first_round.json")
    game.move(3,0,0)
    assert game.floors[game.current_player-1]==2
    game.move(4,0,0)
    assert game.floors[game.current_player-1]==3
    game.move(1,0,0)
    game.move(2,3,1)
    assert game.floors[game.current_player-1]==7

def test_azul_is_legal_move():
    game=Azul()
    game.import_JSON("./tests/resources/game_first_round.json")
    #Taking a color that exists from the displays, and putting it in an empty position is allowed
    assert game.is_legal_move(5,0,2)
    #Taking a color that does not exist from the displays is not allowed
    assert not game.is_legal_move(1,4,2)
    #Taking a color that exists from the center, and putting it in an empty position is allowed
    game.move(5,0,2)
    assert game.is_legal_move(0,1,1)
    #Taking a color that does not exist from the center is not allowed, even if first player token is there
    assert not game.is_legal_move(0,0,0)
    game.import_JSON("./tests/resources/game_first_round.json")
    assert not game.is_legal_move(0,0,0)
    #Filling a color that already exist in the pattern line is allowed, but only if it is the same color as the one already in the line
    game.import_JSON("./tests/resources/game_sample_1.json")
    assert game.is_legal_move(0,0,5)
    assert not game.is_legal_move(0,1,5)
    #You are allowed to put a tile on the pattern line only if that tile does not already exist on the wall
    assert game.is_legal_move(5,0,3)
    assert not game.is_legal_move(5,2,3)

def test_azul_next_player():
    #In a two player game, the order of the first round goes 1,2,1
    game=Azul()
    game.new_round()
    assert game.current_player==1
    game.next_player()
    assert game.current_player==2
    game.next_player()
    assert game.current_player==1
    #In a four player game, the order of the first round goes 1,2,1
    game=Azul(players=4)
    game.new_round()
    assert game.current_player==1
    game.next_player()
    assert game.current_player==2
    game.next_player()
    assert game.current_player==3
    game.next_player()
    assert game.current_player==4
    game.next_player()
    assert game.current_player==1

