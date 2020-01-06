import numpy as np
import random
import sys
import json

random.seed()

class IllegalMove(Exception):
    pass

class GameEnded(Exception):
    pass

class Azul:
    def __init__(self,players=2,state_file=None):
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
        self.end_of_game=False
        self.turn_counter=0
        self.first_player_stats = np.zeros(players)
        self.floor_penalty = np.zeros(players)
        self.max_combo = np.zeros(players)
        #0 - Row, 1 - Color, 2 - Column
        self.completed_lines = np.zeros((players,3))
        if state_file is not None:
            self.import_JSON(state_file)
    def __eq__(self, other):
        return np.array_equal(self.game_board_displays,other.game_board_displays) and np.array_equal(self.game_board_center,other.game_board_center) and np.array_equal(self.pattern_lines,other.pattern_lines) and np.array_equal(self.walls,other.walls) and np.array_equal(self.floors, other.floors) and np.array_equal(self.score,other.score) and self.current_player==other.current_player and self.next_first_player==other.next_first_player and self.players==other.players and self.end_of_game == other.end_of_game and self.turn_counter == other.turn_counter
    def new_round(self):
        #Set current player and reset next player and increment the turn counter
        self.current_player=self.next_first_player
        self.first_player_stats[self.next_first_player-1]+=1
        self.turn_counter += 1
        self.next_first_player=0
        #Game board center starts empty except for the first player token (sixth index).
        self.game_board_center=np.array([0,0,0,0,0,1],dtype=np.int)
        #Factory displays have 4 random tiles (TODO: respect total number of tiles)
        self.game_board_displays=np.zeros((5,5),dtype=np.int)
        for i in range(5):
            for j in range (4):
                self.game_board_displays[i,random.randrange(0,5,1)] += 1
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
            self.turn_counter = data["turn_counter"]
    def export_JSON(self,path):
        with open(path,"w+") as json_file:
            data = json.dumps({
                "game_board_displays": self.game_board_displays.tolist(),
                "game_board_center": self.game_board_center.tolist(),
                "pattern_lines": self.pattern_lines.tolist(),
                "walls": self.walls.tolist(),
                "floors": self.floors.tolist(),
                "score": self.score.tolist(),
                "current_player": self.current_player,
                "next_first_player": self.next_first_player,
                "players": self.players,
                "turn_counter": self.turn_counter
                })
            json_file.write(data)
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
        if pattern != 0:
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
    def is_end_of_round(self):
        return np.count_nonzero(self.game_board_displays) + np.count_nonzero(self.game_board_center) < 1
    def is_end_of_game(self):
        for player in range(self.players):
            #Iterate through all rows for all players
            for i in range(5):
                #If at least one row is full, then the game is ended
                if np.count_nonzero(self.walls[player,i])==5:
                    return True
        return False
    def count_score(self):
        #Current implementation of count score resets the pattern lines and floor and set the wall.
        def to_wall_position(color,pattern):
            #Function used to translate where a certain color ends up on the wall
            return (color+pattern)%5
        def from_wall_position(color,pattern):
            #Function to translate which color a certain position on the wall is
            return (color-pattern)%5
        def count_floor(player):
            #The if statements below capture the behaviour of the three penalty levels.
            if self.floors[player] <= 2:
                count = -self.floors[player]
            elif self.floors[player] <= 5:
                count = -2 - (self.floors[player]-2)*2
            else:
                count = -8 - (self.floors[player]-5)*3
            self.floor_penalty[player] += count
            self.floors[player] = 0
            return count
        def count_wall(player):
            count = 0
            for pattern in range(5):
                for color in range(5):
                    #Check if the pattern line for the particular color is full (equal to the pattern index plus 1)
                    if self.pattern_lines[player,pattern,color] == pattern+1:
                        #In that case, set it to zero and add it to the wall.
                        self.pattern_lines[player,pattern,color]=0
                        self.walls[player,pattern,color] = True
                        #pos_count will be used to tally the score when checking the neighboaring tiles
                        pos_count = 0
                        bonus_count = 0
                        only_row=True
                        only_col=True
                        #Count forwards along the pattern line. These should probably be functionized
                        #Don't count the original tile, but keep track if no other tiles are counted.
                        for i in range(to_wall_position(color,pattern)+1,5):
                            if self.walls[player,pattern,from_wall_position(i,pattern)]:
                                pos_count += 1
                                only_row = False
                            else:
                                #If the position does not match, stop counting along this line
                                break
                        for i in range(to_wall_position(color,pattern)-1,-1,-1):
                            if self.walls[player,pattern,from_wall_position(i,pattern)]:
                                pos_count += 1
                                only_row = False
                            else:
                                break
                        #Don't count the original tile, but keep track if no other tiles are counted.
                        for j in range(pattern+1,5):
                            if self.walls[player,j,to_wall_position(color,pattern-j)]:
                                pos_count += 1
                                only_col = False
                            else:
                                #If the position does not match, stop counting along this line
                                break
                        for j in range(pattern-1,-1,-1):
                            if self.walls[player,j,to_wall_position(color,pattern-j)]:
                                pos_count += 1
                                only_col = False
                            else:
                                #If the position does not match, stop counting along this line
                                break
                        if only_row and only_col:
                            pos_count=1
                        elif not (only_row or only_col):
                            pos_count+=2
                        else:
                            pos_count+=1
                        self.max_combo[player]=max(self.max_combo[player],pos_count)
                        #Check if the row is complete by iterating through it
                        for i in range(5):
                            if self.walls[player,pattern,i]:
                                if i == 4:
                                    bonus_count+=2
                                    self.completed_lines[player,0]+=1
                            else:
                                break
                        #Check if the column (color) is complete by iterating through it
                        for j in range(5):
                            if self.walls[player,j,color]:
                                if j == 4:
                                    bonus_count+=10
                                    self.completed_lines[player,1]+=1
                            else:
                                break
                        #Check if the vertical column (actual column) is complete by iterating through it
                        for k in range(5):
                            if self.walls[player,k,from_wall_position(to_wall_position(color,pattern),k)]:
                                if k == 4:
                                    bonus_count+=7
                                    self.completed_lines[player,2]+=1
                            else:
                                break
                        count += pos_count + bonus_count
            return count
        for player in range(self.players):
            self.score[player]+=count_floor(player)+count_wall(player)
            #You should never be able to go below 0 points
            if self.score[player] < 0:
                self.score[player] = 0
    def step(self, display, color, pattern):
        #You are not allowed to make a move on a game that has ended
        if self.end_of_game:
            raise GameEnded
        #If the move is illegal, raise and exception
        if not self.is_legal_move(display, color, pattern):
            raise IllegalMove
        #If no exceptions, make the move
        self.move(display, color, pattern)
        #Score if the round has ended. Then check if the game has ended
        if self.is_end_of_round():
            self.count_score()
            if self.is_end_of_game():
                self.end_of_game=True
            else:
                self.new_round()
        else:
            self.next_player()
    def get_statistics(self):
        return {"player_score":self.score[0],"opponent_score":self.score[1],"rounds":self.turn_counter,"percent_first_player":self.first_player_stats[0]/self.first_player_stats.sum()*100,"floor_penalty":-self.floor_penalty[0],"max_combo":self.max_combo[0],"completed_rows":self.completed_lines[0,0],"completed_columns":self.completed_lines[0,2],"completed_colors":self.completed_lines[0,1],"win_percent":self.score[0]>self.score[1]}

if __name__ == "__main__":
    game=Azul()
    game.import_JSON("./tests/resources/game_end_of_round_1.json")
    game.count_score()
    

