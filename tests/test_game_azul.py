import pytest
import os
import numpy as np
from game.azul import *

#Script dir as per suggestion here: https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
script_dir = os.path.dirname(__file__)

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
    # Turn counter should be set to 0
    assert Azul().turn_counter == 0
    # Test the functionality of creating a game from a json file
    random.seed(1)
    game = Azul()
    game.new_round()
    assert game == Azul(state_file=script_dir+"/resources/game_first_round_seed_1.json")
    #Test that rule determining the first player with an integer
    game = Azul(rules={"first_player":1})
    game.new_round()
    assert game.current_player==1
    game = Azul(rules={"first_player":2})
    game.new_round()
    assert game.current_player==2
    with pytest.raises(IllegalRule):
        Azul(rules={"first_player":3})
    #Test that rule determining the first player with a random choice
    first_players = np.array([])
    for i in range(100):
        game = Azul(rules={"first_player":"Random"})
        game.new_round()
        first_players=np.append(first_players,game.current_player)
    assert np.count_nonzero(first_players==1) > 0
    assert np.count_nonzero(first_players==2) > 0
    assert np.count_nonzero(first_players==1)+np.count_nonzero(first_players==2) == 100



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
    #Check that turn timer correctly increments its value with a new round
    game=Azul()
    prev_turn_counter=game.turn_counter
    game.new_round()
    assert game.turn_counter == prev_turn_counter + 1

def test_azul_eq():
    game1=Azul()
    game2=Azul()
    #Test that two initialized boards are equal
    assert game1==game2
    #Test that two random boards are not equal 
    game1.new_round()
    game2.new_round()
    assert game1!=game2

def test_azul_import_JSON():
    #Test that importing the empty game reference is equal to a new game.
    game=Azul()
    imported_game=Azul()
    imported_game.import_JSON(script_dir+"/resources/game_empty.json")
    assert game == imported_game
    #Test that importing first round reference is equal to the first game of a round with seed 1
    random.seed(1)
    game=Azul()
    game.new_round()
    imported_game=Azul()
    imported_game.import_JSON(script_dir+"/resources/game_first_round_seed_1.json")
    assert game == imported_game
    random.seed()

def test_azul_export_JSON(tmp_path):
    #Test a new game with new round
    game=Azul()
    game.new_round()
    game.export_JSON(tmp_path / "test_export.json")
    imported_game=Azul()
    imported_game.import_JSON(tmp_path / "test_export.json")
    assert game == imported_game
    #Test a random game, so that all fields are tested
    game.import_JSON(script_dir+"/resources/game_sample_1.json")
    game.export_JSON(tmp_path / "test_export.json")
    imported_game.import_JSON(tmp_path / "test_export.json")
    assert game == imported_game

def test_azul_move():
    game = Azul()
    #When moving from a display with other colors, some end up in the center. The display shall be empty
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    game.move(5,0,2)
    assert np.array_equal(game.game_board_displays[4],np.zeros(5,))
    assert np.array_equal(game.game_board_center,np.array([0,1,2,0,0,1]))
    assert np.array_equal(game.pattern_lines[game.current_player-1,1],np.array([1,0,0,0,0]))
    #When moving from a display with only one color, nothing shall end up in the center
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    game.move(2,3,4)
    assert np.array_equal(game.game_board_displays[1],np.zeros(5,))
    assert np.array_equal(game.game_board_center,np.array([0,0,0,0,0,1]))
    assert np.array_equal(game.pattern_lines[game.current_player-1,3],np.array([0,0,0,4,0]))
    #When moving to a wall with not enough room, it will fill up and the rest will end up on the floor
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    game.move(2,3,2)
    assert np.array_equal(game.pattern_lines[game.current_player-1,1],np.array([0,0,0,2,0]))
    assert game.floors[game.current_player-1]==2
    #When taking from the center, also take the first player token, which takes one space on the floor
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    game.move(1,0,2)
    game.move(0,1,1)
    assert np.array_equal(game.game_board_center,np.array([0,0,1,0,0,0],dtype=int))
    assert np.array_equal(game.pattern_lines[game.current_player-1,0],np.array([0,1,0,0,0],dtype=int))
    assert game.next_first_player==game.current_player
    assert game.floors[game.current_player-1]==1
    #Tiles in the center stack. Also check that overfilling works when filling a non empty pattern
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    game.move(1,0,3)
    game.move(3,0,3)
    assert np.array_equal(game.game_board_center,np.array([0,2,1,1,0,1],dtype=int))
    assert np.array_equal(game.pattern_lines[game.current_player-1,2],np.array([3,0,0,0,0],dtype=int))
    assert game.floors[game.current_player-1]==1
    #Giving a pattern value of one makes the tiles go directly to the floor and adding to the floor stacks
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    game.move(3,0,0)
    assert game.floors[game.current_player-1]==2
    game.move(4,0,0)
    assert game.floors[game.current_player-1]==3
    game.move(1,0,0)
    game.move(2,3,1)
    assert game.floors[game.current_player-1]==7

def test_azul_is_legal_move():
    #TODO: Should test that the move is not out of bounds for the current playing field
    game=Azul()
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    #Taking a color that exists from the displays, and putting it in an empty position is allowed
    assert game.is_legal_move(5,0,2)
    #Taking a color that does not exist from the displays is not allowed
    assert not game.is_legal_move(1,4,2)
    #Taking a color that exists from the center, and putting it in an empty position is allowed
    game.move(5,0,2)
    assert game.is_legal_move(0,1,1)
    #Taking a color that does not exist from the center is not allowed, even if first player token is there
    assert not game.is_legal_move(0,0,0)
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    assert not game.is_legal_move(0,0,0)
    #Filling a color that already exist in the pattern line is allowed, but only if it is the same color as the one already in the line
    game.import_JSON(script_dir+"/resources/game_sample_1.json")
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

def test_azul_is_end_of_round():
    game=Azul()
    #Check that a game in the middle of the round does not trigger end of game
    game.import_JSON(script_dir+"/resources/game_sample_1.json")
    assert not game.is_end_of_round()
    game.move(0,0,5)
    assert not game.is_end_of_round()
    #Check that after the final move is made, end of round is true
    game.import_JSON(script_dir+"/resources/game_end_of_round_1.json")
    assert not game.is_end_of_round()
    game.move(0,3,3)
    assert game.is_end_of_round()

def test_azul_is_end_of_game():
    game = Azul()
    #Test moves that should not result in end of game
    game.import_JSON(script_dir+"/resources/game_end_of_round_2.json")
    assert not game.is_end_of_game()
    game.move(0,4,1)
    game.next_player()
    game.move(0,0,3)
    game.count_score()
    assert not game.is_end_of_game()
    #Test moves that should result in end of game after score has been counted
    game.import_JSON(script_dir+"/resources/game_end_of_round_2.json")
    game.move(0,0,1)
    game.next_player()
    game.move(0,4,1)
    game.count_score()
    assert game.is_end_of_game()

def test_azul_count_score():
    game=Azul()
    #Check that the players get the correct score when no bonuses are applied and that the pattern lines and well end up in the right state
    game.import_JSON(script_dir+"/resources/game_end_of_round_1.json")
    prev_score=np.copy(game.score)
    game.count_score()
    assert np.array_equal(game.score,prev_score+np.array([5 + 5 + 1 - 2, 4 + 2 + 3 - 8]))
    #Check that the floor has been emptied
    assert np.array_equal(game.floors,np.zeros(2))
    #Check that full pattern lines have been emptied
    assert np.array_equal(game.pattern_lines[0],np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,2,0,0],[0,0,0,0,0],[0,0,3,0,0]]))
    assert np.array_equal(game.pattern_lines[1],np.array([[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[2,0,0,0,0],[0,0,0,0,0]]))
    #Check that walls have gotten their corresponding tiles
    assert np.array_equal(game.walls[0],np.array([[1,1,1,0,0],[1,1,1,0,0],[0,0,0,0,0],[0,1,0,0,1],[0,0,0,0,0]]))
    assert np.array_equal(game.walls[1],np.array([[1,1,1,1,0],[0,0,0,0,1],[0,0,1,0,0],[0,1,0,0,0],[1,1,0,0,0]]))
    #Check that the algorithm works when a wall tile is added in the same round
    game.import_JSON(script_dir+"/resources/game_end_of_round_1.json")
    prev_score=np.copy(game.score)
    game.move(0,3,3)
    game.count_score()
    assert np.array_equal(game.score,prev_score+np.array([5 + 5 + 1 - 2, 4 + 2 + 3 + 3 - 8]))
    #Check that algorithm can gives score for color and column combos
    game.import_JSON(script_dir+"/resources/game_end_of_round_2.json")
    prev_score=np.copy(game.score)
    game.move(0,4,1)
    game.next_player()
    game.move(0,0,3)
    game.count_score()
    assert np.array_equal(game.score,prev_score+np.array([5 + 7 + 10, 5 + 7]))
    #Check that algorithm can gives score for rows
    game.import_JSON(script_dir+"/resources/game_end_of_round_2.json")
    prev_score=np.copy(game.score)
    game.move(0,0,1)
    game.next_player()
    game.move(0,4,1)
    game.count_score()
    assert np.array_equal(game.score,prev_score+np.array([2 - 2, 5 + 2]))
    #Check that you can't go below 0 points
    game.import_JSON(script_dir+"/resources/game_end_of_round_2.json")
    game.move(0,0,0)
    game.next_player()
    game.move(0,4,0)
    game.count_score()
    assert np.array_equal(game.score,np.array([0, 0]))

def test_azul_test_step():
    game=Azul()
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    #Check that a step executes the move and passes to the next player
    game.step(5,0,2)
    assert np.array_equal(game.game_board_displays[4],np.zeros(5,))
    assert np.array_equal(game.game_board_center,np.array([0,1,2,0,0,1]))
    assert np.array_equal(game.pattern_lines[0,1],np.array([1,0,0,0,0]))
    assert game.current_player==2
    #Making an illegal move should raise an exception and the state should not change 
    import copy
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    prev_game_state = copy.copy(game)
    with pytest.raises(IllegalMove):
        game.step(1,4,2)
    assert game==prev_game_state
    #Making the last move should trigger a score count, make the player with the first player token the current player and reset the board
    game.import_JSON(script_dir+"/resources/game_end_of_round_1.json")
    prev_score=np.copy(game.score)
    next_first_player=game.next_first_player
    game.step(0,3,3)
    #After new round, no display should have more than 4 tiles
    for tile in game.game_board_displays:
        assert np.sum(tile) == 4
    #After new round, game board center should have zero color tiles
    assert np.count_nonzero(game.game_board_center[:5]) == 0
    #After new round, game board center should have one white tile
    assert game.game_board_center[5] == 1
    #Make sure the score has been counted correctly
    assert np.array_equal(game.score,prev_score+np.array([5 + 5 + 1 - 2, 4 + 2 + 3 + 3 - 8]))
    #Make sure the next player token is passed
    assert game.current_player==next_first_player
    assert game.next_first_player == 0
    #Completing a row should trigger end of game after the round is over
    game.import_JSON(script_dir+"/resources/game_end_of_round_2.json")
    prev_score=np.copy(game.score)
    game.step(0,0,1)
    assert not game.end_of_game
    game.step(0,4,1)
    assert np.array_equal(game.score,prev_score+np.array([2 - 2, 5 + 2]))
    assert game.end_of_game
    #Trying to make a step after the game has ended should raise an exception
    with pytest.raises(GameEnded):
        game.step(0,0,0)