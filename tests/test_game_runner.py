import os
import numpy as np
import torch
import random

from azulnet import GameRunner, Agent, RandomAgent, Azul
from azulnet import check_all_valid, nn_serialize, nn_deserialize

#Script dir as per suggestion here: https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
script_dir = os.path.dirname(__file__)

def test_init():
    gamerunner = GameRunner()
    #Move counter should be set to 0
    assert gamerunner.move_counter == 0
    #Player should start with 0 score
    assert gamerunner.player_score == 0
    #Game board should be set
    game_board = gamerunner.game.game_board_displays
    for i in range(5):
        assert np.sum(game_board[i])==4
    game_center = gamerunner.game.game_board_center
    assert np.array_equal(game_center,np.array([0,0,0,0,0,1]))

def test_step():
    #Use seed 1 to get a state equal to the test resource "first round"
    random.seed(1)
    gamerunner = GameRunner()
    reward, end_of_game = gamerunner.step(nn_serialize(1,0,2))
    #The turn should have gone back to the original player
    assert gamerunner.game.current_player == 1
    #For this seed, game should not have ended after one step
    assert not end_of_game
    #Score on the original board should not have been counted
    assert np.array_equal(gamerunner.game.score,np.zeros(2))
    #Load game_end_of_round_2 to see that game_board gets reset and player gets back to one
    gamerunner = GameRunner()
    gamerunner.game.import_JSON(script_dir+"/resources/game_end_of_round_2.json")
    reward, end_of_game = gamerunner.step(nn_serialize(0,4,1))
    assert gamerunner.game.current_player == 1
    #For this scenario, game should have ended after one step
    assert not end_of_game
    #Load game_end_of_round_2 to see that the game ends if the right steps are made
    gamerunner = GameRunner()
    gamerunner.game.import_JSON(script_dir+"/resources/game_end_of_round_2.json")
    gamerunner.player_score=49-32
    random.seed(1)
    reward, end_of_game = gamerunner.step(nn_serialize(0,0,1))
    assert end_of_game
    #Check that player score is same as score difference
    assert gamerunner.player_score==gamerunner.game.score[0]-gamerunner.game.score[1]
    #Load game_end_of_round_3 to see that game_board gets reset
    gamerunner = GameRunner()
    gamerunner.game.import_JSON(script_dir+"/resources/game_end_of_round_3.json")
    gamerunner.player_score=49-32
    reward, end_of_game = gamerunner.step(nn_serialize(0,3,0))
    #For this scenario, game should not have ended after one step
    assert not end_of_game
    #With this scenario, reward should be -6
    assert reward == -6
    #Since board is reset, the player_score should be equal to the game score difference
    assert gamerunner.player_score==gamerunner.game.score[0]-gamerunner.game.score[1]
    #Game board should be set
    game_board = gamerunner.game.game_board_displays
    for i in range(5):
        assert np.sum(game_board[i])==4
    game_center = gamerunner.game.game_board_center
    assert np.array_equal(game_center,np.array([0,0,0,0,0,1]))
    assert gamerunner.game.current_player == 1

def test_get_state():
    gamerunner = GameRunner()
    state_flat = gamerunner.get_state()
    assert np.sum(state_flat) == 4*5 + 1
    assert np.size(state_flat) == 5*5 + 6 + 5*5*2 + 5*5*2 + 2 + 2 + 1

def test_nn_tools_serialize():
    #Iterate through all serializable tuples and test that you get the same thing back
    for i in range(6):
        for j in range(5):
            for k in range(6):
                assert(i,j,k)==nn_deserialize(nn_serialize(i,j,k))

def test_nn_tools_deserialize():
    #Iterate through all deserializable integers and test that you get the same thing back
    for i in range(6*5*6):
        assert i == nn_serialize(*nn_deserialize(i))

def test_check_all_valid():
    game=Azul()
    #On a new board, none of the colors should be available or displays, pattern availability is undefined (?)
    assert np.array_equal(check_all_valid(game),np.zeros(180,dtype="bool"))
    #Sample all the actions that should be available on the resource called "game_first_round"
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    all_valid = check_all_valid(game)
    for j in [0,1,2]:
        for k in range(6):
            assert all_valid[nn_serialize(1,j,k)]
    for j in [3]:
        for k in range(6):
            assert all_valid[nn_serialize(2,j,k)]
    for j in [0,1,3]:
        for k in range(6):
            assert all_valid[nn_serialize(3,j,k)]
    for j in [0,3]:
        for k in range(6):
            assert all_valid[nn_serialize(4,j,k)]
    for j in [0,1,2]:
        for k in range(6):
            assert all_valid[nn_serialize(5,j,k)]
    #Check that center is valid when there are other colors there
    game.import_JSON(script_dir+"/resources/game_sample_1.json")
    all_valid = check_all_valid(game)
    assert all_valid[nn_serialize(0,0,4)]    
