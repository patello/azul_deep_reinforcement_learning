from game.nn_runner import *
import numpy as np

def test_nn_runner_init():
    #NNRunner should start a 2 player game
    assert NNRunner().game.players == 2
    #Move counter should be set to 0
    assert NNRunner().move_counter == 0
    #Player should start with 0 score
    assert NNRunner().player_score == 0

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
    game.import_JSON("/usr/tests/resources/game_first_round.json")
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
    game.import_JSON("/usr/tests/resources/game_sample_1.json")
    all_valid = check_all_valid(game)
    assert all_valid[nn_serialize(0,0,4)]
