from game.nn_runner import *
from neural.agent import Agent
import numpy as np

def test_nnrunner_init():
    agent=Agent()
    nnrunner = NNRunner(agent)
    #NNRunner should start a 2 player game
    assert nnrunner.game.players == 2
    #Move counter should be set to 0
    assert nnrunner.move_counter == 0
    #Player should start with 0 score
    assert nnrunner.player_score == 0
    #Game board should be set
    game_board = nnrunner.game.game_board_displays
    for i in range(5):
        assert np.sum(game_board[i])==4
    game_center = nnrunner.game.game_board_center
    assert np.array_equal(game_center,np.array([0,0,0,0,0,1]))

def test_nnrunner_step():
    #Use seed 1 to get a state equal to the test resource "first round"
    random.seed(1)
    agent=Agent()
    nnrunner = NNRunner(agent)
    reward, end_of_game = nnrunner.step(nn_serialize(1,0,2))
    #The turn should have gone back to the original player
    assert nnrunner.game.current_player == 1
    #For this seed, game should not have ended after one step
    assert not end_of_game
    #With this seed, reward should be 0
    assert reward == 0
    #Score on the original board should not have been counted
    assert np.array_equal(nnrunner.game.score,np.zeros(2))
    #Load game_end_of_round_2 to see that game_board gets reset and player gets back to one
    agent=Agent()
    nnrunner = NNRunner(agent)
    nnrunner.game.import_JSON("/tests/resources/game_end_of_round_2.json")
    reward, end_of_game = nnrunner.step(nn_serialize(0,4,1))
    assert nnrunner.game.current_player == 1
    #For this scenario, game should have ended after one step
    assert not end_of_game
    #Load game_end_of_round_2 to see that the game ends if the right steps are made
    agent=Agent()
    nnrunner = NNRunner(agent)
    nnrunner.game.import_JSON("/tests/resources/game_end_of_round_2.json")
    nnrunner.player_score=49-32
    random.seed(1)
    reward, end_of_game = nnrunner.step(nn_serialize(0,0,1))
    assert end_of_game
    #Check that player score is same as score difference
    assert nnrunner.player_score==nnrunner.game.score[0]-nnrunner.game.score[1]
    #Load game_end_of_round_3 to see that game_board gets reset
    agent=Agent()
    nnrunner = NNRunner(agent)
    nnrunner.game.import_JSON("/tests/resources/game_end_of_round_3.json")
    nnrunner.player_score=49-32
    reward, end_of_game = nnrunner.step(nn_serialize(0,3,0))
    #For this scenario, game should not have ended after one step
    assert not end_of_game
    #With this scenario, reward should be -6
    assert reward == -6
    #Since board is reset, the player_score should be equal to the game score difference
    assert nnrunner.player_score==nnrunner.game.score[0]-nnrunner.game.score[1]
    #Game board should be set
    game_board = nnrunner.game.game_board_displays
    for i in range(5):
        assert np.sum(game_board[i])==4
    game_center = nnrunner.game.game_board_center
    assert np.array_equal(game_center,np.array([0,0,0,0,0,1]))
    assert nnrunner.game.current_player == 1

def test_nnrunner_get_state_flat():
    agent=Agent()
    nnrunner = NNRunner(agent)
    state_flat = nnrunner.get_state_flat()
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
    game.import_JSON("/tests/resources/game_first_round.json")
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
    game.import_JSON("/tests/resources/game_sample_1.json")
    all_valid = check_all_valid(game)
    assert all_valid[nn_serialize(0,0,4)]

def test_opponent_random():
    game=Azul()
    game.import_JSON("/tests/resources/game_first_round.json")
    all_valid = check_all_valid(game)
    #Make a number of random moves and check that they are between 0..180 and that they are valid
    moves=np.zeros(1000,dtype="int")
    for i in range(moves.size):
        moves[i]=opponent_random(game)
        assert all_valid[moves[i]]
    assert np.count_nonzero(moves > 179) == 0
    assert np.count_nonzero(moves < 0) == 0

    
