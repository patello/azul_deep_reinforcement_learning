from game.nn_runner import *
from neural.agent import Agent
import copy
import os
import numpy as np

#Script dir as per suggestion here: https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
script_dir = os.path.dirname(__file__)

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
    #Score on the original board should not have been counted
    assert np.array_equal(nnrunner.game.score,np.zeros(2))
    #Load game_end_of_round_2 to see that game_board gets reset and player gets back to one
    agent=Agent()
    nnrunner = NNRunner(agent)
    nnrunner.game.import_JSON(script_dir+"/resources/game_end_of_round_2.json")
    reward, end_of_game = nnrunner.step(nn_serialize(0,4,1))
    assert nnrunner.game.current_player == 1
    #For this scenario, game should have ended after one step
    assert not end_of_game
    #Load game_end_of_round_2 to see that the game ends if the right steps are made
    agent=Agent()
    nnrunner = NNRunner(agent)
    nnrunner.game.import_JSON(script_dir+"/resources/game_end_of_round_2.json")
    nnrunner.player_score=49-32
    random.seed(1)
    reward, end_of_game = nnrunner.step(nn_serialize(0,0,1))
    assert end_of_game
    #Check that player score is same as score difference
    assert nnrunner.player_score==nnrunner.game.score[0]-nnrunner.game.score[1]
    #Load game_end_of_round_3 to see that game_board gets reset
    agent=Agent()
    nnrunner = NNRunner(agent)
    nnrunner.game.import_JSON(script_dir+"/resources/game_end_of_round_3.json")
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

def test_nn_runner_run_episode():
    agent=Agent()
    nnrunner = NNRunner(agent)
    rewards, values, log_probs, entropy_term = nnrunner.run_episode()
    assert len(rewards) == len(values) == len(log_probs)
    assert len(rewards) > 0
    #Entropy should be a single value
    assert list(entropy_term.size())==[]

def test_nn_runner_train():
    agent=Agent()
    nnrunner = NNRunner(agent)
    #Test that training with different batch sizes are possible
    nnrunner.train(batch_size=1,batches=1)
    nnrunner.train(batch_size=2,batches=1)
    nnrunner.train(batch_size=1,batches=2)
    #Test that training works with a random agent
    opponent=RandomAgent()
    nnrunner = NNRunner(agent,opponent)
    nnrunner.train(batch_size=2,batches=2)
    #Test that training works with a default agent
    opponent=Agent()
    nnrunner = NNRunner(agent,opponent)
    nnrunner.train(batch_size=2,batches=2)

def test_nn_runner_train_1_1(benchmark):
    #Benchmark test for training with one 
    agent=Agent()
    opponent=Agent()
    nnrunner = NNRunner(agent,opponent)
    benchmark.pedantic(nnrunner.train,kwargs={"batch_size":1,"batches":1},rounds=10)

def test_nn_runner_run_batch():
    agent=Agent()
    nnrunner = NNRunner(agent)
    #Test that training with different batch sizes are possible
    nnrunner.run_batch(episodes=1)
    nnrunner.run_batch(episodes=2)
    #Test that training works with a random agent
    opponent=RandomAgent()
    nnrunner = NNRunner(agent,opponent)
    nnrunner.run_batch(episodes=2)
    #Test that training works with a default agent
    opponent=Agent()
    nnrunner = NNRunner(agent,opponent)
    nnrunner.run_batch(episodes=2)

def test_random_agent_get_ac_output():
    opponent=RandomAgent()
    game=Azul()
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    all_valid = check_all_valid(game)
    #Make a number of random moves and check that they are between 0..180 and that they are valid
    moves=np.zeros(1000,dtype="int")
    for i in range(moves.size):
        moves[i]=opponent.get_a_output(None,torch.from_numpy(all_valid.reshape(1,180)))
        assert all_valid[moves[i]]
    assert np.count_nonzero(moves > 179) == 0
    assert np.count_nonzero(moves < 0) == 0

def test_nn_runner_run_batch_agent(benchmark):
    #Run batch against randomly initialized opponents
    agent = Agent(base_net_file=None)
    opponent = Agent(base_net_file=None)
    nnrunner = NNRunner(agent,opponent)
    benchmark(nnrunner.run_batch,1)

def test_nn_runner_run_batch_random(benchmark):
    #Run batch against random opponent
    agent = Agent(base_net_file=None)
    opponent = RandomAgent()
    nnrunner = NNRunner(agent,opponent)
    benchmark(nnrunner.run_batch,1)
    
