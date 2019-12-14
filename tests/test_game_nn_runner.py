from game.nn_runner import *

def test_nn_runner_init():
    #NNRunner should start a 2 player game
    assert NNRunner().game.players == 2
    #Move counter should be set to 0
    assert NNRunner().move_counter == 0

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