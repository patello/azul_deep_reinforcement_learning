import copy
import os
import numpy as np
import torch
import random

from azulnet import NNRunner, Agent, RandomAgent, Azul
from azulnet import check_all_valid, nn_serialize, nn_deserialize

#Script dir as per suggestion here: https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
script_dir = os.path.dirname(__file__)

def test_init():
    agent=Agent()
    nnrunner = NNRunner(agent)

def test_nn_runner_run_episode():
    agent=Agent()
    nnrunner = NNRunner(agent)
    rewards, values, log_probs, entropy_term = nnrunner.run_episode()
    assert len(rewards) == len(values) == len(log_probs) == len(entropy_term)
    assert len(rewards) > 0

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
    
