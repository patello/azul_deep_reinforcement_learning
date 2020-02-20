import gym
from neural.agent import Agent
from game.nn_runner import NNRunner
import sys

#Crude script to see which options are filled in

if len(sys.argv) < 2:
    batch_size = 1000
else:
    batch_size = int(sys.argv[1])
if len(sys.argv) < 3:
    net_name=None
else:
    net_name=sys.argv[2]

batches=1000000

agent = Agent(base_net_file=None,learning_rate=3e-4)
nnrunner = NNRunner(agent)
nnrunner.train(batch_size=batch_size,batches=batches,net_name=net_name)
