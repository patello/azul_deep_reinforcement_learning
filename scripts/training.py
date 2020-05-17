from azulnet.agent import Agent
from azulnet.nn_runner import NNRunner
from azulnet.game_runner import GameRunner
import sys

#Crude script to see which options are filled in

if len(sys.argv) < 2:
    batch_size = 10
else:
    batch_size = int(sys.argv[1])
if len(sys.argv) < 3:
    net_name=None
else:
    net_name=sys.argv[2]

batches=1000

agent = Agent(base_net_file=None,learning_rate=3e-4)
game_runner = GameRunner()
nnrunner = NNRunner(agent,game_runner)
nnrunner.train(batch_size=batch_size,batches=batches,net_name=None)
