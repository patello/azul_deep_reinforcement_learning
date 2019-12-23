import gym
from neural.agent import Agent
import sys

#Crude script to see which options are filled in
if len(sys.argv) < 2:
    max_episodes = 1000
else:
    max_episodes = int(sys.argv[1])
if len(sys.argv) < 3:
    net_name="temp"
else:
    net_name=sys.argv[2]

batch_size=1000

agent = Agent(base_net_file="blue_adam_v01",mean_points=1,learning_rate=3e-4)#max(1,max_episodes/1000))
agent.train(max_episode=max_episodes,net_name=net_name,batch_size=batch_size)
