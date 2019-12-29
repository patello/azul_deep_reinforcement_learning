import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class IllegalMask(Exception):
    pass

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=180, learning_rate=3e-6):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)
    
    def forward(self, state_tensor, mask=None):
        #Not used anywhere, consider removing
        if mask is None:
            mask=torch.from_numpy(np.ones((1,180),dtype='bool'))
        #Check that mask consist of at least one "True", otherwise forward will return NaN
        elif np.sum(mask.numpy()) == 0:
            raise IllegalMask
        value = F.relu(self.critic_linear1(state_tensor))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state_tensor))
        policy_dist = self.actor_linear2(policy_dist)
        policy_dist[~mask]=float('-inf')
        policy_dist = F.softmax(policy_dist, dim=1)

        return value, policy_dist