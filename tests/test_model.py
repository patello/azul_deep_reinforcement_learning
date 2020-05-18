import pytest
import numpy as np
import torch

from torch.autograd import Variable
from azulnet import ActorCritic, Agent, NNRunner
from azulnet.model import IllegalMask
from azulnet.game_runner import GameRunner

def test_model_forward():
    ac_net = ActorCritic(136,180)
    agent=Agent()
    game_runner=GameRunner()
    nnrunner = NNRunner(agent,game_runner)
    state = Variable(torch.from_numpy(nnrunner.game_runner.get_state()).float().unsqueeze(0))
    #Test that the correct thing gets returned from the forward pass
    policy_dist, log_policy_dist = ac_net.forward_actor(state)
    value = ac_net.forward_critic(state)
    assert type(value)==torch.Tensor
    assert type(policy_dist)==torch.Tensor
    assert type(log_policy_dist)==torch.Tensor
    assert policy_dist.detach().numpy().size == 180
    assert np.sum(policy_dist.detach().numpy().squeeze(0)) == pytest.approx(1)
    policy_dist, log_policy_dist = ac_net.forward_actor(state)
    value = ac_net.forward_critic(state)
    #Test that masking works ok
    mask = np.zeros((1,180),dtype='bool')
    mask[(0,1)]=True
    policy_dist, log_policy_dist = ac_net.forward_actor(state,torch.from_numpy(mask))
    value = ac_net.forward_critic(state)
    assert type(value)==torch.Tensor
    assert type(policy_dist)==torch.Tensor
    assert type(log_policy_dist)==torch.Tensor
    assert policy_dist[(0,1)]==1
    assert np.sum(policy_dist.detach().numpy().squeeze(0)) == pytest.approx(1)
    #Test that masking with only zeros raises an exception
    mask = np.zeros((1,180),dtype='bool')
    with pytest.raises(IllegalMask):
        ac_net.forward_actor(state,torch.from_numpy(mask))
