import numpy as np  
import torch  
import torch.optim as optim
from game.nn_runner import NNRunner
from torch.autograd import Variable
from model import ActorCritic
import csv

class Agent():

    def __init__(self, use_cnn=False, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = NNRunner()
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_in = 136
        self.num_out = 180

        self.ac_net = ActorCritic(self.num_in, self.num_out)
        self.ac_optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate)
    
    def update(self, rewards, values, next_value, log_probs, entropy):
        qvals = np.zeros(len(values))
        qval = next_value
        for t in reversed(range(len(rewards))):
            qval = rewards[t] + self.gamma * qval
            qvals[t] = qval
        
        values = torch.FloatTensor(values)
        qvals = torch.FloatTensor(qvals)
        log_probs = torch.stack(log_probs)

        advantage = qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = advantage.pow(2).mean()
        ac_loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        self.ac_optimizer.zero_grad()
        ac_loss.backward()
        self.ac_optimizer.step()

    def get_ac_output(self, state,done=False):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value, policy_dist = self.ac_net.forward(state)
        if done:
            return 0,0,value
        valid_policy_dist = np.multiply(policy_dist.detach().numpy().squeeze(0),self.env.get_valid_moves())
        try:
            action = np.random.choice(self.num_out, p=valid_policy_dist/np.sum(valid_policy_dist))
        except:
            print(str(self.env.get_state_flat()))
        return action, policy_dist, value

    def train(self, max_episode, max_step):
        for episode in range(max_episode):
            rewards = []
            values = []
            log_probs = []
            entropy_term = 0
            episode_reward = 0
            
            self.env.reset()
            state = self.env.get_state_flat()
            for steps in range(max_step):
                action, policy_dist, value = self.get_ac_output(state)
                reward, done = self.env.step(action)  
                new_state = self.env.get_state_flat()

                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -torch.sum(policy_dist.mean() * torch.log(policy_dist))
                
                rewards.append(reward)
                values.append(value.detach().numpy()[0])
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state
                episode_reward += reward
                if done:
                    if episode % 1 == 0:                    
                        #print("episode: " + str(episode) + ": " + str(episode_reward)) 
                        with open('/usr/neural/result.txt', mode="a+") as csv_file:
                            result_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            result_file.writerow([episode,episode_reward])
                    break

            _, _, next_value = self.get_ac_output(state,done=True)
            self.update(rewards, values, next_value, log_probs, entropy_term)