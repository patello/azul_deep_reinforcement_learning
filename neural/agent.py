import numpy as np  
import torch  
import torch.optim as optim
from game.nn_runner import NNRunner
from torch.autograd import Variable
from model import ActorCritic
import csv

class Agent():

    class AgentStatistics():
        def __init__(self,nr_of_points=10):
            self.nr_of_points=nr_of_points
            self.statisticsBuffer = {"reward" : np.empty(0), "actor_loss" : np.empty(0), "critic_loss" : np.empty(0), "ac_loss" : np.empty(0)}
            self.statistics = {"reward" : np.empty(0), "actor_loss" : np.empty(0), "critic_loss" : np.empty(0), "ac_loss" : np.empty(0)}
        def update(self,statistics):
            for stat in statistics:
                self.statisticsBuffer[stat]=np.append(self.statisticsBuffer[stat],statistics[stat])
                if self.statisticsBuffer[stat].size >= self.nr_of_points:
                    self.statistics[stat] = np.append(self.statistics[stat],self.statisticsBuffer[stat].mean())
                    self.statisticsBuffer[stat] = np.empty(0)
                
    def __init__(self, base_net_file=None, base_net="Blue Adam", mean_points=1000, use_cnn=False, learning_rate=3e-4, gamma=0.99, buffer_size=10000):
        self.env = NNRunner(mean_points=mean_points)
        self.mean_points=mean_points
        self.agent_statistics = Agent.AgentStatistics(nr_of_points=mean_points)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_in = 136
        self.num_out = 180

        if base_net_file is None:
            if base_net == "Blue Adam":
                self.ac_net = ActorCritic(self.num_in, self.num_out)
        else:
            self.ac_net = torch.load("/usr/neural/models/"+base_net_file+".mx")
        self.ac_optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate)
        
    def update(self, rewards, values, log_probs, entropy):

        qvals = np.zeros((len(values),1))
        
        for episode in range(len(rewards)):
            qval = 0
            for t in reversed(range(len(rewards[episode]))):
                qval = rewards[episode][t] + self.gamma * qval
                qvals[t] = [qval]

        values = torch.stack(values).squeeze(2)
        qvals = torch.FloatTensor(qvals)
        log_probs = torch.stack(log_probs)

        advantage = qvals - values
        actor_loss = (-log_probs * advantage).mean()
        critic_loss = advantage.pow(2).mean()
        #Entropy term becomes infinite when log_probs are zero and needed to be removed from the loss function
        ac_loss = actor_loss + 0.5 * critic_loss# - 0.001 * entropy
        self.ac_optimizer.zero_grad()
        ac_loss.backward()
        self.ac_optimizer.step()
        statistics = {"reward" : np.sum(rewards), "actor_loss" : actor_loss.detach().numpy().squeeze(0), "critic_loss" : critic_loss.detach().numpy().squeeze(0), "ac_loss" : ac_loss.detach().numpy().squeeze(0)}
        self.agent_statistics.update(statistics)

    def get_ac_output(self, state,done=False):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value, policy_dist = self.ac_net.forward(state,torch.from_numpy(self.env.get_valid_moves().reshape(1,180)))
        if done:
            return 0,0,value
        action = np.random.choice(self.num_out, p=policy_dist.detach().numpy().squeeze(0))
        return action, policy_dist, value

    def train(self, max_episode, net_name, batch_size=100):
        with open('/usr/neural/results/'+net_name+'.csv', mode="w") as csv_file:
                    result_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    result_file.writerow(["episode"]+list(self.agent_statistics.statistics.keys())+list(self.env.game_statistics.statistics.keys()))
        for batch in range(batch_size):
            rewards = []
            values = []
            log_probs = []
            entropy_term = 0
            for episode in range(int(max_episode/batch_size)):
                rewards.append([])
                episode_reward = 0
                
                self.env.game_statistics.update(self.env.game.get_statistics())
                self.env.reset()
                state = self.env.get_state_flat()
                #Fixed range for 200 max steps, should be sufficient.
                for steps in range(200):
                    action, policy_dist, value = self.get_ac_output(state)
                    reward, done = self.env.step(action)  
                    new_state = self.env.get_state_flat()

                    log_prob = torch.log(policy_dist.squeeze(0)[action])
                    
                    entropy = -torch.sum(policy_dist.mean() * torch.log(policy_dist))

                    rewards[episode].append(reward)
                    values.append(value)
                    log_probs.append(log_prob)
                    entropy_term += entropy
                    state = new_state
                    episode_reward += reward
                    if done:
                        if episode % self.mean_points == 0:
                            torch.save(self.ac_net,"/usr/neural/models/"+net_name+".mx")
                        break
            self.update(rewards, values, log_probs, entropy_term)
            #if (batch*batch_size+1) % self.mean_points == 0:                    
            with open('/usr/neural/results/'+net_name+'.csv', mode="a+") as csv_file:
                result_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                result_file.writerow(np.concatenate([[batch*batch_size+1],[stat_value[-1] for stat_value in self.agent_statistics.statistics.values()],[stat_value[-1] for stat_value in self.env.game_statistics.statistics.values()]]))
        