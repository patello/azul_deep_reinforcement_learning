import random
import copy
import numpy as np
import torch  
import csv

from azulnet.azul import Azul
from azulnet.game_runner import GameRunner

random.seed()

# Class till will help the neural network to run properly
class NNRunner(GameRunner):    
    def __init__(self,agent,opponent=None,rules={"first_player":"Random","tile_pool":"Lid"}):
        self.agent=agent
        super().__init__(opponent, rules)
    def run_episode(self):
        rewards = []
        values = []
        log_probs = []
        entropy_terms = []
        self.reset()
        #Check if agent is not first player, in that case, the opponent makes a move first.
        while (self.game.current_player != 1):
            self.opponent_move()
        state = self.get_state_flat()
        #Fixed range for 200 max steps, should be sufficient.
        for steps in range(200):
            valid_moves = torch.from_numpy(self.get_valid_moves().reshape(1,180))
            action, policy_dist, log_policy_dist, value = self.agent.get_ac_output(state,valid_moves)
            reward, done = self.step(action)  
            new_state = self.get_state_flat()

            log_prob = log_policy_dist.squeeze(0)[action]
            
            #When calculating entropy, we need to only look at the valid policy distribution. Otherwise, entropy is infinite
            #Masking seems to be needed even when the built in log_softmax is used.
            valid_log_policy_dist = log_policy_dist.masked_select(valid_moves)
            
            #Using the mean() funciton, Should give same result as the original code:
            #-torch.sum(policy_dist.masked_select(valid_moves).mean() * torch.log(policy_dist.masked_select(valid_moves)))
            entropy = -valid_log_policy_dist.mean()

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_terms.append(entropy)
            state = new_state
            if done:
                self.game_statistics.update(self.game.get_statistics())
                return rewards, values, log_probs, entropy_terms
    def run_batch(self, episodes):
        for episode in range(episodes):
            self.run_episode()
        for stat in self.game_statistics.get_stats():
            print(stat + ": " + str(self.game_statistics.get_stats()[stat][-1]))
    def train(self, net_name=None, batch_size=1000, batches=1000):
        if net_name is not None:
            with open('/results/'+net_name+'.csv', mode="w") as csv_file:
                        result_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        result_file.writerow(["batch"]+list(self.agent.agent_statistics.statistics.keys())+list(self.game_statistics.statistics.keys()))
        for batch in range(batches):
            rewards = []
            values = []
            log_probs = []
            qvals = np.array([])
            entropy_term = []
            for episode in range(batch_size):
                episode_rewards, episode_values, episode_log_probs, episode_entropy_term = self.run_episode()
                rewards.append(np.sum(episode_rewards))
                values += episode_values
                log_probs += episode_log_probs
                entropy_term += episode_entropy_term
                episode_qvals = np.zeros(len(episode_rewards))

                qval = 0
                for t in reversed(range(len(episode_rewards))):
                    qval = episode_rewards[t] + self.agent.gamma * qval
                    episode_qvals[t] = qval
                qvals = np.concatenate((qvals,episode_qvals))

            self.agent.update(np.reshape(qvals,(-1,1)), rewards, values, log_probs, entropy_term)
            if (batch+1) % max(1,(batches/1000)) == 0 and net_name is not None:                    
                with open('/results/'+net_name+'.csv', mode="a+") as csv_file:
                    result_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    result_file.writerow(np.concatenate([[batch+1],[stat_value[-1] for stat_value in self.agent.agent_statistics.get_stats().values()],[stat_value[-1] for stat_value in self.game_statistics.get_stats().values()]]))
            if (batch +1)% 1000 == 0 and net_name is not None:
                torch.save(self.agent.ac_net,"/results/"+net_name+".mx")