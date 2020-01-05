from game.azul import Azul

import math
import random
import copy
import numpy as np
import torch  
import csv

random.seed()

# Class till will help the neural network to run properly
class NNRunner:
    class GameStatistics():
        def __init__(self):
            self.statisticsBuffer = {key : np.empty(0) for key in ["player_score","opponent_score","rounds","percent_first_player","floor_penalty","max_combo","completed_rows","completed_columns","completed_colors"]}
            self.statistics =  {key : np.empty(0) for key in ["player_score","opponent_score","rounds","percent_first_player","floor_penalty","max_combo","completed_rows","completed_columns","completed_colors"]}
        def update(self,statistics):
            for stat in statistics:
                self.statisticsBuffer[stat]=np.append(self.statisticsBuffer[stat],statistics[stat])
        def get_stats(self):
            for stat in self.statistics:
                if len(self.statisticsBuffer[stat]) > 0:
                    self.statistics[stat] = np.append(self.statistics[stat],self.statisticsBuffer[stat].mean())
                    self.statisticsBuffer[stat] = np.empty(0)
            return self.statistics
                
    def __init__(self,agent):
        self.game = Azul()
        self.game_statistics = NNRunner.GameStatistics()
        self.agent=agent
        # Start the game with a set board
        self.game.new_round()
        # NNRunner will keep track of the players relative score, in order to see how much is gained
        # or lost.
        self.player_score = 0
        self.move_counter = 0
    def step(self, i):
        self.game.step(*nn_deserialize(i))
        self.move_counter += 1
        while (self.game.current_player != 1 or np.count_nonzero(self.get_valid_moves()) < 2) and not self.game.is_end_of_game():
            state=self.get_state_flat(perspective=self.game.current_player-1)
            valid_moves = torch.from_numpy(self.get_valid_moves().reshape(1,180))
            action,_,_ = self.agent.get_ac_output(state,valid_moves)
            self.game.step(*nn_deserialize(action))
            self.move_counter += 1
        game_copy=copy.deepcopy(self.game)
        game_copy.count_score()
        new_player_score = game_copy.score[0]-game_copy.score[1]
        reward = new_player_score-self.player_score
        self.player_score = new_player_score
        return reward, self.game.is_end_of_game()
    def get_state_flat(self,perspective=0):
        order = [perspective] + list(set(range(self.game.players))-set([perspective]))
        if self.game.next_first_player > 0:
            perspective_next_first_player = ((self.game.next_first_player - 1 - perspective) % self.game.players) + 1
        else:
            perspective_next_first_player = 0
        #Need to make sure this works even for more players. The code below can be used.
        #print(order)
        #print("Next: "+str(self.game.next_first_player)+", Perspective: "+str(perspective)+", From perspective: "+str(perspective_next_first_player))
        return np.concatenate((
            self.game.game_board_displays.flatten(),
            self.game.game_board_center,
            self.game.pattern_lines[order].flatten(),
            self.game.walls[order].flatten(),
            self.game.floors[order],
            self.game.score[order],
            [perspective_next_first_player]))
    def get_valid_moves(self):
        #TODO: Write test for this
        return check_all_valid(self.game)
    def reset(self):
        #TODO: Write test for this
        #Changed this to only run parts of initialization, since I want to keep game statistics intact
        self.game = Azul()
        self.game.new_round()
        self.player_score = 0
        self.move_counter = 0
    def run_episode(self):
        rewards = []
        values = []
        log_probs = []
        entropy_term = 0
        self.reset()
        state = self.get_state_flat()
        #Fixed range for 200 max steps, should be sufficient.
        for steps in range(200):
            valid_moves = torch.from_numpy(self.get_valid_moves().reshape(1,180))
            action, policy_dist, value = self.agent.get_ac_output(state,valid_moves)
            reward, done = self.step(action)  
            new_state = self.get_state_flat()

            log_prob = torch.log(policy_dist.squeeze(0)[action])
            
            entropy = -torch.sum(policy_dist.mean() * torch.log(policy_dist))

            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state
            if done:
                self.game_statistics.update(self.game.get_statistics())
                return rewards, values, log_probs, entropy_term
    def run_batch(self, episodes):
        for episode in range(episodes):
            self.run_episode()
        for stat in self.game_statistics.get_stats():
            print(stat + ": " + str(self.game_statistics.get_stats()[stat][-1]))
    def train(self, net_name=None, batch_size=1000, batches=1000):
        if net_name is not None:
            with open('/neural/results/'+net_name+'.csv', mode="w") as csv_file:
                        result_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        result_file.writerow(["batch"]+list(self.agent.agent_statistics.statistics.keys())+list(self.game_statistics.statistics.keys()))
        for batch in range(batches):
            rewards = []
            values = []
            log_probs = []
            qvals = np.array([])
            entropy_term = 0
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
                with open('/neural/results/'+net_name+'.csv', mode="a+") as csv_file:
                    result_file = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    result_file.writerow(np.concatenate([[batch+1],[stat_value[-1] for stat_value in self.agent.agent_statistics.get_stats().values()],[stat_value[-1] for stat_value in self.game_statistics.get_stats().values()]]))
            if (batch +1)% 1000 == 0 and net_name is not None:
                torch.save(self.agent.ac_net,"/neural/models/"+net_name+".mx")
                        
        
# Returns a integer between 1..180
def nn_serialize(display,color,pattern):
    return display+color*6+pattern*5*6

# Takes an integer between 1..180 (TODO: Implement out of bounds check) 
# and returns the corresponding display, color and pattern
def nn_deserialize(i):
    display = i%6
    color = math.floor(i/6)%5
    pattern = math.floor(i/(5*6))
    return (display, color, pattern)

def check_all_valid(game):
    all_valid = np.zeros(180,dtype="bool")
    for i in range(6*5*6):
        all_valid[i]=game.is_legal_move(*nn_deserialize(i))
    return all_valid

def opponent_random(game):
    weight_table=np.ones(180)
    #Weight of the straight to floor moves should be lower than the others
    for i in range(6):
        for j in range(5):
            weight_table[nn_serialize(i,j,0)]=0.01
    all_valid = check_all_valid(game)
    #Weight should be set to zero for invalid moves
    weight_table=np.multiply(weight_table,all_valid)
    return random.choices(range(180),weights=weight_table)[0]
