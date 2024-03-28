import random
import numpy as np
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from typing import Dict
import matplotlib.pyplot as plt


class Player:
    def __init__(self, name: str, turn: int) -> None:
        self.name = name
        self.turn = turn # -1|1

    def move(self, board_arr) -> int:
        raise NotImplementedError("Implemented by child class.")


class Bot(Player):
    def __init__(self, name: str, turn: int) -> None:
        super().__init__(name, turn)

    def move(self, board_arr) -> int:
        """Make move based on current board state."""
        available_cols = board_arr.sum(axis=0) < board_arr.shape[0]
        available_cols = [c for c, i in zip(list(range(board_arr.shape[1])), available_cols) if i]
        col = random.choice(available_cols)
        return col
    
class RLBot(Player):
    def __init__(self, name: str, turn: int) -> None:
        super().__init__(name, turn)

        self.activation = torch.nn.ReLU
        self.model = self.initialize_model()
        self.loss_fn = torch.nn.MSELoss()
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr) 
        self.gamma = 0.9
        self.epsilon = 0.2

        self.losses = []
        self.rewards = []
        #sqars: state_t, q_vals_t, action_t, reward_t, state_t+1
        self.current_sqars = [None, None, None, None, None]
        self.reward_vals = {
            'draw': 5,
            'win': 10,
            'loss': -10,
            'move': 0,
        }

    def initialize_model(self):
        input_n = 84
        hidden_n = 150
        hidden_n_2 = 100
        output_n = 7
        model = torch.nn.Sequential(
            torch.nn.Linear(input_n, hidden_n),
            self.activation(),
            torch.nn.Linear(hidden_n, hidden_n_2),
            self.activation(),
            torch.nn.Linear(hidden_n_2, output_n),
        )
        model.to(device)
        return model
    
    def get_state_array(self, piece_arrays: Dict):
        state_array = np.stack([piece_arrays[self.turn], piece_arrays[-self.turn]])
        return state_array

    def process_state(self, curr_state: np.array):
        num_cells = curr_state[0].shape[0]*curr_state[0].shape[1]
        curr_state = curr_state.reshape(1, num_cells*2)
        if isinstance(self.activation, torch.nn.ReLU):
            curr_state += np.random.rand(1, num_cells*2)/10.0

        state = torch.from_numpy(curr_state).float().to(device)
        return state

    def move(self, piece_arrays: Dict) -> int:
        """
        @curr_state: 6x7x2 array, rows x cols x sides
        Sides represent friendly and enemy pieces.
        """
        curr_state = self.get_state_array(piece_arrays)
        state = self.process_state(curr_state)
        q_vals = self.model(state)
        q_vals_ = q_vals.data.cpu().numpy()

        if random.random() < self.epsilon:
            action_ = np.random.randint(0, 7)
        else:
            action_ = np.argmax(q_vals_)
        self.current_sqars[0] = state
        self.current_sqars[1] = q_vals
        self.current_sqars[2] = action_
        return action_
    
    def get_reward(self, move_result):
        if move_result is None:
            reward = self.reward_vals['move']
        elif 'draw' in move_result:
            reward = self.reward_vals['draw']
        else:
            win_side = int(move_result.split('_')[-1])
            reward = self.reward_vals['win'] if win_side==self.turn else self.reward_vals['loss']
        self.current_sqars[3] = reward
        self.rewards.append(reward)
        return reward
    
    def reset_vars(self):
        self.current_sqars = [None, None, None, None, None]
    
    def train(self, new_piece_arrays, result):
        new_state = self.get_state_array(new_piece_arrays)
        new_state = self.process_state(new_state)

        self.current_sqars[4] = new_state

        reward = self.get_reward(result)
        
        # get Q values of new state to update last state's Q values
        with torch.no_grad():
            new_q = self.model(new_state)
        max_q = torch.max(new_q)

        # target value
        Y = reward if result is not None else reward + (self.gamma*max_q)
        Y = torch.Tensor([Y]).detach().squeeze().to(device)
        X = self.current_sqars[1].squeeze()[self.current_sqars[2]]
        loss = self.loss_fn(X, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.losses.append(loss.item())
        self.optimizer.step()

        if result is not None: # game epoch is over
            self.reset_vars()

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path).state_dict())

    def plot_results(self, show=False, save_path=None):
        """Plots losses, rewards by move"""
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("Moves")
        ax1.set_ylabel("Rewards")
        ax1.plot(np.arange(len(self.rewards)), self.rewards, color='r')

        ax2 = ax1.twinx()
        ax2.set_ylabel("Loss")
        ax1.plot(np.arange(len(self.rewards)), self.losses, color='b')

        fig.tight_layout()

        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()

    def save_model_and_results(self, model_path: str):
        # model
        torch.save(self.model, model_path+'model.pth')
        # results: losses, rewards, avg. win rate
        np.savetxt(model_path+'losses.csv', np.array(self.losses), delimiter=',', header='losses')
        self.plot_results(save_path=model_path+'results.png')

class Human(Player):
    def __init__(self, name: str, turn: int) -> None:
        super().__init__(name, turn)
