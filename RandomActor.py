from random import choice
import numpy as np


class RandomActor:
    def __init__(self, board):
        super().__init__()
        self.board = board
        self.random_count = 0

    def random_action_func(self):
        self.random_count += 1
        chosen = choice(self.board.get_available_actions().astype(np.int32))
        y, x = chosen
        return y*14+x
