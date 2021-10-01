from random import choice
import numpy as np


class RandomActor:
    def __init__(self, board):
        super().__init__()
        self.board = board
        self.random_count = 0

    def random_action_func(self):
        self.random_count += 1
        return choice(self.board.actions)
