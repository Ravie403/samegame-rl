from random import choice, seed
import numpy as np


class RandomActor:
    def __init__(self, board):
        super().__init__()
        self.board = board
        self.random_count = 0

    def random_action_func(self):
        self.random_count += 1
        seed() # 別関数でseed値の固定が行われているので一度シード値を変更してから実行する
        chosen = choice(self.board.get_available_actions().astype(np.int32))
        y, x = chosen
        return y*14 + x
