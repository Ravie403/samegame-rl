from collections import deque
from math import floor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random

from numpy.core.fromnumeric import repeat

CLEAR_BONUS_INIT = 25
CLEAR_BONUS_DIFF = 50
BONUS_THRESHOLD = 85


class Board():
    def __init__(self, width: int = 14, height: int = 9, block_type: int = 5):
        self.width = width
        self.height = height
        self.block_type = block_type
        self.board = self.generate(width, height, block_type)
        self.finished = False
        self.actions = self.get_available_actions()
        self.score = 0

    def get_board(self):
        return self.board.copy().flatten().astype(np.float32)

    def step(self, block):
        score = self.delete([block // self.width, block % self.width])
        return self.get_board(), score, self.finished, {}

    def action(self, action):
        self.delete(action.astype(np.int32))
        return self.finished

    def reset(self):
        self.board = self.generate(self.width, self.height, self.block_type)
        self.finished = False
        self.actions = self.get_available_actions()
        self.score = 0
        return self.get_board()

    @staticmethod
    def generate(width, height, block_type) -> np.ndarray:
        return np.array([[random.randint(1, block_type)
                          for i in range(width)] for j in range(height)])

    def find_same_blocks(self, block) -> list:
        dr = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        q = deque()
        y, x = block
        color = self.board[y][x]
        if color == 0:
            return []
        result = []
        visited = np.full((self.height, self.width), False)
        q.append(block)
        while len(q) != 0:
            p = q.popleft()
            y, x = p
            if visited[y][x]:
                continue
            result.append([y, x])
            visited[y][x] = True
            for d in dr:
                dx, dy = x + d[0], y + d[1]
                if all([0 <= dx < self.width, 0 <= dy < self.height]) and all([not visited[dy][dx], self.board[dy][dx] == color]):
                    q.append([dy, dx])
        return result

    def calc_result(self) -> int:
        n = np.count_nonzero(self.board != 0)
        return np.int32(self.score + self.calc_clear_bonus_score(n))

    @staticmethod
    def calc_sum_for_arithmetic(n, a, d) -> int:
        return (n / 2) * (2 * a + (n - 1) * d)

    def calc_clear_bonus_score(self, n) -> int:
        return self.calc_sum_for_arithmetic(max(floor((1 - n / (self.width * self.height)) * 100) - BONUS_THRESHOLD, 0),
                                            CLEAR_BONUS_INIT, CLEAR_BONUS_DIFF)

    def get_finished_bonus(self) -> int:
        return self.calc_clear_bonus_score(np.count_nonzero(self.board != 0))

    def get_available_actions(self) -> np.ndarray:
        result = []
        for x in range(self.width):
            for y in range(self.height):
                if len(self.find_same_blocks([y, x])) > 1:
                    result.append([y, x])
        return np.array(result, dtype=np.float32)

    def delete(self, block) -> int:
        blocks = self.find_same_blocks(block)
        if len(blocks) < 2:
            return 0
        for b in blocks:
            self.board[b[0]][b[1]] = 0
        score = len(blocks) ** 2 * 5
        self.score += score
        self.apply_gravity()
        self.actions = self.get_available_actions()
        if self.actions.size == 0:
            self.finished = True
        return score

    def apply_gravity(self) -> None:
        for y in range(self.width):
            v = self.board[:, y]
            v = v[v != 0]
            z = np.zeros(self.height - len(v), dtype=np.int32)
            self.board[:, y] = np.hstack([z, v])
        v = self.board[:, np.any(self.board != 0, axis=0)]
        z = np.zeros([self.height, self.width - v.shape[1]], dtype=np.int32)
        self.board = np.hstack([z, v])


if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    b = Board()
    an = ax.pcolor(b.board.copy(), cmap='Accent', vmin=0, vmax=7)
    anis = []
    anis.append([an])
    while not b.finished:
        s = b.delete(b.get_available_actions().astype(np.int32)[0])
        anis.append([ax.pcolor(b.board.copy(), cmap='Accent', vmin=0, vmax=7), ax.text(
            0.5, 1.01, f'Score:{b.score} +{s}', ha='center', va='bottom', transform=ax.transAxes, fontsize='large')])
    ani = animation.ArtistAnimation(fig, anis, interval=100, repeat_delay=1000)
    # ani.save('animation.gif')
    plt.show()
