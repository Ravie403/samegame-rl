from collections import deque
from math import floor
import numpy as np
import random

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

    def reset(self):
        self.board = self.generate(self.width, self.height, self.block_type)
        self.finished = False
        self.actions = self.get_available_actions()
        self.score = 0
        return self.get_board()

    @staticmethod
    def generate(width: int, height: int , block_type: int) -> np.ndarray:
        random.seed(3141592) # 非常に心苦しいがランダム生成では厳しいので固定シードにする
        return np.array([[random.randint(1, block_type)
                          for i in range(width)] for j in range(height)])

    def find_same_blocks(self, block: list) -> list:
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
    def calc_sum_for_arithmetic(n: int, a:int , d: int) -> int:
        return (n / 2) * (2 * a + (n - 1) * d)

    def calc_clear_bonus_score(self, n: int) -> int:
        return self.calc_sum_for_arithmetic(
            max(floor((1 - n / (self.width * self.height)) * 100) - BONUS_THRESHOLD, 0),
            CLEAR_BONUS_INIT, CLEAR_BONUS_DIFF
        )

    def get_finished_bonus(self) -> int:
        return self.calc_clear_bonus_score(np.count_nonzero(self.board != 0))

    def get_available_actions(self) -> np.ndarray:
        result = []
        for x in range(self.width):
            for y in range(self.height):
                if len(self.find_same_blocks([y, x])) > 1:
                    result.append([y, x])
        return np.array(result, dtype=np.float32)

    def delete(self, block: list) -> int:
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
