from collections import deque
from math import floor
import numpy as np
import random

CLEAR_BONUS_INIT = 25
CLEAR_BONUS_DIFF = 50


def calcSumForArithmetic(n, a, d):
    return (n / 2) * (2 * a + (n - 1) * d)


def calcClearBonusScore(n):
    return calcSumForArithmetic(max(floor((1 - n / (14 * 9)) * 100) - 85, 0),
                                CLEAR_BONUS_INIT, CLEAR_BONUS_DIFF)


class Board():
    def __init__(self):
        super().__init__()
        self.reset()

    def getBoard(self):
        return self.board.flatten().astype(np.float32)

    def reset(self):
        self.board = self.generate()
        self.finished = False
        self.actions = self.get_available_actions()
        self.score = 0

    def generate(self):
        return np.array([[random.randint(1, 5)
                          for i in range(14)] for j in range(9)], dtype=np.int32)

    def findSameBlocks(self, block):
        dr = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        q = deque()
        x = block[1]
        y = block[0]
        color = self.board[y][x]
        if color == 0:
            return []
        result = []
        visited = np.full((9, 14), False)
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
                if all([0 <= dx < 14, 0 <= dy < 9]) and all([not visited[dy][dx], self.board[dy][dx] == color]):
                    q.append([dy, dx])
        return result

    def calcResult(self):
        n = np.count_nonzero(self.board != 0)
        return np.int32(self.score + calcClearBonusScore(n))

    def get_available_actions(self):
        result = []
        for x in range(14):
            for y in range(9):
                if len(self.findSameBlocks([y, x])) > 1:
                    result.append([y, x])
        return np.array(result, dtype=np.float32)

    def delete(self, block):
        blocks = self.findSameBlocks(block)
        if (len(blocks) < 2):
            return -100
        for b in blocks:
            self.board[b[0]][b[1]] = 0
        score = len(blocks) ** 2 * 5
        self.score += score
        self.applyGravity()
        self.actions = self.get_available_actions()
        if self.actions.size == 0:
            self.finished = True
        return score

    def applyGravity(self):
        for y in range(14):
            v = self.board[:, y]
            v = v[v != 0]
            z = np.zeros(9 - len(v), dtype=np.int32)
            self.board[:, y] = np.hstack([z, v])
        v = self.board[:, np.any(self.board != 0, axis=0)]
        z = np.zeros([9, 14 - v.shape[1]], dtype=np.int32)
        self.board = np.hstack([z, v])


if __name__ == '__main__':
    b = Board()
    while not b.finished:
        b.delete(b.actions[0])
    print("="*30)
    print(b.board)
    print("=" * 30)
    print(b.calcResult())
