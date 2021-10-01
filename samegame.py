"""
Writer: Ravie403 (Takuma Honjo), <info@tkm.blue>
samegame reinforce learning
thanks to https://qiita.com/uezo/items/87b25c93199d72a56a9a
"""
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np
from board import Board
from RandomActor import RandomActor


# Q関数
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels):
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_hidden_channels),
            l3=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        h = F.leaky_relu(self.l2(h))
        act = chainerrl.action_value.DiscreteActionValue(self.l3(h))
        # print(x.shape, h.shape, act)
        return act


b = Board()
ra = RandomActor(b)

# 環境と行動の次元数
obs_size = 126
n_actions = 126

# Q-functionとオプティマイザーのセットアップ
q_func = QFunction(obs_size, n_actions, obs_size**2)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
# 報酬の割引率
gamma = 0.95

explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.3, decay_steps=50000, random_action_func=ra.random_action_func)
# Experience ReplayというDQNで用いる学習手法で使うバッファ
# TODO: DQN => A3C
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
# Agentの生成
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100)

# 学習ゲーム回数
n_episodes = 2000
# カウンタの宣言
score = 0
last_state = None
for i in range(1, n_episodes + 1):
    b.reset()
    reward = 0
    while not b.finished:
        action = agent.act_and_train(b.getBoard(), reward)
        if type(action) == np.int32:
            action = [action // 14, action % 14]
        else:
            action = action.astype(np.int32)
        reward += b.delete(action)
        last_state = b.getBoard()
    score = max(score, b.calcResult())
    agent.stop_episode_and_train(b.getBoard(), score, True)

    if i % 100 == 0:
        print(f"episode: {i} / rnd: {ra.random_count} / Maxscore: {score} / statistics: {agent.get_statistics()} / epsilon: {agent.explorer.epsilon}")
    if i % 10000 == 0:
        agent.save(f"result_{str(i)}")

print("Training finished.")
