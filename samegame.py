"""
Writer: Ravie403 (Takuma Honjo), <info@tkm.blue>
samegame reinforce learning
thanks to https://qiita.com/uezo/items/87b25c93199d72a56a9a
"""
import chainer
import chainerrl
import numpy as np
from board import Board
from RandomActor import RandomActor
import time
from Q import QFunction

b = Board()
ra = RandomActor(b)

# 環境と行動の次元数
obs_size = 126
n_actions = 126
start = time.perf_counter()
# Q-functionとオプティマイザーのセットアップ
print('setup q-func, optimizer, explorer, and agent...')
s1 = time.perf_counter()
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
e1 = time.perf_counter()
print(f'completed in {round(e1-s1, 3)} sec.')


def parse_statistics(stats):
    avg, los, upd = stats
    return f'Q_avg: {round(avg[1])}, loss_avg: {round(los[1])}, n_upd: {upd[1]}'


# 学習ゲーム回数
n_episodes = 500
# カウンタの宣言
score = 0

STEP_THRESHOLD: int = 100
SAVE_TIMES: int = 50
print("Start training.")
for i in range(1, n_episodes + 1):
    obs = b.reset()
    reward = 0
    done = False
    step = 0
    ra.random_count = 0
    s2 = time.perf_counter()
    while not done and step < STEP_THRESHOLD:
        action = agent.act_and_train(obs, reward // 5)
        obs, reward, done, _ = b.step(action)
        step += 1
    s2 = time.perf_counter() - s2
    reward = b.calc_result() if done else 0
    score = max(score, reward)
    stats = parse_statistics(agent.get_statistics())
    print(f"episode: {i} in {s2 // 60} min. / rnd: {ra.random_count}/{step} / Max: {score} / statistics: {stats} / epsilon: {round(agent.explorer.epsilon, 2)}")

    if i % SAVE_TIMES == 0:
        agent.save(f"results/result_{str(i)}")

    agent.stop_episode_and_train(obs, reward // 5, True)

fin = time.perf_counter()
print(f"Training finished in {(fin-start)//3600} hrs.")
