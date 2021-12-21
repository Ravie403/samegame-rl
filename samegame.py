"""
Writer: Ravie403 (Takuma Honjo), <info@tkm.blue>
samegame reinforce learning
"""
# stdlib
import logging
import sys
import time

# learning lib
import pfrl
from torch.optim import Adam
from torch import cuda

# user definition
from board import Board
from RandomActor import RandomActor
from Q import QFunction

print('setup envrionment...')
b = Board()
ra = RandomActor(b)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(asctime)s %(message)s')

print('setup q-func, optimizer, explorer, and agent...', end=' ')
s1 = time.perf_counter()
obs_size: int = 126
n_actions: int = 126

q_func = QFunction(obs_size, n_actions, obs_size**2)
optimizer = Adam(q_func.parameters() ,eps=1e-2)

explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.3, decay_steps=500000, random_action_func=ra.random_action_func)
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)

gpu = 1 if cuda.is_available() else -1
gamma = 0.95
agent = pfrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100, gpu=gpu
)

e1 = time.perf_counter()
print(f'completed in {round(e1-s1, 3)} sec.')

steps = 1e6
eval_interval= 1e4
eval_n_steps, eval_n_episodes = None, 10
train_max_episode_len = 200

pfrl.experiments.train_agent_with_evaluation(
    agent, b, steps=steps, eval_interval=eval_interval,
    eval_n_steps=eval_n_steps, eval_n_episodes=eval_n_episodes,
    train_max_episode_len=train_max_episode_len, outdir='./results'
)
print(f'all experiments were completed in {round((time.perf_counter()-s1) // 3600, 3)} hrs.')
