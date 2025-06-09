import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from copy import deepcopy
from collections import deque
from typing import Callable, Any, Union
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dqn import QFunc, Trainer, LunarLanderAgent


def main():
    parser = argparse.ArgumentParser(description='训练月球着陆器DQN模型')
    parser.add_argument('--dest', type=str, default='models', help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--episodes', type=int, default=50, help='每轮训练的回合数')
    parser.add_argument('--batch-size', type=int, default=512, help='批次大小')
    parser.add_argument('--capacity', type=int, default=500_000, help='经验回放缓冲区容量')
    parser.add_argument('--sync-rate', type=int, default=10, help='目标网络同步频率')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dest = args.dest

    # 确保保存目录存在
    if not os.path.exists(dest):
        os.makedirs(dest)

    epochs = args.epochs
    episodes = args.episodes
    batch_size = args.batch_size
    capacity = args.capacity
    sync_rate = args.sync_rate

    # 创建环境
    env_train = gym.make('LunarLander-v2')

    # 初始化策略网络和目标网络
    policy_net = QFunc(4, 8, 512, 512).to(device)
    target_net = QFunc(4, 8, 512, 512).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # 创建智能体
    agent = LunarLanderAgent(policy_net, env_train)
    
    # 设置优化器和探索策略
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    epsilon_fun: Callable[[Any], Union[int, Any]] = lambda e: max(1 - (e / epochs) * 2, 0.1)
    
    # 创建经验回放缓冲区和训练器
    replay_buffer = deque(maxlen=capacity)
    trainer = Trainer(
        policy_net=policy_net,
        target_net=target_net,
        batch_size=batch_size,
        optimizer=optimizer,
        sync_rate=sync_rate
    )
    
    # 设置TensorBoard
    writer = SummaryWriter(os.path.join('tensorboard'))

    # 训练循环
    current_best = None
    current_best_ret = -1e10

    for epoch_idx in tqdm(range(epochs), desc="训练进度"):
        # 设置当前的探索率
        curr_eps = epsilon_fun(epoch_idx)
        
        # 收集多个回合的经验
        returns = [agent.play_episode(replay_buffer, eps=curr_eps) for _ in range(episodes)]
        average_return = np.mean(returns)
        
        # 训练模型
        average_loss = trainer.run(replay_buffer)
        
        # 记录训练指标
        writer.add_scalar('train/reward', average_return, epoch_idx)
        writer.add_scalar('train/loss', average_loss, epoch_idx)
        writer.add_scalar('train/epsilon', curr_eps, epoch_idx)

        # 保存性能最好的模型
        if average_return > current_best_ret:
            current_best_ret = average_return
            current_best = deepcopy(policy_net.state_dict())

        # 验证当前模型
        with torch.no_grad():
            total_reward = agent.play_episode([], validation=True)
            writer.add_scalar('valid/reward', total_reward, epoch_idx)

        # 定期保存模型
        if epoch_idx % 5 == 0:
            if current_best is None: current_best = policy_net.state_dict()
            torch.save(current_best, os.path.join(dest, f'model_{epoch_idx}.ckpt'))
            current_best = None
            current_best_ret = -1e10

    # 保存最终模型
    torch.save(current_best, os.path.join(dest, f'model.ckpt'))
    print(f"训练完成，最终模型已保存到 {os.path.join(dest, 'model.ckpt')}")


if __name__ == "__main__":
    main()
