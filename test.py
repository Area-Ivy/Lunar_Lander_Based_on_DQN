import os
import argparse
import torch
import gymnasium as gym
import numpy as np
from PIL import Image
import imageio
from dqn import QFunc, LunarLanderAgent


def main():
    parser = argparse.ArgumentParser(description='测试月球着陆器DQN模型')
    parser.add_argument('--model-ckpt', type=str, default='models/model.ckpt', help='模型检查点路径')
    parser.add_argument('--episodes', type=int, default=1, help='测试回合数')
    parser.add_argument('--render', action='store_true', help='是否渲染环境')
    parser.add_argument('--save-gif', action='store_true', help='是否保存为GIF')
    parser.add_argument('--gif-path', type=str, default='lunar_lander.gif', help='GIF保存路径')
    parser.add_argument('--fps', type=int, default=30, help='GIF帧率')
    args = parser.parse_args()

    episodes = args.episodes
    model_checkpoints = args.model_ckpt
    render_mode = 'rgb_array' if args.save_gif else ('human' if args.render else None)

    # 检查模型文件是否存在
    if not os.path.exists(model_checkpoints):
        print(f"错误：模型文件 '{model_checkpoints}' 不存在")
        return

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 创建环境
    env = gym.make('LunarLander-v2', render_mode=render_mode)

    # 加载模型
    print(f"加载模型: {model_checkpoints}")
    qfunc = QFunc(4, 8, 512, 512).to(device)
    qfunc.load_state_dict(torch.load(model_checkpoints, map_location=device))
    
    # 创建智能体
    agent = LunarLanderAgent(qfunc, env)

    # 运行测试回合
    total_rewards = []
    
    for episode in range(episodes):
        frames = []
        reward = play_and_record_episode(agent, frames) if args.save_gif else agent.play_episode([], validation=True)
        total_rewards.append(reward)
        print(f"回合 {episode+1}/{episodes}, 奖励: {reward:.2f}")
        
        # 保存GIF
        if args.save_gif and frames:
            gif_path = f"{os.path.splitext(args.gif_path)[0]}_{episode+1}.gif" if episodes > 1 else args.gif_path
            print(f"保存GIF到: {gif_path}")
            imageio.mimsave(gif_path, frames, fps=args.fps)
    
    # 输出平均奖励
    if episodes > 1:
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"平均奖励: {avg_reward:.2f}")


def play_and_record_episode(agent, frames):
    """播放一个回合并记录帧"""
    total_reward = 0
    curr_observations, _ = agent.env.reset()
    
    # 记录初始帧
    frame = agent.env.render()
    frames.append(np.array(frame))
    
    while True:
        action = agent.get_action(curr_observations, validation=True, eps=0)
        next_observations, reward, terminated, truncated, _ = agent.env.step(action)
        
        # 记录帧
        frame = agent.env.render()
        frames.append(np.array(frame))
        
        done = terminated or truncated
        total_reward += reward
        
        if done:
            break
        curr_observations = next_observations
    
    return total_reward


if __name__ == "__main__":
    main()
