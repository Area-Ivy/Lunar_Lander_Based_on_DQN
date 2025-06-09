# 月球着陆器 DQN 强化学习

## 项目简介
这个项目使用深度Q网络（DQN）算法训练智能体完成OpenAI Gymnasium中的月球着陆任务。

## 项目结构
*  `/assets`
存放 `README.md` 文件所需的相关图片资源
*  `/models` 
模型文件
*  `dqn.py`
包含神经网络模型、数据集和智能体的定义
*  `train.py` 
包含训练模型的功能
* `test.py` 
包含测试训练好的模型的功能
* `Report.pptx`
实验报告 PPT

## 环境要求

```
gymnasium[box2d]
torch
numpy
tqdm
tensorboard
tensorboardX
imageio
Pillow
```

## 环境搭建

1. 安装 Anaconda，设置 Anaconda 环境变量，并在命令提示符输入 `conda --version` 查看版本信息

2. 创建 Reinforcement Learning conda 环境，Python 版本 3.9.19

   ```bash
   conda create -n rl python==3.9.19
   ```

3. 激活 Reinforcement Learning conda 环境

   ```bash
   conda activate rl
   ```

4. 安装相关依赖

    ```bash
   pip install -r requirements.txt
   ```

## 项目运行

### 训练模型

使用以下命令训练DQN模型：

```bash
python train.py --dest models --epochs 50 --episodes 50 --batch-size 512
```

参数说明：
- `--dest`: 模型保存目录（默认值：'models'）
- `--epochs`: 训练轮数（默认值：50）
- `--episodes`: 每轮训练的回合数（默认值：50）
- `--batch-size`: 批次大小（默认值：512）
- `--capacity`: 经验回放缓冲区容量（默认值：500,000）
- `--sync-rate`: 目标网络同步频率（默认值：10）

### 测试模型

使用以下命令测试训练好的模型：

```bash
python test.py --model-ckpt models/model.ckpt --episodes 1 --render
```

参数说明：
- `--model-ckpt`: 模型检查点路径（默认值：'models/model.ckpt'）
- `--episodes`: 测试回合数（默认值：1）
- `--render`: 是否渲染环境（默认不渲染）

### 生成GIF

使用以下命令生成训练好的模型执行过程的GIF：

```bash
python test.py --model-ckpt models/model.ckpt --save-gif --gif-path lunar_landing.gif
```

参数说明：
- `--save-gif`: 是否保存为GIF
- `--gif-path`: GIF保存路径（默认值：'lunar_lander.gif'）
- `--fps`: GIF帧率（默认值：30）

### 查看训练数据

  ```bash
  tensorboard --logdir=tensorboard
  ```

## 项目架构

实现包括以下主要组件：

1. **QFunc**: Q函数的神经网络架构
2. **ReplayBufferDataset**: 经验回放缓冲区的数据集类
3. **LunarLanderAgent**: 与环境交互的智能体
4. **Trainer**: 处理训练过程的类

## 训练流程

训练过程遵循以下步骤：
1. 初始化策略网络和目标网络
2. 使用ε-贪婪策略收集经验
3. 从经验回放缓冲区中提取批次数据更新Q网络
4. 定期将策略网络同步到目标网络
5. 根据验证性能保存最佳模型

## 训练结果

<p align="center">
  <img src="assets\lunar_landing_5.gif" width="19%">
  <img src="assets\lunar_landing_10.gif" width="19%">
  <img src="assets\lunar_landing_15.gif" width="19%">
  <img src="assets\lunar_landing_20.gif" width="19%">
  <img src="assets\lunar_landing_25.gif" width="19%">
  <br>5 / 10 / 15 / 20 / 25 Iterations<br>
  <img src="assets\lunar_landing_30.gif" width="19%">
  <img src="assets\lunar_landing_35.gif" width="19%">
  <img src="assets\lunar_landing_40.gif" width="19%">
  <img src="assets\lunar_landing_45.gif" width="19%">
  <img src="assets\lunar_landing_50.gif" width="19%">
  <br>30 / 35 / 40 / 45 / 50 Iterations<br>
</p>


## 模型说明

该项目使用的神经网络结构是一个简单的前馈网络，包含：
- 输入层：对应月球着陆器的8维观测空间
- 两个512单元的隐藏层，使用ReLU激活函数
- 输出层：对应4个可能的动作值（Q值） 