# Q-Learning 迷宫寻路项目

## 项目简介
本项目是一个使用 Q-learning 算法实现的智能体，用于提交国科大强化学习作业。在基于 Pygame 构建的二维迷宫环境中学习如何从起点找到终点。用户可以亲自操作智能体（玩家模式），也可以训练 AI 或加载已训练的 AI 模型进行游戏。项目支持使用预设地图、随机生成的复杂迷宫地图，以及加载上一次生成的地图。

## 技术栈
- Python 3.x
- Pygame (用于游戏界面和交互)
- NumPy (用于Q表等数值计算)

## 项目结构
```
Q_learning/
├── .git/               # Git版本控制目录
├── .gitignore          # 指定Git忽略的文件和目录
├── config.py           # 项目的配置文件，包含各种参数 (颜色、游戏设置、Q-learning参数等)
├── game_env.py         # 定义迷宫游戏环境 (地图解析、状态、动作、奖励、停滞检测)
├── main.py             # 项目主入口，处理模式选择、Pygame事件循环和整体流程控制
├── q_learning_agent.py # 实现Q-learning算法的核心逻辑 (Q表管理、动作选择、模型保存/加载)
├── utils.py            # 包含辅助函数 (状态索引转换、随机迷宫地图生成)
├── model/              # 存放训练好的Q表模型文件 (例如 q_table_map_default_ep_10000.pkl)
│   └── [示例模型文件.pkl]
└── README.md           # 本项目说明文件
```

## 核心功能
- **多种游戏模式**:
    - **玩家模式**: 用户可以通过键盘手动控制智能体在迷宫中移动。
    - **AI 训练与游玩**:
        - 选择或生成地图。
        - 输入训练轮数，从头开始训练一个新的 Q-learning 模型。
        - 训练过程中会显示智能体的学习过程（有头训练）或不显示（无头训练，速度更快）。
        - 训练完成后，AI 会使用学习到的策略进行游玩。
    - **加载并游玩上次训练的模型**: 可以直接加载上一次成功训练并保存的模型，让AI在该模型对应的地图上进行游玩。
- **地图管理**:
    - **预设地图**: `config.py` 中定义了 `SIMPLE_MAP` 和 `DEFAULT_MAP`。
    - **随机生成复杂地图**: 用户可以指定行数和列数（3-50之间），程序使用随机深度优先搜索(DFS)算法生成迷宫。
    - **加载上次生成的地图**: 如果之前生成过随机地图，可以选择加载它。
- **Q-Learning 实现**:
    - **Q表**: 使用 NumPy 数组存储，大小为 `(状态空间大小, 动作空间大小)`。
    - **状态**: 定义为智能体在迷宫中的 `(行, 列)` 位置，并转换为一维索引。
    - **动作**: 上、下、左、右。
    - **奖励机制**:
        - 到达终点: 正奖励 (对生成的复杂地图奖励更高)
        -撞到障碍物: 负奖励
        - 每走一步: 小的负奖励 (鼓励走捷径)
        - 无效移动 (撞墙): 较大的负奖励
        - **停滞惩罚**: 如果智能体在一定步数内只访问了极少数独特位置，则会受到额外惩罚，以避免其原地打转。
    - **探索与利用**: 使用 epsilon-greedy策略，`epsilon` 会随着训练轮数的增加而衰减。
- **模型持久化**:
    - 训练好的Q表以及相关的元数据（如地图名称、训练轮数、epsilon值）会以 `.pkl` 格式保存在 `model/` 目录下。
    - 模型文件名包含地图名称和训练轮数，例如 `q_table_map_default_ep_10000.pkl`。
    - 加载模型时会进行兼容性检查（基于状态空间大小）。
- **用户界面**:
    - 基于 Pygame 实现的简单图形化菜单，用于模式选择、地图选择、参数输入等。

## 如何开始

### 依赖项
1.  确保你已经安装了 Python 3.x。
2.  安装项目所需的依赖库：
    ```bash
    pip install pygame numpy
    ```
    (可以创建一个 `requirements.txt` 文件来简化这个过程: `pip freeze > requirements.txt`，然后使用 `pip install -r requirements.txt` 安装)

### 运行项目
直接运行 `main.py` 文件启动程序：
```bash
python Q_learning/main.py
```
程序启动后，会进入主菜单：

1.  **Player Mode**:
    - 选择地图。
    - 使用键盘方向键控制智能体移动。
2.  **AI Play (Train & Run)**:
    - 选择地图（或生成新地图）。
    - 输入训练轮数。
    - 选择是否进行"有头训练"（显示每一步，较慢）或"无头训练"（不显示，较快）。
    - 训练完成后，AI 会自动在该地图上进行游玩。
    - 训练好的模型会自动保存。
3.  **Play Last Trained Model**:
    - 如果之前有成功训练并保存的模型，此选项会加载该模型及其对应的地图，并让AI进行游玩。
4.  **Exit**: 退出程序。

### 配置文件 (`config.py`)
`config.py` 文件包含许多可以调整的参数，主要包括：
- **Pygame 相关**:
    - `MAX_GAME_SCREEN_WIDTH`, `MAX_GAME_SCREEN_HEIGHT`: 游戏界面的最大尺寸。
    - `FPS`: 游戏刷新率。
    - 各种颜色定义 (`WHITE`, `BLACK`, `AGENT_COLOR`, `GOAL_COLOR` 等)。
- **游戏元素符号**: `START`, `GOAL`, `OBSTACLE`, `PATH`。
- **预设地图**: `SIMPLE_MAP`, `DEFAULT_MAP`。
- **Q-Learning 参数**:
    - `LEARNING_RATE` (学习率alpha)
    - `DISCOUNT_FACTOR` (折扣因子gamma)
    - `EXPLORATION_RATE` (初始探索率epsilon)
    - `EXPLORATION_DECAY_RATE` (探索率衰减率)
    - `MIN_EXPLORATION_RATE` (最小探索率)
- **停滞惩罚机制参数**:
    - `STAGNATION_WINDOW`: 用于检测停滞的步数窗口。
    - `STAGNATION_MAX_UNIQUE_POS`: 在窗口内允许的最少独特位置数，低于此则视为停滞。
    - `STAGNATION_PENALTY`: 停滞时应用的惩罚值。
- **AI游玩模式延时**: `AI_PLAY_DELAY_BASE` 等。
- **奖励配置**: `REWARD_GOAL`, `REWARD_OBSTACLE`, `REWARD_MOVE` 等。
- **模型保存**:
    - `MODEL_DIR`: 模型保存目录 (默认为 `Q_learning/model/`)。
    - `MODEL_SAVE_INTERVAL`: 训练过程中自动保存模型的轮数间隔。
- **会话信息**: `LAST_SESSION_INFO_FILE` 用于保存上一次训练的地图和模型信息。

## 模型说明
- 模型（Q表及相关参数）保存在 `model/` 目录下，文件格式为 `.pkl`。
- 文件名通常遵循 `q_table_map_[map_name]_ep_[total_episodes].pkl` 的格式。
    - `[map_name]` 是地图的名称（例如 `default`, `simple`, 或随机生成的地图名如 `generated_10x10_random_seed_xxxx`）。
    - `[total_episodes]` 是该模型训练的总轮数。
- 加载模型时，程序会检查模型的状态空间大小是否与当前地图兼容。

## 未来工作 (可选)
- [ ] 实现更复杂的地图元素（例如传送门、不同类型的地面）。
- [ ] 尝试更高级的强化学习算法（如 SARSA, Deep Q-Network）。
- [ ] 优化超参数选择过程。
- [ ] 增强用户界面，例如提供更详细的训练统计信息。
- [ ] 允许在训练过程中调整参数。

## 贡献
欢迎对此项目做出贡献！如果你有任何建议或想要修复bug，请随时提交 Pull Request 或创建 Issue。

## 许可证
MIT License