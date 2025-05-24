# 游戏和训练的配置文件

import pygame # pygame 虽然在这里没直接用，但颜色等常量最终是给pygame用的

# Pygame 相关配置
# SCREEN_WIDTH = 1800 # 不再需要全局定义游戏屏幕大小，由地图决定
# SCREEN_HEIGHT = 1600 # 不再需要全局定义游戏屏幕大小，由地图决定
MAX_GAME_SCREEN_WIDTH = 600 # 游戏界面本身的最大宽度
MAX_GAME_SCREEN_HEIGHT = 800 # 游戏界面本身的最大高度
SCREEN_PADDING = 20 # 屏幕边缘与地图的间距
MENU_SCREEN_WIDTH = 600 # Increased width for new button
MENU_SCREEN_HEIGHT = 350 # Increased height for new button
FPS = 30 # 游戏刷新率

# 颜色定义 (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)       # 通常用于错误或重要提示
GREEN = (0, 255, 0)     # 通常用于成功或目标
BLUE = (0, 0, 255)      # 通常用于玩家或特殊元素
GRAY = (128, 128, 128)  # 深灰色，可用作障碍物
LIGHT_GRAY = (200, 200, 200) # 浅灰色，可用作网格线
LIGHT_BLUE = (173, 216, 230) # 淡蓝色，用作起点的背景
DISABLED_COLOR = (180, 180, 180) # 用于禁用按钮的颜色

# Specific Game Element Colors
AGENT_COLOR = BLUE
GOAL_COLOR = GREEN
OBSTACLE_COLOR = BLACK
PATH_COLOR = WHITE
GRID_LINE_COLOR = LIGHT_GRAY

# 游戏元素符号 (用于定义地图布局)
START = 'S'
GOAL = 'G'
OBSTACLE = 'X'
PATH = ' '
START_GOAL_SINGLE = 'P' # For 1x1 maps where Start is Goal

# 示例地图布局
SIMPLE_MAP = [
    "S  X",
    "   X",
    " XXX",
    "   G"
]

DEFAULT_MAP = [
    "S         X",
    " XXXXXXXX X",
    " X      X X",
    " X XXXX X X",
    " X X    X X",
    " X X XXXX X",
    " X X G  X X", 
    " X X    X X",
    " X XXXXXX X",
    "          X",
    "XXXXXXXXXXX"
]

# Q-Learning 参数
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EXPLORATION_RATE = 1.0      # Epsilon: 初始探索率
EXPLORATION_DECAY_RATE = 0.9995 # Epsilon衰减率 (可以调整以适应10000轮训练)
MIN_EXPLORATION_RATE = 0.01 # 最小探索率

# 停滞惩罚机制参数 (Stagnation Detection for GameEnvironment)
STAGNATION_WINDOW = 15      # Number of recent steps to consider for stagnation
STAGNATION_MAX_UNIQUE_POS = 3 # If agent visits fewer than this many unique positions in STAGNATION_WINDOW steps
STAGNATION_PENALTY = -15    # Penalty applied if stagnation is detected 

# AI游玩模式配置
AI_PLAY_DELAY_BASE = 3 # Base delay for AI playing (ms) - Set to 300ms for ~0.3s per step
AI_PLAY_DELAY_MAP_FACTOR = 0 # Factor to reduce delay by map size (rows*cols) - Set to 0 for fixed delay
MIN_AI_PLAY_DELAY = 20 # Minimum delay (can be kept as is, or also set to 300 if strict fixed delay is always desired)
HEADED_TRAINING_REFRESH_INTERVAL = 20 # 有头训练时，每隔多少步刷新一次游戏画面

# 奖励配置
REWARD_GOAL = 100
REWARD_OBSTACLE = -100
REWARD_MOVE = -1
REWARD_INVALID_MOVE = -10
REWARD_GOAL_GENERATED_MAP = 500 # For larger, generated maps

# 地图生成默认尺寸 (用于AI游玩模式，若无上次地图且用户选择自动生成时)
DEFAULT_GENERATED_MAP_ROWS = 10
DEFAULT_GENERATED_MAP_COLS = 10

# --- Training Configuration (This seems like a remnant, AUTO_ADDITIONAL_TRAINING_EPISODES is not actively used in the current flow) ---
# AUTO_ADDITIONAL_TRAINING_EPISODES = 10000 # 自动额外训练的轮数

# 模型保存配置
MODEL_DIR = "Q_learning/model/" # 模型保存目录
MODEL_SAVE_INTERVAL = 1000  # 每隔多少轮次保存一次模型（训练过程中）
# GLOBAL_MODEL_FILENAME = "global_best_q_agent.pkl" # 这个与按地图保存策略冲突，暂时不用 

# 上一次会话信息保存文件
LAST_SESSION_INFO_FILE = "Q_learning/last_session_info.pkl" 