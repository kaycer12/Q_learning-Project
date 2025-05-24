# 定义游戏环境 (地图, 状态, 动作, 奖励)
import numpy as np
import pygame
import config
from utils import state_to_index # 我们会用到这个函数
from collections import deque # 用于高效处理 recent_positions

# TODO: 从config导入配置

class GameEnvironment:
    def __init__(self, map_layout, map_name="custom", map_type="fixed"):
        # 地图和显示配置
        self.map_layout = map_layout
        self.map_name = map_name
        self.map_type = map_type # "fixed" or "generated"
        self.rows = len(self.map_layout)
        self.cols = 0
        if self.rows > 0:
            self.cols = max(len(row) for row in self.map_layout) if self.map_layout else 0
            # Pad shorter rows to make the map rectangular internally
            self.map_layout = [row.ljust(self.cols, config.PATH) for row in self.map_layout]
        else:
            self.rows = 0 # Should not happen with valid maps
            self.cols = 0 # Ensure cols is also 0 if rows is 0

        # 动态计算 tile_size 和屏幕尺寸
        if self.rows > 0 and self.cols > 0:
            # 使用 MAX_GAME_SCREEN_WIDTH/HEIGHT 作为游戏画面的最大可用空间
            available_width = config.MAX_GAME_SCREEN_WIDTH - 2 * config.SCREEN_PADDING
            available_height = config.MAX_GAME_SCREEN_HEIGHT - 2 * config.SCREEN_PADDING
            
            tile_size_w = available_width // self.cols
            tile_size_h = available_height // self.rows
            self.tile_size = max(1, min(tile_size_w, tile_size_h)) # Ensure tile_size is at least 1
        else:
            self.tile_size = 20 # Default for empty/invalid map to avoid division by zero

        self.screen_width = self.cols * self.tile_size + 2 * config.SCREEN_PADDING
        self.screen_height = self.rows * self.tile_size + 2 * config.SCREEN_PADDING

        # 解析地图
        self.start_pos = None
        self.goal_pos = None
        self.obstacles = []
        self._parse_map()

        if self.start_pos is None:
            raise ValueError("Map is missing a Start 'S' position")
        if self.goal_pos is None:
            raise ValueError("Map is missing a Goal 'G' position")

        self.agent_pos = self.start_pos

        # 动作空间: 0: 上, 1: 下, 2: 左, 3: 右
        self.action_space = [0, 1, 2, 3]  # 0:Up, 1:Down, 2:Left, 3:Right
        self.action_space_size = len(self.action_space)

        # 状态空间大小
        self.state_space_size = self.rows * self.cols

        # 奖励定义
        if self.map_type == "generated":
            self.reward_goal = config.REWARD_GOAL_GENERATED_MAP
            print(f"Using generated map goal reward: {self.reward_goal}")
        elif self.map_name == "simple": # Example: different reward for simple map
            self.reward_goal = config.REWARD_GOAL # Or a specific one like config.REWARD_GOAL_SIMPLE
        else:
            self.reward_goal = config.REWARD_GOAL
        
        self.reward_obstacle = config.REWARD_OBSTACLE
        self.reward_move = config.REWARD_MOVE
        self.reward_invalid_move = config.REWARD_INVALID_MOVE

        # 停滞检测相关
        self.stagnation_window = config.STAGNATION_WINDOW
        self.stagnation_max_unique_pos = config.STAGNATION_MAX_UNIQUE_POS
        self.stagnation_penalty = config.STAGNATION_PENALTY
        self.recent_positions = deque(maxlen=self.stagnation_window)
        self.stagnation_applied_this_step = False

    def _parse_map(self):
        for r, row_str in enumerate(self.map_layout):
            for c, char in enumerate(row_str):
                if char == config.START:
                    self.start_pos = (r, c)
                    self.agent_pos = self.start_pos
                elif char == config.GOAL:
                    self.goal_pos = (r, c)
                elif char == config.START_GOAL_SINGLE: # Handle P for 1x1 map
                    self.start_pos = (r,c)
                    self.goal_pos = (r,c)
                    self.agent_pos = self.start_pos
                elif char == config.OBSTACLE:
                    self.obstacles.append((r, c))
        if self.start_pos is None:
            # Fallback if S is missing, place at first available path or (0,0)
            print("Warning: Start position 'S' not found in map. Defaulting to first path or (0,0).")
            for r, row_str in enumerate(self.map_layout):
                for c, char in enumerate(row_str):
                    if char == config.PATH:
                        self.start_pos = (r,c); self.agent_pos=(r,c); break
                if self.start_pos: break
            if self.start_pos is None and self.rows > 0 and self.cols > 0 : self.start_pos = (0,0); self.agent_pos=(0,0); self.map_layout[0] = config.START + self.map_layout[0][1:]
        
        if self.goal_pos is None and not (self.start_pos and self.map_layout[self.start_pos[0]][self.start_pos[1]] == config.START_GOAL_SINGLE) :
            print("Warning: Goal position 'G' not found in map. Defaulting to last path or (rows-1,cols-1).")
            for r in range(self.rows -1, -1, -1):
                for c in range(self.cols -1, -1, -1):
                    if self.map_layout[r][c] == config.PATH:
                        self.goal_pos = (r,c); break
                if self.goal_pos: break
            if self.goal_pos is None and self.rows > 0 and self.cols > 0: self.goal_pos = (self.rows-1, self.cols-1); self.map_layout[self.rows-1] = self.map_layout[self.rows-1][:self.cols-1] + config.GOAL

    def reset(self):
        """重置环境到初始状态，返回初始状态的索引。"""
        self.agent_pos = self.start_pos
        self.recent_positions.clear()
        self.recent_positions.append(self.agent_pos)
        self.stagnation_applied_this_step = False
        return state_to_index(self.agent_pos, self.cols)

    def step(self, action_idx):
        """执行一个动作，返回 (next_state_index, reward, done, info)。"""
        self.stagnation_applied_this_step = False # Reset for this step
        current_r, current_c = self.agent_pos
        next_r, next_c = current_r, current_c

        if action_idx == 0:  # Up
            next_r -= 1
        elif action_idx == 1:  # Down
            next_r += 1
        elif action_idx == 2:  # Left
            next_c -= 1
        elif action_idx == 3:  # Right
            next_c += 1
        
        reward = self.reward_move # Default reward for a valid move
        done = False

        # Check boundaries
        if not (0 <= next_r < self.rows and 0 <= next_c < self.cols):
            reward = self.reward_invalid_move
            # Agent stays in the same position
            next_r, next_c = current_r, current_c
        else:
            # Check for obstacles
            cell_type = self.map_layout[next_r][next_c]
            if cell_type == config.OBSTACLE:
                reward = self.reward_obstacle # Higher penalty for hitting obstacle
                # Agent stays in the same position
                next_r, next_c = current_r, current_c
            else:
                self.agent_pos = (next_r, next_c) # Move agent
                if self.agent_pos == self.goal_pos:
                    reward = self.reward_goal
                    done = True
                # For START_GOAL_SINGLE maps, if agent is on S/P and it's also G
                elif cell_type == config.START_GOAL_SINGLE and self.start_pos == self.goal_pos:
                    reward = self.reward_goal
                    done = True

        # Stagnation detection (only if not done)
        if not done:
            self.recent_positions.append(self.agent_pos) # Add current (possibly new) position
            if len(self.recent_positions) == self.stagnation_window:
                unique_pos_count = len(set(self.recent_positions))
                if unique_pos_count <= self.stagnation_max_unique_pos:
                    reward += self.stagnation_penalty # Apply stagnation penalty
                    self.stagnation_applied_this_step = True
                    # print(f"Stagnation detected! Positions in window: {list(self.recent_positions)}, Unique: {unique_pos_count}. Penalty {self.stagnation_penalty} applied.")

        next_state_idx = state_to_index(self.agent_pos, self.cols)
        info = {'stagnation_applied': self.stagnation_applied_this_step}
        return next_state_idx, reward, done, info

    def render(self, screen):
        """在Pygame窗口中绘制游戏当前状态。"""
        screen.fill(config.PATH_COLOR)
        # Calculate the starting offset for the map grid due to padding
        map_offset_x = config.SCREEN_PADDING
        map_offset_y = config.SCREEN_PADDING

        for r in range(self.rows):
            for c in range(self.cols):
                rect = pygame.Rect(map_offset_x + c * self.tile_size, 
                                    map_offset_y + r * self.tile_size, 
                                    self.tile_size, self.tile_size)
                cell_char = self.map_layout[r][c]
                cell_color = config.PATH_COLOR 
                is_agent_pos = (self.agent_pos == (r,c))

                if cell_char == config.OBSTACLE:
                    cell_color = config.OBSTACLE_COLOR
                elif cell_char == config.GOAL or ((r,c) == self.goal_pos): # Check goal_pos as well for P maps
                    cell_color = config.GOAL_COLOR
                elif cell_char == config.START or ((r,c) == self.start_pos and not is_agent_pos) : # Don't overdraw agent if S is also agent_pos
                    cell_color = config.LIGHT_BLUE # Color for start, if not also goal or obstacle
                elif cell_char == config.START_GOAL_SINGLE: # P case
                    cell_color = config.GOAL_COLOR # Treat P like Goal for rendering if agent not on it
                
                pygame.draw.rect(screen, cell_color, rect)
                pygame.draw.rect(screen, config.GRID_LINE_COLOR, rect, 1) # Draw grid lines

                if is_agent_pos:
                    agent_center_x = map_offset_x + c * self.tile_size + self.tile_size // 2
                    agent_center_y = map_offset_y + r * self.tile_size + self.tile_size // 2
                    pygame.draw.circle(screen, config.AGENT_COLOR, (agent_center_x, agent_center_y), self.tile_size // 3)
        pygame.display.flip() # 更新整个屏幕

    def get_state_index(self):
        """获取当前智能体位置对应的状态索引"""
        return state_to_index(self.agent_pos, self.cols)

    # 你可以添加一个关闭Pygame窗口的方法，如果Pygame在GameEnvironment内部初始化的话
    # def close(self):
    #     pygame.quit()

# 示例用法 (用于测试 GameEnvironment)
if __name__ == '__main__':
    pygame.init()
    
    # 使用 config 中的 SIMPLE_MAP
    env = GameEnvironment(map_layout=config.SIMPLE_MAP, map_name="simple_test")
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption(f"Game Environment Test - {env.map_name} ({env.rows}x{env.cols})")

    running = True
    clock = pygame.time.Clock()
    env.reset()
    env.render(screen)

    print(f"地图大小: {env.rows}x{env.cols}")
    print(f"起点: {env.start_pos}, 终点: {env.goal_pos}")
    print(f"障碍物: {env.obstacles}")
    print(f"当前智能体位置: {env.agent_pos}, 状态索引: {env.get_state_index()}")
    print(f"状态空间大小: {env.state_space_size}, 动作空间大小: {env.action_space_size}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                
                if action is not None:
                    next_state_idx, reward, done, info = env.step(action)
                    env.render(screen)
                    print(f"动作: {action}, 下一状态索引: {next_state_idx}, 奖励: {reward}, 是否结束: {done}")
                    print(f"智能体新位置: {env.agent_pos}")
                    if done:
                        print("游戏结束!")
                        # 可以选择重置游戏 env.reset()
                        # env.render(screen)

        clock.tick(config.FPS)
    
    pygame.quit() 