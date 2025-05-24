# 辅助函数 (如选择最优模型, 文件处理等)
import os
import glob
import pickle
import numpy as np
import config # For MODEL_DIR, though it's passed as an argument usually
import random # For maze generation

# TODO: 从config导入 MODEL_DIR (This is no longer relevant as MODEL_DIR is removed)

def get_best_model_path(model_dir, pattern="*.pkl"):
    """遍历模型目录，根据提供的严格模式找到最新的或轮数最高的模型文件。
    模式示例: "q_table_map_simple_ep_*.pkl"
    If no files match the exact pattern, returns None.
    NOTE: This function is no longer used in the current simplified main flow.
    """
    # if not model_dir: 
    #     print("Error: model_dir not specified in get_best_model_path.")
    #     return None
    # search_pattern = os.path.join(model_dir, pattern)
    # model_files = glob.glob(search_pattern)
    # if not model_files: return None
    # best_model_file = None
    # max_episodes = -1
    # for f_path in model_files:
    #     filename = os.path.basename(f_path)
    #     try:
    #         name_parts = filename.replace('.pkl', '').split('_')
    #         ep_found = False
    #         # Check for pattern like ..._ep_NUM.pkl
    #         if len(name_parts) >= 3 and name_parts[-2] == "ep" and name_parts[-1].isdigit():
    #             episodes = int(name_parts[-1])
    #             ep_found = True
    #         # Check for pattern like ..._interrupted_ep_NUM.pkl
    #         elif len(name_parts) >= 4 and name_parts[-3] == "interrupted" and name_parts[-2] == "ep" and name_parts[-1].isdigit():
    #             episodes = int(name_parts[-1])
    #             ep_found = True
    #         
    #         if ep_found:
    #             if episodes > max_episodes:
    #                 max_episodes = episodes
    #                 best_model_file = f_path
    #             # If current file is 'interrupted' and has same episodes as a non-interrupted one, prefer non-interrupted.
    #             elif episodes == max_episodes and best_model_file and \
    #                  "interrupted" in filename and "interrupted" not in os.path.basename(best_model_file):
    #                 pass # Keep existing best_model_file (non-interrupted)
    #             # If current file is not interrupted and has same episodes as an 'interrupted' one, prefer current.
    #             elif episodes == max_episodes and best_model_file and \
    #                  "interrupted" not in filename and "interrupted" in os.path.basename(best_model_file):
    #                 best_model_file = f_path # Prefer non-interrupted
    #             elif episodes == max_episodes:
    #                 if best_model_file is None: 
    #                     best_model_file = f_path
    #                 pass 
    # 
    #     except (ValueError, IndexError) as e:
    #         continue 
    # 
    # if best_model_file:
    #     return best_model_file
    # elif model_files: 
    #     return max(model_files, key=os.path.getmtime)
    # else:
    #     return None
    print(f"utils.get_best_model_path('{model_dir}', '{pattern}') called, but model loading is disabled.")
    return None

def state_to_index(state, map_cols):
    """将 (row, col) 状态转换为Q表的一维索引。"""
    if not isinstance(state, (list, tuple)) or len(state) != 2:
        raise ValueError(f"State must be (row, col) tuple, got: {state}")
    row, col = state
    return row * map_cols + col

def index_to_state(index, map_cols):
    """将Q表的一维索引转换回 (row, col) 状态。"""
    row = index // map_cols
    col = index % map_cols
    return (row, col)

def generate_maze_map(rows, cols):
    """使用随机深度优先搜索(DFS)和回溯生成一个迷宫地图。
    确保起点(S)在(0,0)，终点(G)在(rows-1, cols-1)。
    返回一个字符串列表，代表地图布局 ('S', 'G', 'X', ' ').
    """
    if rows < 1 or cols < 1: # Basic validation
        raise ValueError("Map dimensions must be at least 1x1.")

    if rows == 1:
        if cols == 1: return [config.START_GOAL_SINGLE]
        return [config.START + config.PATH * (cols - 2) + config.GOAL if cols > 1 else config.START_GOAL_SINGLE]
    if cols == 1:
        map_list = [[config.START]] + [[config.PATH] for _ in range(rows - 2)] + [[config.GOAL] if rows > 1 else []]
        if rows == 1: map_list = [[config.START_GOAL_SINGLE]]
        elif rows > 1 and not map_list[-1]: map_list.pop(); map_list.append([config.GOAL])
        return ["".join(r) for r in map_list]
    
    maze = [[config.OBSTACLE for _ in range(cols)] for _ in range(rows)]
    stack = [] 

    def is_valid(r, c):
        return 0 <= r < rows and 0 <= c < cols

    start_r, start_c = 0, 0
    maze[start_r][start_c] = config.PATH 
    stack.append((start_r, start_c))

    while stack:
        curr_r, curr_c = stack[-1]
        neighbors = []
        possible_moves = [(-2, 0, -1, 0), (2, 0, 1, 0), (0, -2, 0, -1), (0, 2, 0, 1)]
        random.shuffle(possible_moves)

        for dr_n, dc_n, dr_w, dc_w in possible_moves:
            next_r, next_c = curr_r + dr_n, curr_c + dc_n
            wall_r, wall_c = curr_r + dr_w, curr_c + dc_w
            if is_valid(next_r, next_c) and maze[next_r][next_c] == config.OBSTACLE:
                neighbors.append((next_r, next_c, wall_r, wall_c))
        
        if neighbors:
            n_r, n_c, w_r, w_c = random.choice(neighbors)
            maze[w_r][w_c] = config.PATH
            maze[n_r][n_c] = config.PATH
            stack.append((n_r, n_c))
        else:
            stack.pop()

    # --- BEGIN FIX for unreachable G --- 
    goal_r, goal_c = rows - 1, cols - 1
    if maze[goal_r][goal_c] == config.OBSTACLE:
        maze[goal_r][goal_c] = config.PATH # Force G cell to be a path
        # print(f"Debug: Goal cell ({goal_r},{goal_c}) was OBSTACLE, forced to PATH.")

        # Check if this newly pathed G cell is connected to any existing path
        is_g_connected = False
        for dr_g, dc_g in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr_g, nc_g = goal_r + dr_g, goal_c + dc_g
            if is_valid(nr_g, nc_g) and maze[nr_g][nc_g] == config.PATH:
                is_g_connected = True
                break
        
        if not is_g_connected and (rows > 1 or cols > 1): # If G is now path but isolated
            # print(f"Debug: Forced-path G cell ({goal_r},{goal_c}) is isolated. Attempting to connect.")
            # Attempt to connect it to one of its OBSTACLE neighbors by making that neighbor a PATH too.
            # Prioritize N, W, S, E for connection attempt for predictability, but shuffle for variety.
            connection_attempts = []
            if goal_r > 0: connection_attempts.append((goal_r - 1, goal_c))  # North
            if goal_c > 0: connection_attempts.append((goal_r, goal_c - 1))  # West
            if goal_r < rows - 2: connection_attempts.append((goal_r + 1, goal_c)) # South (rows-2 because G is rows-1)
            if goal_c < cols - 2: connection_attempts.append((goal_r, goal_c + 1)) # East  (cols-2 because G is cols-1)
            
            random.shuffle(connection_attempts)

            for conn_r, conn_c in connection_attempts:
                if is_valid(conn_r, conn_c) and maze[conn_r][conn_c] == config.OBSTACLE:
                    maze[conn_r][conn_c] = config.PATH
                    # print(f"Debug: Connected isolated G by making neighbor ({conn_r},{conn_c}) PATH.")
                    break # Made one connection
    # --- END FIX for unreachable G --- 

    maze[0][0] = config.START
    maze[rows - 1][cols - 1] = config.GOAL

    map_str_list = ["".join(row_list) for row_list in maze]
    return map_str_list

# Example usage (for testing):
if __name__ == '__main__':
    class MockConfig:
        OBSTACLE = 'X'
        PATH = ' '
        START = 'S'
        GOAL = 'G'
        START_GOAL_SINGLE = 'P'
        MODEL_DIR = "./models_test" 
    config.OBSTACLE = MockConfig.OBSTACLE
    config.PATH = MockConfig.PATH
    config.START = MockConfig.START
    config.GOAL = MockConfig.GOAL
    config.START_GOAL_SINGLE = MockConfig.START_GOAL_SINGLE
    
    test_mazes = {
        "5x5": (5,5),
        "10x15": (10,15),
        "3x3": (3,3),
        "2x2": (2,2),
        "4x4": (4,4),
        "2x5": (2,5),
        "5x2": (5,2),
        "1x5": (1,5),
        "5x1": (5,1),
        "1x1": (1,1)
    }

    for name, (r,c) in test_mazes.items():
        print(f"\nGenerating {name} maze ({r}x{c}):")
        try:
            maze_layout = generate_maze_map(r,c)
            for row_str in maze_layout:
                print(row_str)
            # Basic check: S is at (0,0), G is at (r-1,c-1)
            if not (maze_layout[0][0] == config.START or (r==1 and c==1 and maze_layout[0][0] == config.START_GOAL_SINGLE)):
                print(f"ERROR: Start S not at (0,0) for {name}!")
            if not (maze_layout[r-1][c-1] == config.GOAL or (r==1 and c==1 and maze_layout[0][0] == config.START_GOAL_SINGLE)):
                 print(f"ERROR: Goal G not at ({r-1},{c-1}) for {name}!")
            if r > 1 and c > 1 and maze_layout[r-1][c-1] == config.OBSTACLE:
                 print(f"ERROR: Goal G is an OBSTACLE for {name} after generation attempt!")

        except ValueError as e:
            print(f"Error generating {name}: {e}")

# 你可能还需要其他辅助函数，例如：
# - 加载地图文件
# - 简单的GUI元素 (如按钮、文本显示)
# - 性能评估函数 (用于AI游玩模式选择最优模型时参考) 