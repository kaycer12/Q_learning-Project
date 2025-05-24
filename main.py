# 项目主入口，负责模式选择和游戏启动
import pygame
import sys
import time
import os
import numpy as np
import datetime # For timestamping during training
import pickle

import config
from game_env import GameEnvironment
from q_learning_agent import QLearningAgent
import utils

# --- Global variable to store info about the last trained session ---
last_trained_session_info = {
    "model_path": None,
    "map_layout": None,
    "map_name": None,
    "map_type": None
}

# --- Functions for saving/loading last session info ---
def save_last_session_info():
    """Saves the last_trained_session_info to a file."""
    try:
        session_file_dir = os.path.dirname(config.LAST_SESSION_INFO_FILE)
        if session_file_dir and not os.path.exists(session_file_dir):
            os.makedirs(session_file_dir)
            # print(f"Created directory for session info file: {session_file_dir}") # Silencing print for cleaner output

        with open(config.LAST_SESSION_INFO_FILE, 'wb') as f:
            pickle.dump(last_trained_session_info, f)
        # print(f"Last session info saved to {config.LAST_SESSION_INFO_FILE}") # Silencing print
    except Exception as e:
        print(f"Error saving last session info: {e}")

def load_last_session_info():
    """Loads the last_trained_session_info from a file if it exists."""
    global last_trained_session_info
    if os.path.exists(config.LAST_SESSION_INFO_FILE):
        try:
            with open(config.LAST_SESSION_INFO_FILE, 'rb') as f:
                loaded_info = pickle.load(f)
            
            # Ensure loaded_info is a dictionary and has all necessary keys
            if isinstance(loaded_info, dict) and \
               all(key in loaded_info for key in ["model_path", "map_layout", "map_name", "map_type"]):
                last_trained_session_info = loaded_info
                print(f"Last session info loaded from {config.LAST_SESSION_INFO_FILE}")
                model_path_val = last_trained_session_info.get("model_path")
                map_name_val = last_trained_session_info.get("map_name")
                if model_path_val:
                    model_basename = os.path.basename(str(model_path_val))
                    print(f"  -> Last model: {model_basename}, Map: {str(map_name_val)}")
                else:
                    print("  -> No model path found in last session.")
            else:
                print(f"Warning: Corrupted/Invalid session info file at {config.LAST_SESSION_INFO_FILE}. Starting fresh.")
                # Optionally, delete the corrupted file
                # os.remove(config.LAST_SESSION_INFO_FILE)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError, Exception) as e:
            print(f"Error loading or parsing session info from {config.LAST_SESSION_INFO_FILE}: {e}. Starting fresh.")
            # Optionally, delete the corrupted file if it caused an error during load
            # try:
            #     os.remove(config.LAST_SESSION_INFO_FILE)
            #     print(f"Removed corrupted session file: {config.LAST_SESSION_INFO_FILE}")
            # except OSError as oe:
            #     print(f"Error removing corrupted session file: {oe}")
    else:
        print(f"No last session info file found at {config.LAST_SESSION_INFO_FILE}. Starting fresh.")

# --- Pygame Setup ---
pygame.init()
pygame.font.init()
font = pygame.font.SysFont(None, 36)
small_font = pygame.font.SysFont(None, 28)
very_small_font = pygame.font.SysFont(None, 22)

# --- Helper Functions for UI ---
def draw_text(surface, text, position, color=config.BLACK, font_to_use=font, center_align=False):
    text_surface = font_to_use.render(text, True, color)
    if center_align:
        text_rect = text_surface.get_rect(center=position)
        surface.blit(text_surface, text_rect)
    else:
        surface.blit(text_surface, position)

def main_menu(screen):
    """显示主菜单并获取用户选择"""
    options = {1: "Player Mode", 2: "AI Play (Train & Run)", 3: "Play Last Trained Model", 4: "Exit"}
    selected_option = 1
    menu_active = True

    screen_width, screen_height = screen.get_size()
    menu_item_height = 40 # Adjusted for more items
    title_y = screen_height // 2 - 100 # Adjusted for more items
    start_y = screen_height // 2 - (len(options) * menu_item_height) // 2 + 20

    while menu_active:
        screen.fill(config.WHITE)
        draw_text(screen, "Q-Learning Maze Game", (screen_width // 2, title_y), config.BLUE, center_align=True)

        for key, value in options.items():
            text_color = config.RED if key == selected_option else config.BLACK
            # Use smaller font for longer menu items if necessary, though current ones are fine
            draw_text(screen, f"{key}. {value}", (screen_width // 2 - 200, start_y + (key-1) * menu_item_height), text_color, font_to_use=small_font)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: selected_option = max(1, selected_option - 1)
                elif event.key == pygame.K_DOWN: selected_option = min(len(options), selected_option + 1)
                elif event.key == pygame.K_RETURN:
                    if options[selected_option] == "Exit": pygame.quit(); sys.exit()
                    return options[selected_option]
    return None

def get_map_dimensions(screen):
    """获取用户输入的地图行数和列数"""
    rows_text = ""
    cols_text = ""
    input_active_rows = True
    input_active_cols = False
    screen_width, _ = screen.get_size()
    error_message = ""

    while input_active_rows or input_active_cols:
        screen.fill(config.WHITE)
        title_pos = (screen_width // 2, 50)
        draw_text(screen, "Enter Map Dimensions", title_pos, config.BLUE, center_align=True)
        prompt_rows = "Rows (3-50, e.g., 15): "
        prompt_cols = "Cols (3-50, e.g., 20): "
        
        row_input_pos = (50, 120)
        col_input_pos = (50, 170)
        error_pos = (50, 220)

        draw_text(screen, prompt_rows + rows_text, row_input_pos, config.BLUE if input_active_rows else config.BLACK, font_to_use=small_font)
        draw_text(screen, prompt_cols + cols_text, col_input_pos, config.BLUE if input_active_cols else config.BLACK, font_to_use=small_font)
        if error_message:
            draw_text(screen, error_message, error_pos, config.RED, font_to_use=very_small_font)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                error_message = ""
                if event.key == pygame.K_ESCAPE: return None, None # Allow escape
                if input_active_rows:
                    if event.key == pygame.K_RETURN:
                        if rows_text.isdigit() and 3 <= int(rows_text) <= 50:
                            input_active_rows = False; input_active_cols = True
                        else:
                            error_message = "Invalid rows. Must be a number between 3 and 50."; rows_text = ""
                    elif event.key == pygame.K_BACKSPACE: rows_text = rows_text[:-1]
                    elif event.unicode.isdigit(): rows_text += event.unicode
                elif input_active_cols:
                    if event.key == pygame.K_RETURN:
                        if cols_text.isdigit() and 3 <= int(cols_text) <= 50:
                            try: return int(rows_text), int(cols_text)
                            except ValueError: error_message = "Error in numbers."; cols_text = ""
                        else:
                            error_message = "Invalid cols. Must be a number between 3 and 50."; cols_text = ""
                    elif event.key == pygame.K_BACKSPACE: cols_text = cols_text[:-1]
                    elif event.unicode.isdigit(): cols_text += event.unicode
                    elif event.key == pygame.K_UP and not cols_text: input_active_cols = False; input_active_rows = True
    return None, None

def get_training_episodes(screen):
    """获取用户输入的训练轮数 (不再需要 previously_trained_episodes)"""
    input_text = ""
    input_active = True
    prompt = "Enter training episodes (e.g., 10000):"
    screen_width, _ = screen.get_size()

    while input_active:
        screen.fill(config.WHITE)
        draw_text(screen, prompt, (screen_width // 2, 100), font_to_use=small_font, center_align=True)
        draw_text(screen, input_text, (screen_width // 2, 150), config.BLUE, center_align=True)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return None 
                if event.key == pygame.K_RETURN:
                    if input_text.isdigit() and int(input_text) > 0:
                        return int(input_text)
                    else: 
                        input_text = "" # Clear invalid input
                        # Optionally add an error message on screen here
                elif event.key == pygame.K_BACKSPACE: input_text = input_text[:-1]
                elif event.unicode.isdigit(): input_text += event.unicode
    return None

def choose_map(screen, context="main_menu", last_generated_map_details=None):
    """地图选择逻辑基本不变，但AI Play的context不再需要特殊处理以排除生成选项"""
    map_options_dict = {}
    current_key = 1
    map_options_dict[current_key] = ("Simple Map", config.SIMPLE_MAP, "simple", "fixed"); current_key +=1
    map_options_dict[current_key] = ("Default Map", config.DEFAULT_MAP, "default", "fixed"); current_key +=1
    map_options_dict[current_key] = ("Generate Random Complex Map", None, "generated_new", "generated"); current_key +=1
    if last_generated_map_details:
        map_options_dict[current_key] = (f"Load Last: {last_generated_map_details['name']}", 
                                         last_generated_map_details['layout'], 
                                         last_generated_map_details['name'], 
                                         "generated"); current_key += 1
    # Removed AI training specific logic, as "AI Play" now has its own generation/selection path directly.
    # If AI needs an auto-generated map and none is last_generated, it will use defaults.
    map_options_dict[current_key] = ("Back to Main Menu", None, "back", "back"); current_key +=1

    selected_map_key = 1
    menu_active = True
    screen_width, screen_height = screen.get_size()
    menu_item_height = 35
    title_y = screen_height // 2 - 100
    start_y = screen_height // 2 - (len(map_options_dict) * menu_item_height) // 2 + 30

    while menu_active:
        screen.fill(config.WHITE)
        draw_text(screen, "Select Map", (screen_width // 2, title_y), config.BLUE, center_align=True)
        
        for key, (text, _, _, _) in map_options_dict.items():
            text_color = config.RED if key == selected_map_key else config.BLACK
            # Special handling for potentially long generated map names
            font_to_use_for_item = very_small_font if "generated_" in text.lower() or "last:" in text.lower() else small_font
            draw_text(screen, f"{key}. {text}", (screen_width // 2 - 180, start_y + (key-1)*menu_item_height), text_color, font_to_use=font_to_use_for_item)
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return None, "back", None # Allow escape to go back
                if event.key == pygame.K_UP: selected_map_key = max(1, selected_map_key - 1)
                elif event.key == pygame.K_DOWN: selected_map_key = min(len(map_options_dict), selected_map_key + 1)
                elif event.key == pygame.K_RETURN:
                    choice_text, map_layout_val, map_name_val, map_type_val = map_options_dict[selected_map_key]

                    if map_name_val == "back": return None, "back", None # Signal to go back

                    if map_name_val == "generated_new":
                        rows, cols = get_map_dimensions(screen)
                        if rows and cols:
                            try:
                                generated_layout = utils.generate_maze_map(rows, cols)
                                generated_map_name = f"random_{rows}x{cols}"
                                print(f"Generated random map: {generated_map_name} with {rows} rows, {cols} cols.")
                                return generated_layout, generated_map_name, "generated"
                            except ValueError as e: # Handle errors from generate_maze_map (e.g. too small dims)
                                screen.fill(config.WHITE)
                                draw_text(screen, f"Map Gen Error: {e}", (screen_width//2, 100), config.RED, center_align=True, font_to_use=small_font)
                                draw_text(screen, "Press any key to return.", (screen_width//2, 150), config.BLACK, center_align=True, font_to_use=small_font)
                                pygame.display.flip(); pygame.time.wait(500)
                                wait_key=True; 
                                while wait_key: 
                                    for e_k in pygame.event.get(): 
                                        if e_k.type==pygame.QUIT: pygame.quit();sys.exit()
                                        if e_k.type==pygame.KEYDOWN:wait_key=False
                                continue # Back to map selection
                        else: continue # User exited dimension input, back to map selection
                    
                    return map_layout_val, map_name_val, map_type_val
    return None, None, None

def player_mode(screen, game_env):
    """用户游玩模式"""
    screen_game = pygame.display.set_mode((game_env.screen_width, game_env.screen_height))
    pygame.display.set_caption(f"Player Mode ({game_env.map_name.capitalize()} Map)")
    game_env.reset()
    running = True
    clock = pygame.time.Clock()
    game_over_message = ""

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False; break
            if event.type == pygame.KEYDOWN and not game_over_message:
                action_idx = None
                if event.key == pygame.K_ESCAPE: running = False; break
                if event.key == pygame.K_UP: action_idx = 0
                elif event.key == pygame.K_DOWN: action_idx = 1
                elif event.key == pygame.K_LEFT: action_idx = 2
                elif event.key == pygame.K_RIGHT: action_idx = 3
                elif event.key == pygame.K_r: game_env.reset(); game_over_message = ""; action_idx = None
                
                if action_idx is not None:
                    _, reward, done, _ = game_env.step(action_idx)
                    if done:
                        if reward >= game_env.reward_goal: game_over_message = "Congratulations! You won! (R to restart)"
                        else: game_over_message = "Game Over! (R to restart)"
            elif event.type == pygame.KEYDOWN and game_over_message:
                 if event.key == pygame.K_r: game_env.reset(); game_over_message = ""
        if not running: break

        game_env.render(screen_game)
        if game_over_message:
            draw_text(screen_game, game_over_message, (game_env.screen_width // 2, game_env.screen_height // 2), config.RED, center_align=True)
        pygame.display.flip()
        clock.tick(config.FPS)

def _execute_training_for_ai_play(screen, game_env, agent, episodes_to_train, headless=False, original_menu_screen_config=None):
    """Helper function to run the training session for AI Play mode.
       Agent is modified in place. Returns True if training completed, False if interrupted.
    """
    # episodes_to_train is now passed as an argument
    caption_map_name = game_env.map_name.capitalize()
    
    agent.total_trained_episodes = 0 
    agent.epsilon = agent.initial_epsilon 

    game_screen_for_headed_train = None # Initialize
    if not headless:
        game_screen_for_headed_train = pygame.display.set_mode((game_env.screen_width, game_env.screen_height))
        pygame.display.set_caption(f"AI Training for Play ({caption_map_name} - 0/{episodes_to_train})")
    else:
        if original_menu_screen_config: 
             screen_for_headless_status = pygame.display.set_mode(original_menu_screen_config)
        else: # Fallback if no original config passed (should not happen in normal flow)
             screen_for_headless_status = screen 
        pygame.display.set_caption(f"AI Headless Training... ({caption_map_name})")
        screen_for_headless_status.fill(config.WHITE)
        draw_text(screen_for_headless_status, f"Headless Training: 0%", (screen_for_headless_status.get_width()//2, screen_for_headless_status.get_height()//2), config.BLUE, center_align=True)
        pygame.display.flip()

    clock = pygame.time.Clock()
    session_start_time = time.time()
    # Corrected f-string syntax for the print statement below
    now_str_for_log = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str_for_log}] Starting AI Training for AI Play: {episodes_to_train} episodes on '{game_env.map_name}'. Headless: {headless}")
    training_interrupted = False
    paused = False # For pause functionality

    for i in range(episodes_to_train):
        current_episode = i + 1
        
        # Handle pause
        while paused:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: training_interrupted = True; paused = False; break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: training_interrupted = True; paused = False; break
                    if event.key == pygame.K_p: paused = False # Resume
            if training_interrupted: break
            # Draw pause message on the relevant screen
            pause_screen_surface = game_screen_for_headed_train if not headless else screen_for_headless_status
            if pause_screen_surface:
                # Create a semi-transparent overlay for pause
                overlay = pygame.Surface(pause_screen_surface.get_size(), pygame.SRCALPHA)
                overlay.fill((100, 100, 100, 180)) # Dark semi-transparent
                pause_screen_surface.blit(overlay, (0,0))
                draw_text(pause_screen_surface, "PAUSED (P to Resume, Esc to Abort)", 
                          (pause_screen_surface.get_width()//2, pause_screen_surface.get_height()//2), 
                           config.WHITE, center_align=True, font_to_use=font)
                pygame.display.flip()
            clock.tick(10) # Low FPS when paused
        if training_interrupted: break

        if not headless:
             pygame.display.set_caption(f"AI Training ({caption_map_name} - Ep: {current_episode}/{episodes_to_train} | P to Pause)")
        elif headless and (current_episode % (episodes_to_train // 100 if episodes_to_train >=100 else 1) == 0 or current_episode == episodes_to_train) :
            screen_for_headless_status.fill(config.WHITE)
            progress_percent = (current_episode / episodes_to_train) * 100
            draw_text(screen_for_headless_status, f"Headless Training: {progress_percent:.0f}%", (screen_for_headless_status.get_width()//2, screen_for_headless_status.get_height()//2), config.BLUE, center_align=True)
            now_str_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            draw_text(screen_for_headless_status, f"{now_str_datetime} | Ep: {current_episode}/{episodes_to_train}", (screen_for_headless_status.get_width()//2, screen_for_headless_status.get_height()//2 + 40), config.GRAY, center_align=True, font_to_use=small_font)
            pygame.display.flip()

        current_state_idx = game_env.reset()
        done = False; episode_reward = 0; step_count = 0
        # Adjusted max_steps for potentially very large maps, ensure it's at least a certain amount
        max_steps = max(50, (game_env.rows * game_env.cols) * 2) # Increased multiplier slightly
        if game_env.map_type == "generated": max_steps = max(100, (game_env.rows * game_env.cols) * 3)

        while not done and step_count < max_steps:
            action_idx = agent.choose_action(current_state_idx, is_training=True)
            next_state_idx, reward_val, done, info = game_env.step(action_idx)
            agent.update_q_table(current_state_idx, action_idx, reward_val, next_state_idx)
            current_state_idx = next_state_idx
            episode_reward += reward_val
            step_count += 1
            if not headless:
                if step_count % config.HEADED_TRAINING_REFRESH_INTERVAL == 0 or done or step_count == max_steps:
                    game_env.render(game_screen_for_headed_train) # Use the dedicated game screen
                    info_y_start = 5
                    draw_text(game_screen_for_headed_train, f"Map: {caption_map_name}", (5,info_y_start), font_to_use=very_small_font)
                    draw_text(game_screen_for_headed_train, f"Training Ep: {current_episode}/{episodes_to_train}", (5,info_y_start+15), font_to_use=very_small_font)
                    draw_text(game_screen_for_headed_train, f"Steps: {step_count}", (5,info_y_start+30), font_to_use=very_small_font)
                    draw_text(game_screen_for_headed_train, f"Epsilon: {agent.epsilon:.4f}", (5,info_y_start+45), font_to_use=very_small_font)
                    pygame.display.flip()
            
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT: training_interrupted = True; break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: training_interrupted = True; break
                    if event.key == pygame.K_p: paused = True; break # Pause training
            if training_interrupted or paused : break # Break inner loop if interrupted or paused
        if training_interrupted: break # Break outer loop if interrupted
        
        agent.update_exploration_rate()
        agent.total_trained_episodes = current_episode 
        

        # Console logging for both headed and headless, including date & time
        log_interval = episodes_to_train // 20 if episodes_to_train >= 20 else 1 # Log ~20 times or every ep for short runs
        if (current_episode % log_interval == 0) or (current_episode == episodes_to_train) or done:

            now_datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now_datetime_str}] Map: '{game_env.map_name}' | Train Ep: {current_episode}/{episodes_to_train}, Steps: {step_count}, Reward: {episode_reward:.0f}, Epsilon: {agent.epsilon:.4f}")
            if headless: pygame.event.pump() # Keep pygame responsive for OS events
    
    session_duration = time.time() - session_start_time
    final_datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if training_interrupted:
        print(f"[{final_datetime_str}] Training for AI Play interrupted by user after {current_episode-1} episodes. Duration: {session_duration:.2f}s")
        return False
    
    print(f"[{final_datetime_str}] Finished training for AI Play: {episodes_to_train} episodes in {session_duration:.2f}s. Final Epsilon: {agent.epsilon:.4f}")
    
    # Save the model and get the path
    saved_model_path = agent.save_model()
    if saved_model_path:
        global last_trained_session_info
        last_trained_session_info["model_path"] = saved_model_path
        last_trained_session_info["map_layout"] = game_env.map_layout # Save a copy, not reference
        last_trained_session_info["map_name"] = game_env.map_name
        last_trained_session_info["map_type"] = game_env.map_type
        print(f"Last trained session info updated in memory. Model: {os.path.basename(saved_model_path)}, Map: {game_env.map_name}")
        save_last_session_info() # Persist to file
    else:
        print("Warning: Model saving failed, last_trained_session_info not updated or saved.")
        
    return True

def ai_play_mode(screen, game_env, agent, original_menu_screen_config=None, model_was_loaded=False):
    """AI游玩模式 (agent is passed in, trained in memory). User starts play.
    """
    caption_map_name = game_env.map_name.capitalize()
    screen_game = pygame.display.set_mode((game_env.screen_width, game_env.screen_height))
    play_mode_title = f"AI Play ({caption_map_name} Map - {'Loaded Global Model' if model_was_loaded else 'Trained In-Session'})"
    pygame.display.set_caption(play_mode_title)
    
    agent_info_display = f"(Epsilon: {agent.epsilon:.4f}, Trained Eps: {agent.total_trained_episodes if model_was_loaded else 'N/A'})"
    
    # Initialize clock before the waiting loop
    clock = pygame.time.Clock()

    current_state_idx = game_env.reset()
    game_env.render(screen_game) # Initial render
    draw_text(screen_game, "Press ANY KEY to start AI play... (Esc to cancel)", 
              (game_env.screen_width // 2, game_env.screen_height - 30), 
              config.BLUE, center_align=True, font_to_use=small_font)
    pygame.display.flip()

    wait_for_start = True
    user_cancelled = False
    while wait_for_start:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: user_cancelled = True; wait_for_start = False; break
                wait_for_start = False # Any other key starts
        if user_cancelled: break
        clock.tick(10) # Now clock is guaranteed to be initialized
    
    if user_cancelled:
        if original_menu_screen_config:
            screen = pygame.display.set_mode(original_menu_screen_config)
            pygame.display.set_caption("Q-Learning Game - Main Menu")
        return # Return to menu

    # If not cancelled, clock is already initialized above.
    # done = False; running = True; # clock = pygame.time.Clock() <- Moved up
    done = False; running = True
    total_reward_val = 0; steps = 0
    max_steps_play = (game_env.rows * game_env.cols) * 4 

    # Dynamic AI Play Delay
    map_size_heuristic = game_env.rows * game_env.cols
    current_ai_play_delay = int(config.AI_PLAY_DELAY_BASE - (map_size_heuristic * config.AI_PLAY_DELAY_MAP_FACTOR))
    current_ai_play_delay = max(config.MIN_AI_PLAY_DELAY, current_ai_play_delay)
    print(f"AI Play using delay: {current_ai_play_delay}ms (based on map size {map_size_heuristic})")

    while running and not done and steps < max_steps_play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False; break
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False; break
        if not running: break

        action_idx = agent.choose_action(current_state_idx, is_training=False)
        next_state_idx, reward_val, done, info = game_env.step(action_idx)
        current_state_idx = next_state_idx
        total_reward_val += reward_val
        steps += 1

        game_env.render(screen_game)
        info_y = screen_game.get_height() - 20
        if info_y < 50 : info_y = screen_game.get_height() - 40 # Adjust for small screens
        draw_text(screen_game, f"Map: {caption_map_name} {agent_info_display}", (5, info_y), config.GRAY, font_to_use=very_small_font)
        draw_text(screen_game, f"Steps: {steps}", (5,5), font_to_use=very_small_font)
        draw_text(screen_game, f"Reward: {total_reward_val:.1f}", (5,25), font_to_use=very_small_font)
        pygame.display.flip()
        pygame.time.wait(current_ai_play_delay)
        clock.tick(config.FPS)

    message = ""
    if done: message = "AI Won!" if reward_val >= game_env.reward_goal else "AI finished."
    elif steps >= max_steps_play: message = "AI reached max steps."
    else: message = "AI Play ended by user."
        
    if message:
        draw_text(screen_game, message, (game_env.screen_width // 2, game_env.screen_height // 2), 
                    config.GREEN if "Won" in message else config.RED, center_align=True)
        pygame.display.flip()
        pygame.time.wait(2000) 
    
    if original_menu_screen_config:
        screen = pygame.display.set_mode(original_menu_screen_config)
        pygame.display.set_caption("Q-Learning Game - Main Menu")

# --- Main Game Loop ---
last_generated_map_details = None 

def main_loop():
    global last_generated_map_details
    # load_last_session_info() # Moved to if __name__ == '__main__' block to be called once on startup
    menu_screen_dims = (config.MENU_SCREEN_WIDTH, config.MENU_SCREEN_HEIGHT)
    screen = pygame.display.set_mode(menu_screen_dims)
    pygame.display.set_caption("Q-Learning Game - Main Menu")

    while True:
        current_screen_for_menu = pygame.display.set_mode(menu_screen_dims) # Ensure menu screen is active
        pygame.display.set_caption("Q-Learning Game - Main Menu")
        main_menu_choice = main_menu(current_screen_for_menu)
        
        if main_menu_choice == "Exit": break
        if main_menu_choice is None: continue 

        if main_menu_choice == "Player Mode":
            current_map_layout, map_name, map_type = choose_map(current_screen_for_menu, context="main_menu", last_generated_map_details=last_generated_map_details)
            if map_name == "back" or current_map_layout is None: continue
            if map_type == "generated":
                 last_generated_map_details = {'name': map_name, 'layout': current_map_layout}
            game_env = GameEnvironment(map_layout=current_map_layout, map_name=map_name, map_type=map_type)
            player_mode(current_screen_for_menu, game_env) # Pass the current screen
        
        elif main_menu_choice == "AI Play (Train & Run)":
            current_map_layout, map_name, map_type = choose_map(current_screen_for_menu, context="ai_play", last_generated_map_details=last_generated_map_details)
            if map_name == "back" or current_map_layout is None: continue
            if map_type == "generated": 
                 last_generated_map_details = {'name': map_name, 'layout': current_map_layout}
            
            game_env = GameEnvironment(map_layout=current_map_layout, map_name=map_name, map_type=map_type)
            agent = QLearningAgent(state_space_size=game_env.state_space_size, action_space_size=game_env.action_space_size)
            
            # Get training episodes from user
            episode_choice_screen = pygame.display.set_mode((config.MENU_SCREEN_WIDTH, 250)) # Temp screen for episode input
            pygame.display.set_caption("Set Training Episodes")
            episodes_to_run = get_training_episodes(episode_choice_screen)
            if episodes_to_run is None: # User pressed Esc
                continue # Back to main menu

            # Ask for headless/headed training
            choice_screen_dims = (450, 250)
            headless_choice_screen = pygame.display.set_mode(choice_screen_dims)
            pygame.display.set_caption("AI Play Setup")
            headless_choice_screen.fill(config.WHITE)
            draw_text(headless_choice_screen, f"AI Training for: {game_env.map_name.capitalize()}", (choice_screen_dims[0]//2, 30), config.BLUE, center_align=True, font_to_use=small_font)
            draw_text(headless_choice_screen, f"Training for {episodes_to_run} episodes.", (choice_screen_dims[0]//2, 70), config.BLACK, center_align=True, font_to_use=small_font)
            draw_text(headless_choice_screen, "Y: Headless Training (faster)", (choice_screen_dims[0]//2, 120), font_to_use=small_font, center_align=True)
            draw_text(headless_choice_screen, "N: Headed Training (visual)", (choice_screen_dims[0]//2, 150), font_to_use=small_font, center_align=True)
            draw_text(headless_choice_screen, "Esc: Back to Map Select", (choice_screen_dims[0]//2, 200), config.GRAY, font_to_use=very_small_font, center_align=True)
            pygame.display.flip()
            
            headless_selected, choice_made = False, False
            waiting_for_choice = True
            while waiting_for_choice:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_y: headless_selected = True; waiting_for_choice = False; choice_made = True
                        elif event.key == pygame.K_n: headless_selected = False; waiting_for_choice = False; choice_made = True
                        elif event.key == pygame.K_ESCAPE: waiting_for_choice = False; choice_made = False
            
            if choice_made:
                # Pass the appropriate screen surface to _execute_training_for_ai_play
                # For headless, it will use original_menu_screen_config for status updates.
                # For headed, it creates its own game_screen.
                # The `screen` variable here is currently headless_choice_screen. We should pass a consistent one.
                training_completed = _execute_training_for_ai_play(current_screen_for_menu, # Use the main menu screen ref for headless status updates
                                                                 game_env, agent, episodes_to_run,
                                                                 headless=headless_selected, 
                                                                 original_menu_screen_config=menu_screen_dims)
                if training_completed:
                    # Pass model_was_loaded=False as it was just trained in this session, not loaded from a file initially by user for this mode.
                    ai_play_mode(current_screen_for_menu, game_env, agent, original_menu_screen_config=menu_screen_dims, model_was_loaded=False)

        elif main_menu_choice == "Play Last Trained Model":
            if last_trained_session_info and last_trained_session_info.get("model_path") and \
               last_trained_session_info.get("map_layout") and last_trained_session_info.get("map_name") is not None:
                
                print(f"Attempting to load last trained session: Map='{last_trained_session_info['map_name']}', Model='{os.path.basename(last_trained_session_info['model_path'])}'")
                game_env = GameEnvironment(map_layout=last_trained_session_info["map_layout"],
                                           map_name=last_trained_session_info["map_name"],
                                           map_type=last_trained_session_info["map_type"])
                
                map_details_for_agent = {
                    'name': game_env.map_name, 
                    'rows': game_env.rows, 
                    'cols': game_env.cols
                }
                agent = QLearningAgent(state_space_size=game_env.state_space_size, 
                                     action_space_size=game_env.action_space_size, 
                                     map_details=map_details_for_agent)

                if agent.load_model(model_path=last_trained_session_info["model_path"], for_training=False):
                    ai_play_mode(current_screen_for_menu, game_env, agent, original_menu_screen_config=menu_screen_dims, model_was_loaded=True)
                else:
                    # Model loading failed, display message
                    screen_msg = pygame.display.set_mode(menu_screen_dims)
                    screen_msg.fill(config.WHITE)
                    draw_text(screen_msg, f"Error loading model: {os.path.basename(last_trained_session_info['model_path'])}.", (menu_screen_dims[0]//2, menu_screen_dims[1]//2 - 30), config.RED, center_align=True, font_to_use=small_font)
                    draw_text(screen_msg, "Model might be incompatible or corrupted.", (menu_screen_dims[0]//2, menu_screen_dims[1]//2), config.RED, center_align=True, font_to_use=small_font)
                    draw_text(screen_msg, "Press any key to return to menu.", (menu_screen_dims[0]//2, menu_screen_dims[1]//2 + 40), config.GRAY, center_align=True, font_to_use=very_small_font)
                    pygame.display.flip()
                    pygame.time.wait(500)
                    wait_key_pressed=True
                    while wait_key_pressed:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                            if event.type == pygame.KEYDOWN: wait_key_pressed=False
                        pygame.time.Clock().tick(10)
            else:
                # No last trained session info, display message
                screen_msg = pygame.display.set_mode(menu_screen_dims)
                screen_msg.fill(config.WHITE)
                draw_text(screen_msg, "No last trained model found.", (menu_screen_dims[0]//2, menu_screen_dims[1]//2 - 20), config.RED, center_align=True, font_to_use=small_font)
                draw_text(screen_msg, "Please train a model using 'AI Play (Train & Run)' first.", (menu_screen_dims[0]//2, menu_screen_dims[1]//2 + 20), config.BLACK, center_align=True, font_to_use=small_font)
                draw_text(screen_msg, "Press any key to return to menu.", (menu_screen_dims[0]//2, menu_screen_dims[1]//2 + 60), config.GRAY, center_align=True, font_to_use=very_small_font)
                pygame.display.flip()
                pygame.time.wait(500)
                wait_key_pressed=True
                while wait_key_pressed:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: pygame.quit(); sys.exit()
                        if event.type == pygame.KEYDOWN: wait_key_pressed=False
                    pygame.time.Clock().tick(10)

    save_last_session_info() # Save session info before quitting
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    load_last_session_info() # Load session info at the very beginning
    main_loop() 