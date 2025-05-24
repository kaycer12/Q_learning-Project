# Q-learning算法实现 (Q表, 学习, 保存/加载模型)
import numpy as np
import random
import os
import pickle # 用于保存和加载Q表
import config # 导入配置文件
# from utils import state_to_index # state_to_index is used by GameEnvironment, not directly here

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, map_details=None, learning_rate=config.LEARNING_RATE, 
                 discount_factor=config.DISCOUNT_FACTOR, exploration_rate=config.EXPLORATION_RATE,
                 exploration_decay_rate=config.EXPLORATION_DECAY_RATE, min_exploration_rate=config.MIN_EXPLORATION_RATE):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        # Store map details if provided, for saving with the model
        self.map_name = map_details.get('name', 'unknown') if map_details else 'unknown'
        self.map_rows = map_details.get('rows', -1) if map_details else -1 # Assuming GameEnvironment passes these
        self.map_cols = map_details.get('cols', -1) if map_details else -1
        
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay_rate
        self.min_epsilon = min_exploration_rate
        
        self.q_table = np.zeros((state_space_size, action_space_size))
        self.initial_epsilon = exploration_rate # Store initial epsilon for potential reset if needed
        self.total_trained_episodes = 0 # Tracks total episodes trained *on this agent instance*

    def choose_action(self, state_index, is_training=True):
        if is_training and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_size)  # Explore
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            # If all Q-values for this state are zero (e.g., unvisited state or playing non-trained model),
            # choose randomly to avoid deterministic boring behavior or getting stuck.
            if np.all(self.q_table[state_index] == 0) and not is_training:
                # print(f"Warning: AI playing with no knowledge for state {state_index}, choosing random action.")
                return np.random.choice(self.action_space_size)
            return np.argmax(self.q_table[state_index])  # Exploit

    def update_q_table(self, state_index, action_index, reward, next_state_index):
        old_value = self.q_table[state_index, action_index]
        next_max = np.max(self.q_table[next_state_index]) # Q-value of the best action in the next state
        
        # Q-learning formula
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
        self.q_table[state_index, action_index] = new_value

    def update_exploration_rate(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save_model(self):
        """Saves the Q-table and agent state to a file specific to the map and episodes.
           Includes map identifiers for compatibility checks.
           Returns the filepath if successful, None otherwise.
        """
        if not os.path.exists(config.MODEL_DIR):
            try:
                os.makedirs(config.MODEL_DIR)
            except OSError as e:
                print(f"Error creating directory {config.MODEL_DIR}: {e}")
                return None

        # Construct filename: q_table_map_[map_name]_ep_[total_episodes].pkl
        # Ensure map_name is filesystem-friendly (e.g., replace spaces or invalid chars if any)
        safe_map_name = self.map_name.replace(" ", "_").replace("/", "-")
        filename = f"q_table_map_{safe_map_name}_ep_{self.total_trained_episodes}.pkl"
        filepath = os.path.join(config.MODEL_DIR, filename)

        model_data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'total_trained_episodes': self.total_trained_episodes,
            'map_name': self.map_name, # Original map name for display/reference
            'map_rows': self.map_rows, 
            'map_cols': self.map_cols,
            'state_space_size': self.state_space_size, # For compatibility check
            'action_space_size': self.action_space_size
        }
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath} (Map: '{self.map_name}', Trained Eps: {self.total_trained_episodes})")
            return filepath
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")
            return None

    def load_model(self, model_path, for_training=False):
        """Loads the Q-table and agent state from a specific model_path.
           Validates map compatibility based on state_space_size.
           Args:
               model_path (str): The full path to the model file.
               for_training (bool): If True, restores saved epsilon. Otherwise, sets epsilon to min_epsilon.
           Returns:
               bool: True if loading was successful and compatible, False otherwise.
        """
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return False
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            # --- Compatibility Check --- 
            loaded_state_space = model_data.get('state_space_size')
            loaded_map_name = model_data.get('map_name', 'unknown_loaded')
            loaded_map_rows = model_data.get('map_rows', -1)
            loaded_map_cols = model_data.get('map_cols', -1)

            if loaded_state_space != self.state_space_size:
                print(f"Model Incompatibility: Loaded model state space ({loaded_state_space}) for map '{loaded_map_name}' ({loaded_map_rows}x{loaded_map_cols}) "
                      f"does not match current agent's state space ({self.state_space_size}) for map '{self.map_name}' ({self.map_rows}x{self.map_cols}).")
                return False
            
            # Optional: Check if map names match as a warning, but dimensions are key.
            if loaded_map_name != self.map_name and self.map_name != 'unknown':
                print(f"Warning: Loading model for map '{loaded_map_name}' into agent configured for map '{self.map_name}'. Dimensions match, proceeding.")

            self.q_table = model_data.get('q_table')
            self.total_trained_episodes = model_data.get('total_trained_episodes', 0)

            if for_training:
                self.epsilon = model_data.get('epsilon', self.initial_epsilon)
                print(f"Model loaded from {os.path.basename(model_path)} for CONTINUED TRAINING. Eps: {self.total_trained_episodes}, Epsilon restored to: {self.epsilon:.4f}")
            else:
                self.epsilon = self.min_epsilon # For AI play, use minimal exploration
                print(f"Model loaded from {os.path.basename(model_path)} for PLAYING. Eps: {self.total_trained_episodes}, Epsilon set to: {self.epsilon:.4f}")
            
            # Final check for Q-table shape
            if self.q_table.shape != (self.state_space_size, self.action_space_size):
                print(f"Critical Error: Loaded Q-table SHAPE {self.q_table.shape} MISMATCH after loading {os.path.basename(model_path)}!")
                self._reset_agent_state() # Reset to a safe state
                return False

            return True
        except Exception as e:
            print(f"Error loading model from {os.path.basename(model_path)}: {e}")
            self._reset_agent_state()
            return False

    def _reset_agent_state(self):
        """Helper to reset agent to a default, untrained state."""
        print("Resetting agent to a default untrained state.")
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        self.epsilon = self.initial_epsilon
        self.total_trained_episodes = 0

    # 状态到索引的转换现在由 utils.py 中的 state_to_index 处理
    # Q表直接使用索引

    # TODO: 可能需要一个方法来将状态 (例如, (x,y) 坐标) 转换为Q表中的索引 