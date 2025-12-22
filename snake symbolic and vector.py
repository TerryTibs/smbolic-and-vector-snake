# standalone_learning_hrl_kb_agent_v31_vector_input.py
#
# VERSION 31.0 (This version): Added Vector (Grid-Based) State Representation
# - The agent now uses a dual-input system, combining the existing 1D symbolic
#   vector with a new 3D grid-based vector representation of the game state.
# - Environment (`SimpleSnakeGame`): Added `get_grid_state()` to produce a
#   (grid_size, grid_size, num_channels) tensor representing the locations of
#   the snake head, body, apple, trap, and obstacles.
# - Neural Network (`build_dqn_network`): The model architecture was completely
#   redesigned using the Keras Functional API to support two inputs:
#     1. A CNN branch to process the 3D grid for spatial feature extraction.
#     2. A Dense branch to process the 1D symbolic vector.
#     - The outputs of these branches are concatenated before the final Q-value heads.
# - Agent (`Hybrid_HRL_KB_Agent`): The replay buffer and training logic were
#   updated to store and process both state representations simultaneously.
# - This hybrid input allows the agent to combine raw visual perception with
#   high-level symbolic reasoning.

import pygame
import random
import numpy as np
from collections import deque
import time
import logging
from typing import Union, Dict, Any, List
import os
import argparse

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

# --- Basic Logging and Setup ---
log_format = '%(asctime)s - %(levelname)s - %(name)s - [%(filename)s:%(lineno)d] - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("Learning_HRL_KB_PoC")
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- GLOBAL CONSTANTS ---
APPLE_COLORS = {'red': (200, 0, 0), 'blue': (0, 0, 200), 'green': (0, 200, 0)}
TRAP_COLORS = {'orange': (255, 100, 0), 'purple': (128, 0, 128), 'cyan': (0, 200, 200)}
APPLE_COLOR_NAMES = sorted(APPLE_COLORS.keys())
TRAP_COLOR_NAMES = sorted(TRAP_COLORS.keys())
apple_color_features = [f'apple_color_{name}' for name in APPLE_COLOR_NAMES]
trap_color_features = [f'trap_color_{name}' for name in TRAP_COLOR_NAMES]
SYMBOLIC_FEATURE_NAMES = sorted([
    'apple_dx', 'apple_dy', 'trap_dx', 'trap_dy', 'apple_dist', 'trap_dist',
    'is_apple_close', 'is_trap_close', 'danger_forward', 'danger_left_rel',
    'danger_right_rel', 'tail_dx', 'tail_dy', 'snake_length_normalized',
    'is_on_edge', 'is_apple_aligned_x', 'is_apple_aligned_y',
    'apple_vector_x', 'apple_vector_y'
] + apple_color_features + trap_color_features)
SYMBOLIC_STATE_SIZE = len(SYMBOLIC_FEATURE_NAMES)
ACTION_SIZE = 4
GRID_CHANNELS = 5 # 0: snake head, 1: snake body, 2: apple, 3: trap, 4: obstacle

# --- NoisyDense Layer (Unchanged) ---
class NoisyDense(layers.Layer):
    def __init__(self, units, **kwargs): super().__init__(**kwargs); self.units = units
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        self.kernel_mu = self.add_weight(shape=(self.input_dim, self.units), initializer='he_uniform', name='kernel_mu')
        self.bias_mu = self.add_weight(shape=(self.units,), initializer='zeros', name='bias_mu')
        sigma_init = tf.constant_initializer(0.5 / np.sqrt(self.input_dim))
        self.kernel_sigma = self.add_weight(shape=(self.input_dim, self.units), initializer=sigma_init, name='kernel_sigma')
        self.bias_sigma = self.add_weight(shape=(self.units,), initializer=sigma_init, name='bias_sigma')
    def call(self, inputs):
        epsilon_in = self._f(tf.random.normal(shape=(self.input_dim, 1))); epsilon_out = self._f(tf.random.normal(shape=(1, self.units)))
        kernel_epsilon = tf.matmul(epsilon_in, epsilon_out); bias_epsilon = epsilon_out
        kernel = self.kernel_mu + self.kernel_sigma * kernel_epsilon; bias = self.bias_mu + self.bias_sigma * tf.squeeze(bias_epsilon)
        return tf.matmul(inputs, kernel) + bias
    def _f(self, x): return tf.sign(x) * tf.sqrt(tf.abs(x))

# --- PER Buffer (Unchanged) ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.0001):
        self.capacity=capacity; self.alpha=alpha; self.beta=beta; self.beta_increment=beta_increment; self.pos=0; self.buffer=[]; self.priorities=np.zeros((capacity,), dtype=np.float32); self.max_priority=1.0
    def add(self, experience):
        if len(self.buffer) < self.capacity: self.buffer.append(experience)
        else: self.buffer[self.pos] = experience
        self.priorities[self.pos] = self.max_priority; self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size):
        probs = self.priorities[:len(self.buffer)] ** self.alpha; probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]
        total = len(self.buffer); weights = (total * probs[indices]) ** (-self.beta); weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        return experiences, indices, np.array(weights, dtype=np.float32)
    def update_priorities(self, batch_indices, td_errors, epsilon=1e-5):
        priorities = np.abs(td_errors) + epsilon; self.priorities[batch_indices] = priorities
        self.max_priority = max(self.max_priority, np.max(priorities))
    def __len__(self): return len(self.buffer)

# --- Environment (Added get_grid_state) ---
class SimpleSnakeGame:
    def __init__(self, render=True, grid_size=10):
        if render: pygame.init(); self.screen=pygame.display.set_mode((grid_size*30,grid_size*30)); self.font=pygame.font.SysFont('Arial',18)
        self.render_mode=render; self.grid_size=grid_size; self.cell_size=30; self.clock=pygame.time.Clock(); self.action_map={0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}
        self.apple_colors=APPLE_COLORS; self.trap_colors=TRAP_COLORS; self.apple_color_names=APPLE_COLOR_NAMES; self.trap_color_names=TRAP_COLOR_NAMES; self.reset()
    def reset(self):
        self.snake=deque([(self.grid_size//2,self.grid_size//2),(self.grid_size//2-1,self.grid_size//2)]); self.direction=(1,0)
        self.apple_pos=self._place_item(); self.apple_color_name=random.choice(self.apple_color_names)
        self.trap_pos=self._place_item(); self.trap_color_name=random.choice(self.trap_color_names)
        self.obstacle_pos=self._place_item()
        self.score=0; self.steps=0; self.done=False; self.steps_since_last_apple=0; self.starvation_limit=self.grid_size*self.grid_size
        return self.get_symbolic_state(), self.get_grid_state() # <-- MODIFIED
    def _place_item(self):
        while True:
            pos=(random.randint(0,self.grid_size-1),random.randint(0,self.grid_size-1))
            if pos not in list(self.snake)+[getattr(self,'apple_pos',None),getattr(self,'trap_pos',None),getattr(self,'obstacle_pos',None)]: return pos
            
    def get_grid_state(self) -> np.ndarray: # <-- NEW
        """Creates a 3D grid representation of the game state."""
        grid = np.zeros((self.grid_size, self.grid_size, GRID_CHANNELS), dtype=np.float32)
        if not self.snake: return grid
        # Channel 0: Snake Head
        head_x, head_y = self.snake[0]
        grid[head_x, head_y, 0] = 1.0
        # Channel 1: Snake Body
        for part in list(self.snake)[1:]:
            grid[part[0], part[1], 1] = 1.0
        # Channel 2: Apple
        grid[self.apple_pos[0], self.apple_pos[1], 2] = 1.0
        # Channel 3: Trap
        grid[self.trap_pos[0], self.trap_pos[1], 3] = 1.0
        # Channel 4: Obstacle
        grid[self.obstacle_pos[0], self.obstacle_pos[1], 4] = 1.0
        return grid

    def get_symbolic_state(self)->Dict:
        state={}; head=self.snake[0]; state['apple_dx'],state['apple_dy']=self.apple_pos[0]-head[0],self.apple_pos[1]-head[1]; state['trap_dx'],state['trap_dy']=self.trap_pos[0]-head[0],self.trap_pos[1]-head[1]; state['apple_dist']=np.linalg.norm([state['apple_dx'],state['apple_dy']]); state['trap_dist']=np.linalg.norm([state['trap_dx'],state['trap_dy']]); state['is_apple_close']=1.0 if state['apple_dist']<(self.grid_size/3) else 0.0; state['is_trap_close']=1.0 if state['trap_dist']<2.0 else 0.0; dx,dy=self.direction; dir_forward=self.direction; dir_left_rel=(dy,-dx); dir_right_rel=(-dy,dx); state['danger_forward']=1.0 if self._is_danger((head[0]+dir_forward[0],head[1]+dir_forward[1])) else 0.0; state['danger_left_rel']=1.0 if self._is_danger((head[0]+dir_left_rel[0],head[1]+dir_left_rel[1])) else 0.0; state['danger_right_rel']=1.0 if self._is_danger((head[0]+dir_right_rel[0],head[1]+dir_right_rel[1])) else 0.0; state['tail_dx'],state['tail_dy']=(head[0]-self.snake[-1][0],head[1]-self.snake[-1][1]) if len(self.snake)>1 else (0,0); max_len=self.grid_size*self.grid_size; state['snake_length_normalized']=len(self.snake)/max_len if max_len>0 else 0.0; head_x,head_y=head; state['is_on_edge']=1.0 if (head_x==0 or head_x==self.grid_size-1 or head_y==0 or head_y==self.grid_size-1) else 0.0; state['is_apple_aligned_x'],state['is_apple_aligned_y']=(1.0 if state['apple_dx']==0 else 0.0,1.0 if state['apple_dy']==0 else 0.0); state['apple_vector_x'],state['apple_vector_y']=(state['apple_dx']/state['apple_dist'],state['apple_dy']/state['apple_dist']) if state['apple_dist']>1e-6 else (0.0,0.0)
        for name in self.apple_color_names: state[f'apple_color_{name}']=1.0 if name==self.apple_color_name else 0.0
        for name in self.trap_color_names: state[f'trap_color_{name}']=1.0 if name==self.trap_color_name else 0.0
        return state
    def _is_danger(self,pos): x,y=pos; return not(0<=x<self.grid_size and 0<=y<self.grid_size) or pos in list(self.snake)[:-1] or pos==self.trap_pos or pos==self.obstacle_pos
    def step(self,action):
        if self.done: return self.get_symbolic_state(), self.get_grid_state(), 0, True # <-- MODIFIED
        self.steps+=1; self.steps_since_last_apple+=1; reward=-0.01; new_dir=self.action_map.get(action,self.direction)
        if len(self.snake)>1 and new_dir[0]==-self.direction[0] and new_dir[1]==-self.direction[1]: new_dir=self.direction
        self.direction=new_dir; head=self.snake[0]; new_head=(head[0]+self.direction[0],head[1]+self.direction[1])
        if self.steps_since_last_apple>self.starvation_limit+len(self.snake): self.done=True; reward-=5.0
        if self._is_danger(new_head): self.done=True; reward=-10.0; return self.get_symbolic_state(), self.get_grid_state(), reward,self.done # <-- MODIFIED
        self.snake.appendleft(new_head)
        if new_head==self.apple_pos: self.score+=1; reward=10.0; self.apple_pos=self._place_item(); self.apple_color_name=random.choice(self.apple_color_names); self.trap_pos=self._place_item(); self.trap_color_name=random.choice(self.trap_color_names); self.steps_since_last_apple=0
        else: self.snake.pop()
        return self.get_symbolic_state(), self.get_grid_state(), reward, self.done # <-- MODIFIED
    def is_done(self): return self.done
    def get_direction(self): return self.direction
    def render(self):
        if not self.render_mode: return
        self.screen.fill((20,20,20)); [pygame.draw.rect(self.screen,(0,200,0),(p[0]*self.cell_size,p[1]*self.cell_size,self.cell_size,self.cell_size)) for p in self.snake]; apple_rgb=self.apple_colors[self.apple_color_name]; trap_rgb=self.trap_colors[self.trap_color_name]; pygame.draw.rect(self.screen,apple_rgb,(self.apple_pos[0]*self.cell_size,self.apple_pos[1]*self.cell_size,self.cell_size,self.cell_size)); pygame.draw.rect(self.screen,trap_rgb,(self.trap_pos[0]*self.cell_size,self.trap_pos[1]*self.cell_size,self.cell_size,self.cell_size)); pygame.draw.rect(self.screen,(100,100,100),(self.obstacle_pos[0]*self.cell_size,self.obstacle_pos[1]*self.cell_size,self.cell_size,self.cell_size)); score_text=self.font.render(f"Score: {self.score}",True,(255,255,255)); self.screen.blit(score_text,(5,5)); pygame.display.flip(); self.clock.tick(60)
    def close(self):
        if self.render_mode: pygame.quit()

# --- Classical Rule-Based Agent (Unchanged) ---
class ClassicalRuleBasedAgent:
    def __init__(self, action_map: Dict[int, tuple]):
        self.action_map = action_map
        self.action_indices = {v: k for k, v in action_map.items()}
    def get_action(self, symbolic_state: Dict, current_direction: tuple) -> int:
        dx, dy = current_direction; dir_forward = current_direction; dir_left_rel = (dy, -dx); dir_right_rel = (-dy, dx)
        action_forward = self.action_indices.get(dir_forward); action_left = self.action_indices.get(dir_left_rel); action_right = self.action_indices.get(dir_right_rel)
        if symbolic_state.get('danger_forward', 0.0) == 1.0:
            logger.debug("Classical Agent: Danger detected ahead. Turning.")
            if symbolic_state.get('danger_left_rel', 0.0) == 0.0 and action_left is not None: return action_left
            elif symbolic_state.get('danger_right_rel', 0.0) == 0.0 and action_right is not None: return action_right
            elif action_left is not None: return action_left
            elif action_right is not None: return action_right
            else: return action_forward if action_forward is not None else 0
        apple_dx = symbolic_state.get('apple_dx', 0); apple_dy = symbolic_state.get('apple_dy', 0); cross_product = dx * apple_dy - dy * apple_dx
        if cross_product > 0 and action_left is not None: return action_left
        elif cross_product < 0 and action_right is not None: return action_right
        else:
            if action_forward is not None: return action_forward
            elif action_left is not None: return action_left
            else: return action_right if action_right is not None else 0

# --- Reusable Components (KB Unchanged) ---
class SymbolicKnowledgeBase:
    def __init__(self): self.rules=[("AppleRule",lambda s:s.get('is_apple_close')and not s.get('danger_forward'),{'type':'option_bias','option_name':'GoToApple','bias_value':5.0}),("TrapRule",lambda s:s.get('is_trap_close'),{'type':'option_bias','option_name':'AvoidTrap','bias_value':10.0})]
    def reason(self,state): return [dict(con,reason=n) for n,c,con in self.rules if c(state)]

# --- Dueling Network Builder (MODIFIED for dual input) ---
def build_dqn_network(grid_shape, symbolic_size, output_size, name="DQN", learning_rate=0.00025, use_noisy=True, dueling=True):
    # --- CNN Branch for Grid Input ---
    grid_input = layers.Input(shape=grid_shape, name='grid_input')
    cnn_branch = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(grid_input)
    cnn_branch = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cnn_branch)
    cnn_branch = layers.Flatten()(cnn_branch)

    # --- Dense Branch for Symbolic Input ---
    symbolic_input = layers.Input(shape=(symbolic_size,), name='symbolic_input')
    symbolic_branch = layers.Dense(32, activation='relu')(symbolic_input)

    # --- Merged Branch ---
    merged = layers.concatenate([cnn_branch, symbolic_branch])
    x = layers.Dense(128, activation='relu')(merged)
    if use_noisy: x = NoisyDense(64)(x); x = layers.Activation('relu')(x)
    else: x = layers.Dense(64, activation='relu')(x)

    # --- Dueling Heads ---
    if dueling:
        value_head = NoisyDense(1, name='value_head')(x) if use_noisy else layers.Dense(1, name='value_head')(x)
        advantage_head = NoisyDense(output_size, name='advantage_head')(x) if use_noisy else layers.Dense(output_size, name='advantage_head')(x)
        def aggregate_streams(streams): value, advantage = streams; return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        outputs = layers.Lambda(aggregate_streams, name='combine_q_values')([value_head, advantage_head])
    else:
        outputs = NoisyDense(output_size)(x) if use_noisy else layers.Dense(output_size)(x)

    model = models.Model(inputs=[grid_input, symbolic_input], outputs=outputs, name=name)
    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# --- Main Agent (MODIFIED for dual input) ---
class Hybrid_HRL_KB_Agent:
    def __init__(self, game: SimpleSnakeGame):
        self.game = game; self.kb = SymbolicKnowledgeBase()
        self.classical_agent = ClassicalRuleBasedAgent(game.action_map)
        self.n_step = 3
        self.grid_shape = (game.grid_size, game.grid_size, GRID_CHANNELS) # <-- NEW
        self.options = self._setup_options()
        self.meta_controller = self._setup_meta_controller()
        self.active_option: Union[Dict, None] = None; self.option_start_state_vec = None; self.option_start_grid_state = None; self.option_cumulative_extrinsic_reward = 0.0
        self.target_update_counter = 0; self.target_update_freq = 100
        
    def _setup_options(self):
        options = {}
        for name in ["GoToApple", "AvoidTrap"]:
            options[name] = {
                'name': name,
                'online_network': build_dqn_network(self.grid_shape, SYMBOLIC_STATE_SIZE, ACTION_SIZE, name=f"{name}_Online"),
                'target_network': build_dqn_network(self.grid_shape, SYMBOLIC_STATE_SIZE, ACTION_SIZE, name=f"{name}_Target"),
                'replay_buffer': PrioritizedReplayBuffer(2000), 'batch_size': 32, 'n_step_buffer': deque(maxlen=self.n_step)
            }
            options[name]['target_network'].set_weights(options[name]['online_network'].get_weights())
        return options

    def _setup_meta_controller(self):
        num_options = len(self.options)
        mc = {
            "option_names": list(self.options.keys()),
            "option_indices": {name: i for i, name in enumerate(self.options.keys())},
            "online_network": build_dqn_network(self.grid_shape, SYMBOLIC_STATE_SIZE, num_options, name="Meta_Online"),
            "target_network": build_dqn_network(self.grid_shape, SYMBOLIC_STATE_SIZE, num_options, name="Meta_Target"),
            "replay_buffer": PrioritizedReplayBuffer(5000), "batch_size": 16, "gamma_meta": 0.99, 'n_step_buffer': deque(maxlen=self.n_step)
        }
        mc['target_network'].set_weights(mc['online_network'].get_weights())
        return mc
    
    def _update_target_networks(self):
        logger.debug("Updating target networks.")
        for option in self.options.values(): option['target_network'].set_weights(option['online_network'].get_weights())
        self.meta_controller['target_network'].set_weights(self.meta_controller['online_network'].get_weights())
    
    def _vectorize_symbolic_state(self, symbolic_state: dict) -> np.ndarray:
        return np.array([symbolic_state.get(key, 0.0) for key in SYMBOLIC_FEATURE_NAMES], dtype=np.float32)

    def get_action(self, symbolic_state: dict, grid_state: np.ndarray, current_direction: tuple) -> int:
        if symbolic_state.get('danger_forward') and symbolic_state.get('is_trap_close'):
            logger.info("Classical agent override: Immediate trap detected ahead. Taking evasive action.")
            return self.classical_agent.get_action(symbolic_state, current_direction)

        state_vec = self._vectorize_symbolic_state(symbolic_state)
        grid_state_exp = np.expand_dims(grid_state, axis=0)
        state_vec_exp = np.expand_dims(state_vec, axis=0)
        
        if self.active_option:
            opt_name = self.active_option['name']
            terminated = (opt_name == "GoToApple" and (not symbolic_state['is_apple_close'] or symbolic_state['is_trap_close'])) or (opt_name == "AvoidTrap" and not symbolic_state['is_trap_close'])
            if terminated:
                exp = (self.option_start_state_vec, self.option_start_grid_state, self.meta_controller['option_indices'][opt_name], self.option_cumulative_extrinsic_reward, state_vec, grid_state, self.game.is_done())
                self._process_n_step(self.meta_controller, exp, self.meta_controller['gamma_meta'])
                self.active_option = None
        if self.active_option is None:
            q_values = self.meta_controller['online_network'].predict([grid_state_exp, state_vec_exp], verbose=0)[0]
            for advice in self.kb.reason(symbolic_state):
                if advice['type'] == 'option_bias' and (opt_idx := self.meta_controller['option_indices'].get(advice['option_name'])) is not None: q_values[opt_idx] += advice['bias_value']
            idx = np.argmax(q_values)
            self.active_option = self.options[self.meta_controller['option_names'][idx]]
            self.option_start_state_vec = state_vec
            self.option_start_grid_state = grid_state
            self.option_cumulative_extrinsic_reward = 0.0
            
        return np.argmax(self.active_option['online_network'].predict([grid_state_exp, state_vec_exp], verbose=0)[0])
    
    def learn_components(self, prev_sym_state, prev_grid_state, action, reward, next_sym_state, next_grid_state, done, global_step):
        if self.active_option:
            opt = self.active_option
            intrinsic_reward=20.0 if opt['name']=="GoToApple" and reward>5 else (prev_sym_state['apple_dist']-next_sym_state['apple_dist'])*5.0 if opt['name']=="GoToApple" else ((next_sym_state['trap_dist']-prev_sym_state['trap_dist'])*2.0 if (next_sym_state['trap_dist']-prev_sym_state['trap_dist'])>0 else -5.0)
            exp = (self._vectorize_symbolic_state(prev_sym_state), prev_grid_state, action, intrinsic_reward, self._vectorize_symbolic_state(next_sym_state), next_grid_state, done)
            self._process_n_step(opt, exp, 0.95)
            self.option_cumulative_extrinsic_reward += reward
            if global_step % 4 == 0 and len(opt['replay_buffer']) >= opt['batch_size']: self._train_model(opt, 0.95)
        if done:
            for opt in self.options.values():
                while len(opt['n_step_buffer']) > 0: self._process_n_step(opt, None, 0.95, force_flush=True)
            mc = self.meta_controller
            while len(mc['n_step_buffer']) > 0: self._process_n_step(mc, None, mc['gamma_meta'], force_flush=True)
            if len(mc['replay_buffer']) >= mc['batch_size']: self._train_model(mc, mc['gamma_meta'])
        if self.target_update_counter > 0 and self.target_update_counter % self.target_update_freq == 0: self._update_target_networks()
        
    def _process_n_step(self, model_dict, new_experience, gamma, force_flush=False):
        if new_experience: model_dict['n_step_buffer'].append(new_experience)
        if len(model_dict['n_step_buffer']) >= self.n_step or (force_flush and len(model_dict['n_step_buffer']) > 0):
            R = sum([(gamma**i) * model_dict['n_step_buffer'][i][3 if 'online_network' in model_dict else 2] for i in range(len(model_dict['n_step_buffer']))]) # Adjust index for reward
            
            s_sym, s_grid, a, _, ns_sym, ns_grid, _ = model_dict['n_step_buffer'][0]
            _, _, _, _, final_ns_sym, final_ns_grid, final_done = model_dict['n_step_buffer'][-1]
            
            model_dict['replay_buffer'].add((s_sym, s_grid, a, R, final_ns_sym, final_ns_grid, final_done))
            if not force_flush: model_dict['n_step_buffer'].popleft()
            elif force_flush: model_dict['n_step_buffer'].clear()

    def _train_model(self, model_dict, gamma):
        self.target_update_counter += 1
        experiences, indices, is_weights = model_dict['replay_buffer'].sample(model_dict['batch_size'])
        s_sym, s_grid, a, r, ns_sym, ns_grid, d = map(np.array, zip(*experiences))
        
        online_net = model_dict['online_network']; target_net = model_dict['target_network']
        
        actions_from_online_net = np.argmax(online_net.predict([ns_grid, ns_sym], verbose=0), axis=1)
        q_values_from_target_net = target_net.predict([ns_grid, ns_sym], verbose=0)
        
        batch_indices = np.arange(model_dict['batch_size'])
        next_q_values = q_values_from_target_net[batch_indices, actions_from_online_net]
        targets = r + (gamma ** self.n_step) * next_q_values * (1 - d)
        
        target_f = online_net.predict([s_grid, s_sym], verbose=0); a = a.astype(int)
        td_errors = targets - target_f[batch_indices, a]
        model_dict['replay_buffer'].update_priorities(indices, td_errors)
        target_f[batch_indices, a] = targets
        
        online_net.train_on_batch([s_grid, s_sym], target_f, sample_weight=is_weights)

    def save_models(self, path="models"):
        os.makedirs(path, exist_ok=True); self.meta_controller['online_network'].save(os.path.join(path, "meta_controller_q_net.keras"))
        for opt in self.options.values(): opt['online_network'].save(os.path.join(path, f"option_{opt['name']}_policy.keras"))
        logger.info("Models saved successfully.")
    def load_models(self, path="models"):
        try:
            custom_obj = {'NoisyDense': NoisyDense}
            self.meta_controller['online_network'] = models.load_model(os.path.join(path, "meta_controller_q_net.keras"), custom_objects=custom_obj)
            for opt_name in self.options.keys(): self.options[opt_name]['online_network'] = models.load_model(os.path.join(path, f"option_{opt_name}_policy.keras"), custom_objects=custom_obj)
            self._update_target_networks(); logger.info("Models loaded successfully.")
        except Exception as e: logger.warning(f"Could not load saved models from '{path}'. This is expected if architecture changed. Starting from scratch. Error: {e}")

# --- Main Runner (Updated for dual state) ---
def main(args):
    env = SimpleSnakeGame(render=args.render, grid_size=10) # <-- Pass grid_size
    agent = Hybrid_HRL_KB_Agent(env)
    agent.load_models()
    total_scores, total_steps = [], 0
    for episode in range(1, args.episodes + 1):
        symbolic_state, grid_state = env.reset() # <-- Get both states
        agent.active_option = None; episode_reward = 0; done = False
        
        while not done:
            if env.render_mode:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: return

            current_direction = env.get_direction()
            action = agent.get_action(symbolic_state, grid_state, current_direction) # <-- Pass both states
            
            next_symbolic_state, next_grid_state, reward, done = env.step(action) # <-- Get both next_states
            total_steps += 1
            
            agent.learn_components(symbolic_state, grid_state, action, reward, next_symbolic_state, next_grid_state, done, total_steps)
            
            episode_reward += reward
            symbolic_state = next_symbolic_state
            grid_state = next_grid_state # <-- Update grid_state
            
            if env.render_mode: env.render()
            if not env.render_mode and env.steps % 100 == 0:
                print(f"\rEpisode: {episode}, Step: {env.steps}, Total Steps: {total_steps}", end="")

        if not env.render_mode: print()
        
        logger.info(f"--- Episode {episode} END --- Score: {env.score}, Steps: {env.steps}, Total Reward: {episode_reward:.2f}")
        total_scores.append(env.score)
        
        if episode % args.save_every == 0 and episode > 0: agent.save_models()
        
    env.close()
    if total_scores: logger.info(f"\n--- Training Complete ---\nAverage score over {len(total_scores)} episodes: {np.mean(total_scores):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a learning HRL-KB agent."); parser.add_argument("-r","--render",action="store_true",help="Enable graphical rendering."); parser.add_argument("-e","--episodes",type=int,default=10000,help="Total episodes to train."); parser.add_argument("-s","--save_every",type=int,default=100,help="Save models every N episodes."); cli_args = parser.parse_args(); main(cli_args)
