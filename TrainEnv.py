#!/usr/bin/env python3
"""
Deep Reinforcement Learning with StarCoder for Code Generation
=============================================================

This module implements the PromptsEnv environment for training the CodeCloak agent
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard Library Imports
import gc
import logging
import time

# Third-Party Imports
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# Reinforcement Learning Imports
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO

# Transformer and NLP Imports
from transformers import AutoTokenizer, AutoModel
from transformers import logging as transformers_logging

# Code Analysis Imports
from codebleu import calc_codebleu

# Local Imports
from CodeManipulations import *

from DataSet import *
from Normalizations import NormalizeObservation_3, NormalizeMultipleRewards
from utils import *
from SendRequestLLM import generate_response
from EvalEnv import ValPromptsEnv


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Set the logging level for the transformers library
transformers_logging.set_verbosity_error()  # Show only errors
logging.getLogger("transformers").setLevel(logging.ERROR)  # Again, show only errors
logging.getLogger().setLevel(logging.ERROR)  # Disable all warnings from the root logger

# =============================================================================
# CONSTANTS AND GLOBAL CONFIGURATION
# =============================================================================

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special Tokens
MASK_TOKEN = "<mask>"
SEPARATOR_TOKEN = "<sep>"
PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"

# Dataset Paths
VAL_DATASET_PATH = "PATH/TO/VALIDATION/DATASET"
TRAIN_DATASET_PATH = "PATH/TO/TRAIN/DATASET"

# Model Configuration
EMBEDDING_SIZE = 768
MAX_SEQUENCE_LENGTH = 1024
MAX_STEPS_PER_EPISODE = 15
NUM_ACTIONS = 12

# =============================================================================
# DATASET INITIALIZATION
# =============================================================================

val_data_set = PromptsDataset(VAL_DATASET_PATH)
prompts_data_set = PromptsDataset(TRAIN_DATASET_PATH)

# =============================================================================
# CODE MANIPULATION ACTIONS
# =============================================================================

# Define action lists for different code segments
list_of_changes_segment_0 = [
    detect_and_replace_pii_segment_0,
    change_random_lines_segment_0,
    delete_random_line_segment_0,
    insert_random_line_segment_0,
    del_function_body_special_tokens_incremental_segment_0,
    del_all_except_last_function_body_special_tokens_segment_0,
    del_all_function_body_special_tokens_segment_0,
    del_functions_incremental_segment_0,
    change_all_function_names_final_final_segment_0,
    change_all_variables_names_final_segment_0,
    change_all_argument_names_final_segment_0,
    stop_changes_segment_0
]

list_of_changes_segment_1 = [
    detect_and_replace_pii_segment_1,
    change_random_lines_segment_1,
    delete_random_line_segment_1,
    insert_random_line_segment_1,
    del_function_body_special_tokens_incremental_segment_1,
    del_all_except_last_function_body_special_tokens_segment_1,
    del_all_function_body_special_tokens_segment_1,
    del_functions_incremental_segment_1,
    change_all_function_names_final_final_segment_1,
    change_all_variables_names_final_segment_1,
    change_all_argument_names_final_segment_1,
    stop_changes_segment_1
]

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

# Initialize tokenizer and model
tokenizer = prepare_tokenizer("bigcode/starencoder", PAD_TOKEN, SEPARATOR_TOKEN, CLS_TOKEN, MASK_TOKEN)
model = AutoModel.from_pretrained("bigcode/starencoder").to(device)


# =============================================================================
# MAIN ENVIRONMENT CLASS
# =============================================================================

class PromptsEnv(gym.Env):
    """
    Custom Environment for Code Manipulation using Reinforcement Learning

    This environment allows an agent to iteratively modify code prompts to
    improve the quality of generated code responses. The agent can apply
    various transformations to code segments and receives rewards based on
    the similarity between original and generated responses.
    """

    def __init__(self, prompts_data_set, embedding_size=EMBEDDING_SIZE):
        """Initialize the environment with dataset and configuration."""
        super(PromptsEnv, self).__init__()

        # Environment Configuration
        self.prompts_data_set = prompts_data_set
        self.state_size = embedding_size + 1 + 1 + 1  # embedding + segment + scores + cursor
        self.gc_counter = 0
        self.max_steps = MAX_STEPS_PER_EPISODE

        # Action and Observation Spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1 * np.inf, high=np.inf,
            shape=(self.state_size,), dtype=np.float32
        )

        # Environment State Variables
        self._reset_episode_variables()
        self._initialize_sample()

        # Action mappings
        self.changes_0 = list_of_changes_segment_0
        self.changes_1 = list_of_changes_segment_1

    def _reset_episode_variables(self):
        """Reset variables that change during an episode."""
        self.current_segment = 0
        self.terminated = False
        self.truncated = False
        self.counter_steps = 0
        self.flag = 0
        self.reward = [0, 0]
        self.last_reward = [0, 0]
        self.name_map = {}

        # Metric tracking - only CodeBLEU
        self.code_bleu_request = 0
        self.code_bleu_response = 0

        # Response tracking
        self.convert_back_manipulated = ""
        self.origin_response = ""
        self.vec_response = None

    def _initialize_sample(self):
        """Initialize a new sample from the dataset."""
        self.current_sample = self.prompts_data_set.get_random_sample()
        self.original_combined = self.current_sample[0]['truncated_combined']
        self.manipulated_combined = self.current_sample[0]['truncated_combined']
        self.current_segment_0 = self.current_sample[0]['segment_0']
        self.current_segment_1 = self.current_sample[0]['segment_1']

        # Get embeddings
        self.current_embedding_segment_0 = get_code_embedding(
            self.current_sample[0]['segment_0'], tokenizer, model, device, CLS_TOKEN, SEPARATOR_TOKEN,
            MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE
        )
        self.current_embedding_segment_1 = get_code_embedding(
            self.current_sample[0]['segment_1'], tokenizer, model, device, CLS_TOKEN, SEPARATOR_TOKEN,
            MAX_SEQUENCE_LENGTH, EMBEDDING_SIZE
        )
        self.original_code_embedding_segment_0 = self.current_embedding_segment_0.copy()
        self.original_code_embedding_segment_1 = self.current_embedding_segment_1.copy()

        # Initialize state
        self.vec_combined = get_code_embedding(
            self.current_segment_0, tokenizer, model, device, CLS_TOKEN, SEPARATOR_TOKEN, MAX_SEQUENCE_LENGTH,
            EMBEDDING_SIZE
        )
        self.state = np.array([0] * self.state_size).astype(np.float32)
        self.state[:-3] = self.vec_combined
        self.state[-1] = int(self.current_sample[0]['normalized_line_number'])
        self.state[-2] = int(self.current_sample[0]['normalized_cursor_pos'])

    def _calculate_reward_components(self, reverted_response):
        """Calculate reward components - only using CodeBLEU."""
        # CodeBLEU Component
        score = self.code_bleu_response - self.code_bleu_request
        if 0.6 <= score:
            CodeBLEU = 6 * score
        elif 0.6 > score >= 0.4:
            CodeBLEU = 5 * score
        elif 0.4 > score >= 0.1:
            CodeBLEU = 3 * score
        elif 0.1 > score > 0:
            CodeBLEU = 2 * score
        else:
            CodeBLEU = score

        return CodeBLEU

    def step(self, action):
        """Execute one step in the environment."""
        t = time.time()
        self.counter_steps += 1
        torch.cuda.empty_cache()

        # Select and execute action
        if self.current_segment == 0:
            action_name = self.changes_0[action].__name__
            the_action = self.changes_0[action]
            self.current_segment_0, self.current_segment_1, self.manipulated_combined, name_map, self.terminated = the_action(
                self.current_segment_0, self.current_segment_1)
        else:
            action_name = self.changes_1[action].__name__
            the_action = self.changes_1[action]
            self.current_segment_0, self.current_segment_1, self.manipulated_combined, name_map, self.terminated = the_action(
                self.current_segment_1, self.current_segment_0)

        self.name_map.update(name_map)

        # Early termination check
        if self.terminated:
            info = {}
            truncated = False
            return np.array(self.state).astype(np.float32), self.reward, self.terminated, truncated, info

        # Update embeddings and state
        if self.current_segment == 0:
            embedding_vector = get_code_embedding(
                self.current_segment_0, tokenizer, model, device, CLS_TOKEN, SEPARATOR_TOKEN, MAX_SEQUENCE_LENGTH,
                EMBEDDING_SIZE
            )
            self.current_embedding_segment_0 = embedding_vector
            del embedding_vector
            torch.cuda.empty_cache()
            self.state[:-3] = self.current_embedding_segment_1

        if self.current_segment == 1:
            embedding_vector = get_code_embedding(
                self.current_segment_1, tokenizer, model, device, CLS_TOKEN, SEPARATOR_TOKEN, MAX_SEQUENCE_LENGTH,
                EMBEDDING_SIZE
            )
            self.current_embedding_segment_1 = embedding_vector
            del embedding_vector
            torch.cuda.empty_cache()
            self.state[:-3] = self.current_embedding_segment_0

        # Update cursor position
        normalized_line_number, normalized_cursor_position = find_normalized_cursor_position(self.manipulated_combined)
        self.state[-1] = normalized_line_number
        self.state[-2] = normalized_cursor_position

        # Generate response
        manipulated_response = generate_response.generate(self.manipulated_combined)

        if manipulated_response is None:
            self.reward = [-10, -10]
            self.terminated = True
            info = {}
            truncated = True
            return np.array(self.state).astype(np.float32), self.reward, self.terminated, truncated, info

        torch.cuda.empty_cache()

        # Generate original response if needed
        if self.flag == 0:
            self.origin_response = generate_response.generate(self.original_combined)
            torch.cuda.empty_cache()
            self.flag += 1

        # Process responses
        reverted_response = convert_back(manipulated_response, self.name_map)
        del manipulated_response

        # Calculate CodeBLEU metrics
        answer_request = calc_codebleu(
            [self.original_combined], [self.manipulated_combined],
            lang="python", weights=(0.1, 0.1, 0.4, 0.4), tokenizer=tokenize_code
        )
        answer_response = calc_codebleu(
            [self.origin_response], [reverted_response],
            lang="python", weights=(0.1, 0.1, 0.4, 0.4), tokenizer=tokenize_code
        )

        self.code_bleu_request = answer_request['codebleu']
        self.code_bleu_response = answer_response['codebleu']

        # Update state
        self.state[-3] = 0.5 if self.current_segment == 0 else 1

        # Calculate rewards - only using CodeBLEU
        CodeBLEU = self._calculate_reward_components(reverted_response)

        last_reward_CodeBLEU, _ = self.last_reward[0], self.last_reward[1]
        self.reward = [CodeBLEU - last_reward_CodeBLEU, 0]
        self.last_reward = [CodeBLEU, 0]

        self.truncated = False
        info = {}

        # Clean up memory
        torch.cuda.empty_cache()

        # Switch segments
        last_segment = self.current_segment
        self.current_segment = 1 if last_segment == 0 else 0

        # Check for truncation
        if self.counter_steps == self.max_steps:
            self.terminated = False
            self.truncated = True
            return np.array(self.state).astype(np.float32), self.reward, self.terminated, self.truncated, {}

        torch.cuda.empty_cache()
        return np.array(self.state).astype(np.float32), self.reward, self.terminated, self.truncated, info

    def reset(self, seed=41, options=None):
        """Reset the environment to initial state."""
        self.gc_counter += 1
        if self.gc_counter % 10 == 0:
            gc.collect()

        torch.cuda.empty_cache()
        super().reset(seed=seed, options=options)

        # Reset all variables
        self._reset_episode_variables()
        self._initialize_sample()

        # Initialize state
        self.current_embedding_segment_0 = get_code_embedding(
            self.current_segment_0, tokenizer, model, device, CLS_TOKEN, SEPARATOR_TOKEN, MAX_SEQUENCE_LENGTH,
            EMBEDDING_SIZE
        )
        self.current_embedding_segment_1 = get_code_embedding(
            self.current_segment_1, tokenizer, model, device, CLS_TOKEN, SEPARATOR_TOKEN, MAX_SEQUENCE_LENGTH,
            EMBEDDING_SIZE
        )
        self.original_code_embedding_segment_0 = self.current_embedding_segment_0.copy()
        self.original_code_embedding_segment_1 = self.current_embedding_segment_1.copy()

        original_concat_vectors = np.concatenate((
            self.original_code_embedding_segment_0,
            self.original_code_embedding_segment_1
        ))
        original_concat_vectors_reshape = original_concat_vectors.reshape(-1, 1)

        self.original_reduced_vectors = self.pca.fit_transform(original_concat_vectors_reshape)

        self.vec_combined = get_code_embedding(
            self.current_segment_0, tokenizer, model, device, CLS_TOKEN, SEPARATOR_TOKEN, MAX_SEQUENCE_LENGTH,
            EMBEDDING_SIZE
        )

        # initialize states
        self.state = np.array([0] * self.state_size).astype(np.float32)

        self.state[:-3] = self.vec_combined
        self.state[-1] = int(self.current_sample[0]['normalized_line_number'])
        self.state[-2] = int(self.current_sample[0]['normalized_cursor_pos'])

        # Additional reset variables
        self.i = self.current_sample[1]

        return self.state, {}

    def render(self, mode='human'):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Close the environment."""
        pass


# =============================================================================
# ENVIRONMENT FACTORY FUNCTIONS
# =============================================================================

def make_custom_envs():
    """Create a single custom environment with normalization wrappers."""
    env = PromptsEnv(prompts_data_set)

    # Wrap the environment with observation normalization
    env = NormalizeObservation_3(env)

    # Wrap the environment with reward normalization
    env = Monitor(NormalizeMultipleRewards(env))

    return env


def make_custom_envs_f(num_envs=8):
    """Create multiple vectorized environments."""
    return DummyVecEnv([make_custom_envs for _ in range(num_envs)])


def train_model():
    """Main training function for the PPO model."""
    print("Starting training...")

    # Setup directories and logging
    log_dir = "PATH/TO/LOG/DIR"
    stats_dir = "PATH/TO/STATS/DIR"
    model_dir = "PATH/TO/MODEL/DIR"

    # Create directories
    for directory in [log_dir, stats_dir, model_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Hyperparameters
    hyperparams = {
        'learning_rate': 0.00025,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'n_steps': 128,
        'batch_size': 64,
        'n_epochs': 10,
        'clip_range': 0.2,
        'entropy_coef': 0.01
    }

    # Policy architecture
    policy_kwargs = {
        "net_arch": [
            {"pi": [256, 256, 256, 128], "vf": [256, 256, 256, 128]},
        ],
    }

    # Create environments
    env = make_custom_envs_f()

    # Create validation environment
    eval_env = ValPromptsEnv(val_data_set)
    eval_env = NormalizeObservation_3(eval_env)
    eval_env = NormalizeMultipleRewards(eval_env)

    # Create callbacks
    tensorboard_callback = TensorboardCallback(eval_env=eval_env, eval_freq=2048, log_freq=128)
    reduce_lr_call_back = ReduceLROnPlateauCallback()

    # Initialize model
    model_ppo = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        tensorboard_log=log_dir,
        verbose=1,
        device="cuda",
        **hyperparams,
        policy_kwargs=policy_kwargs
    )

    # Training loop
    print("Starting training loop with periodic saves every 4096 steps")
    total_time_steps = 4096

    for i in range(1, 300):
        model_ppo.learn(
            total_timesteps=total_time_steps,
            reset_num_timesteps=False,
            tb_log_name='R_ppo',
            callback=[tensorboard_callback, reduce_lr_call_back]
        )

        # Save model and statistics
        model_ppo.save(f"{model_dir}/{total_time_steps * i}")

        for env_idx, wrapped_env in enumerate(env.envs):
            wrapped_env.obs_rms.save_stats(
                f'{stats_dir}/r_ppo_real_observation_stats_env{env_idx}_{total_time_steps * i}_abbreviation.npz'
            )
            wrapped_env.return_rms_metric1.save_stats(
                f'{stats_dir}/r_ppo_rms_metric1{total_time_steps * i}.npz'
            )
            wrapped_env.return_rms_metric2.save_stats(
                f'{stats_dir}/r_ppo_rms_metric2{total_time_steps * i}.npz'
            )

    env.close()


