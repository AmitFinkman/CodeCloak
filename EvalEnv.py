#!/usr/bin/env python3
"""
Evaluation Environment for DRL StarCoder
========================================

Key Differences from Training Environment:
- Sequential sample iteration (no random sampling)
- LLM model called only once when agent terminates/truncates (not every step)
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard Library Imports
import gc
import logging

# Third-Party Imports
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# Transformer and NLP Imports
from transformers import logging as transformers_logging
from transformers import AutoModel

# Code Analysis Imports
from codebleu import calc_codebleu

# Local Imports
from CodeManipulations  import *
from StarEncoder import star_encoder
from DataSet import *
from SendRequestLLM import generate_response
from utils import (
    find_normalized_cursor_position,
    get_code_embeddings_StarEncoder,
    get_code_embedding,
    extract_text,
    tokenize_code,
    prepare_tokenizer,
    convert_back
)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

transformers_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

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

# Model Configuration
EMBEDDING_SIZE = 768
MAX_SEQUENCE_LENGTH = 1024
MAX_STEPS_PER_EPISODE = 15
NUM_ACTIONS = 12

# =============================================================================
# MODEL INITIALIZATION
# =============================================================================

# Initialize tokenizer and model
tokenizer = prepare_tokenizer("bigcode/starencoder", PAD_TOKEN, SEPARATOR_TOKEN, CLS_TOKEN, MASK_TOKEN)
model = AutoModel.from_pretrained("bigcode/starencoder").to(device)

# =============================================================================
# CODE MANIPULATION ACTIONS
# =============================================================================

# Define action lists for different code segments (same as training)
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
# EVALUATION ENVIRONMENT CLASS
# =============================================================================

class ValPromptsEnv(gym.Env):
    """
    Validation Environment for Code Manipulation Evaluation

    This environment is used for evaluating trained agents on validation datasets.
    It iterates through samples sequentially and provides detailed evaluation metrics.
    """

    def __init__(self, prompts_data_set, embedding_size=EMBEDDING_SIZE, evaluation=False):
        """Initialize the evaluation environment."""
        super(ValPromptsEnv, self).__init__()

        # Environment Configuration
        self.evaluation = evaluation
        self.prompts_data_set = prompts_data_set
        self.state_size = embedding_size + 1 + 1 + 1  # embedding + segment + scores + cursor
        self.max_steps = MAX_STEPS_PER_EPISODE

        # Action and Observation Spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1 * np.inf, high=np.inf,
            shape=(self.state_size,), dtype=np.float32
        )

        # Sequential sample tracking for evaluation
        self.sample_counter = -1
        self.global_counter_steps = 0

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
        self.reverted_response = ""
        self.vec_response = None

    def _initialize_sample(self):
        """Initialize a new sample from the dataset (sequential for evaluation)."""
        self.current_sample = self.prompts_data_set[self.sample_counter]
        self.original_combined = self.current_sample['truncated_combined']
        self.manipulated_combined = self.current_sample['truncated_combined']
        self.current_segment_0 = self.current_sample['segment_0']
        self.current_segment_1 = self.current_sample['segment_1']

        # Get embeddings
        self.current_embedding_segment_0 = get_code_embedding(
            self.current_sample['segment_0'], tokenizer, model, device, CLS_TOKEN, SEPARATOR_TOKEN, MAX_SEQUENCE_LENGTH,
            EMBEDDING_SIZE
        )
        self.current_embedding_segment_1 = get_code_embedding(
            self.current_sample['segment_1'], tokenizer, model, device, CLS_TOKEN, SEPARATOR_TOKEN, MAX_SEQUENCE_LENGTH,
            EMBEDDING_SIZE
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
        self.state[-1] = int(self.current_sample['normalized_line_number'])
        self.state[-2] = int(self.current_sample['normalized_cursor_pos'])

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

    def _handle_episode_termination(self):
        """Handle termination logic (when stop action is called or max steps reached)."""
        torch.cuda.empty_cache()

        # Generate original response if needed
        if not hasattr(self, 'origin_response') or self.origin_response == "":
            self.origin_response = generate_response.generate(self.original_combined)

        # Generate manipulated response
        if self.counter_steps == 1 and any('stop_changes_segment_0' in str(action) for action in self.changes_0):
            # If stopped immediately, use original response
            manipulated_response = self.origin_response
        else:
            manipulated_response = generate_response.generate(self.manipulated_combined)

        # Process response
        reverted_response = convert_back(manipulated_response, self.name_map)
        self.reverted_response = reverted_response

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

        # Calculate reward
        CodeBLEU = self._calculate_reward_components(reverted_response)
        last_reward_CodeBLEU, _ = self.last_reward[0], self.last_reward[1]
        self.reward = (CodeBLEU - last_reward_CodeBLEU, 0)

        torch.cuda.empty_cache()
        return self.reward

    def step(self, action):
        """Execute one step in the environment."""
        self.global_counter_steps += 1
        self.counter_steps += 1
        torch.cuda.empty_cache()
        gc.collect()

        # Select and execute action
        if self.current_segment == 0:
            action_name = self.changes_0[action].__name__
            the_action = self.changes_0[action]
            if self.evaluation:  # Print action name during evaluation
                print(f"Action: {action_name}")
        else:
            action_name = self.changes_1[action].__name__
            the_action = self.changes_1[action]
            if self.evaluation:
                print(f"Action: {action_name}")

        # Execute action
        if self.current_segment == 0:
            self.current_segment_0, self.current_segment_1, self.manipulated_combined, name_map, self.terminated = the_action(
                self.current_segment_0, self.current_segment_1)
        else:
            self.current_segment_0, self.current_segment_1, self.manipulated_combined, name_map, self.terminated = the_action(
                self.current_segment_1, self.current_segment_0)

        self.name_map.update(name_map)

        # Handle termination
        if self.terminated:
            self.reward = self._handle_episode_termination()
            info = {}
            return np.array(self.state).astype(np.float32), self.reward, self.terminated, False, info

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

        # Update cursor position and segment info
        normalized_line_number, normalized_cursor_position = find_normalized_cursor_position(self.manipulated_combined)
        self.state[-1] = normalized_line_number
        self.state[-2] = normalized_cursor_position
        self.state[-3] = 0.5 if self.current_segment == 0 else 1

        # Switch segments
        self.current_segment = 1 if self.current_segment == 0 else 0

        # Check for truncation
        if self.counter_steps == self.max_steps:
            self.reward = self._handle_episode_termination()
            self.terminated = False
            self.truncated = True
            return np.array(self.state).astype(np.float32), self.reward, self.terminated, self.truncated, {}

        self.truncated = False
        torch.cuda.empty_cache()
        gc.collect()

        return np.array(self.state).astype(np.float32), self.reward, self.terminated, self.truncated, {}

    def reset(self, seed=41, options=None):
        """Reset the environment to next sample in sequence."""
        torch.cuda.empty_cache()
        super().reset(seed=seed, options=options)

        # Move to next sample sequentially
        self.sample_counter += 1

        # Reset episode variables
        self._reset_episode_variables()

        # Initialize new sample (with bounds checking)
        dataset_size = len(self.prompts_data_set)
        if self.sample_counter >= dataset_size:
            self.sample_counter = 0  # Reset to beginning if at end

        self._initialize_sample()

        # Additional initialization
        self.i = self.sample_counter

        return self.state, {}

    def render(self, mode='human'):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Close the environment."""
        pass

    def get_current_sample_info(self):
        """Get information about current sample for evaluation tracking."""
        return {
            'sample_id': self.sample_counter,
            'original_combined': self.original_combined,
            'manipulated_combined': self.manipulated_combined,
            'origin_response': getattr(self, 'origin_response', ''),
            'reverted_response': getattr(self, 'reverted_response', ''),
            'code_bleu_request': getattr(self, 'code_bleu_request', 0),
            'code_bleu_response': getattr(self, 'code_bleu_response', 0),
            'steps_taken': self.counter_steps
        }