"""
Code Utilities Module
====================

This module contains utility functions for code processing, embedding generation,
and text manipulation used in the DRL StarCoder training pipeline.

Functions:
- find_normalized_cursor_position: Find normalized cursor position in code
- get_code_embeddings_StarEncoder: Get embeddings using StarEncoder
- get_code_embedding: Get embeddings using pre-trained model
- extract_text: Extract prefix and suffix from formatted code
- tokenize_code: Tokenize Python code using Pygments
- prepare_tokenizer: Initialize tokenizer with special tokens
"""

# =============================================================================
# IMPORTS
# =============================================================================

import re
import torch
from transformers import AutoTokenizer
from pygments import lex
from pygments.lexers import PythonLexer
from StarEncoder import star_encoder

import os

from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np



# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def find_normalized_cursor_position(code, cursor_tag="<fim_suffix>"):
    """
    Find normalized cursor position in code.

    Args:
        code (str): Code string to search in
        cursor_tag (str): Tag to search for (default: "<fim_suffix>")

    Returns:
        tuple: (normalized_line_number, normalized_cursor_pos)
    """
    lines = code.split('\n')
    total_lines = len(lines)
    for line_number, line in enumerate(lines, start=1):
        cursor_pos = line.find(cursor_tag)
        if cursor_pos != -1:
            # Normalize line number
            normalized_line_number = line_number / total_lines
            # Normalize cursor position in line
            normalized_cursor_pos = cursor_pos / len(line) if len(line) > 0 else 0
            return normalized_line_number, normalized_cursor_pos
    return 1, 1


def get_code_embeddings_StarEncoder(code):
    """
    Get code embeddings using StarEncoder.

    Args:
        code (str): Code string to encode

    Returns:
        list: Code embedding vector
    """
    embedding = star_encoder.encode([code])[0]
    return embedding


def get_code_embedding(code, tokenizer, model, device, cls_token, separator_token, max_length=1024, embedding_size=768):
    """
    Get code embedding using the pre-trained model.

    Args:
        code (str): Code string to encode
        tokenizer: Pre-trained tokenizer
        model: Pre-trained model
        device: Computing device (cuda/cpu)
        cls_token (str): Classification token
        separator_token (str): Separator token
        max_length (int): Maximum sequence length
        embedding_size (int): Size of embedding vector

    Returns:
        list: Code embedding vector
    """
    code = f"{cls_token}{code}{separator_token}"
    try:
        # Tokenize the input code
        tokens = tokenizer.encode(code, return_tensors="pt",
                                  max_length=max_length, padding=True, truncation=True).to(device)

        # Disable gradient calculations
        with torch.no_grad():
            outputs = model(tokens)
            embedding = outputs
            return embedding.last_hidden_state[:, 0, :].unsqueeze(1).squeeze().tolist()

    except Exception as e:
        # Log the exception for debugging
        return [0 for _ in range(embedding_size)]


def extract_text(input_string):
    """
    Extract prefix and suffix from formatted code string.

    Args:
        input_string (str): Formatted code string with special tokens

    Returns:
        dict: Dictionary with 'prefix' and 'suffix' keys
    """
    result = {}
    prefix_match = re.search('<fim_prefix>(.*?)<fim_suffix>', input_string, re.DOTALL)
    suffix_match = re.search('<fim_suffix>(.*?)<fim_middle>', input_string, re.DOTALL)
    prefix = prefix_match.group(1).strip() if prefix_match else input_string
    suffix = suffix_match.group(1).strip() if suffix_match else ""
    result['prefix'] = prefix
    result['suffix'] = suffix
    return result


def tokenize_code(code):
    """
    Tokenize Python code using Pygments lexer.

    Args:
        code (str): Python code string to tokenize

    Returns:
        list: List of tokens
    """
    tokens = [token[1] for token in lex(code, PythonLexer()) if token[1]]
    return tokens


def prepare_tokenizer(tokenizer_path, pad_token="<pad>", sep_token="<sep>", cls_token="<cls>", mask_token="<mask>"):
    """
    Initialize and configure the tokenizer with special tokens.

    Args:
        tokenizer_path (str): Path to pre-trained tokenizer
        pad_token (str): Padding token
        sep_token (str): Separator token
        cls_token (str): Classification token
        mask_token (str): Mask token

    Returns:
        AutoTokenizer: Configured tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens({"pad_token": pad_token})
    tokenizer.add_special_tokens({"sep_token": sep_token})
    tokenizer.add_special_tokens({"cls_token": cls_token})
    tokenizer.add_special_tokens({"mask_token": mask_token})
    return tokenizer

class TensorboardCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=20_000, log_freq=128, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.eval_callback = EvalCallback(eval_env, eval_freq=eval_freq, verbose=verbose)

        # To store rewards for logging mean episode reward
        self.rewards = []

    def _on_step(self) -> bool:
        # Collect rewards and log them every `log_freq` steps
        if 'episode' in self.locals:
            episode_rewards = self.locals['episode']['r']
            self.rewards.append(episode_rewards)

        if self.n_calls % self.log_freq == 0:
            # print(self.n_calls)
            # Calculate mean episode reward and log it
            mean_reward = np.mean(self.rewards) if self.rewards else 0
            self.logger.record('train/mean_episode_reward', mean_reward)
            self.rewards = []  # Reset the rewards list after logging

        if self.n_calls % self.eval_freq == 0:
            # Evaluate the policy and log the mean reward of the validation set
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=50)
            print(mean_reward)
            self.logger.record('validation/mean_reward', mean_reward)

        return True


class smallTensorboardCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=5_000, log_freq=128, verbose=1):
        super(smallTensorboardCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_freq = log_freq
        self.eval_callback = EvalCallback(eval_env, eval_freq=eval_freq, verbose=verbose)

        # To store rewards for logging mean episode reward
        self.rewards = []

    def _on_step(self) -> bool:
        # Collect rewards and log them every `log_freq` steps
        if 'episode' in self.locals:
            episode_rewards = self.locals['episode']['r']
            self.rewards.append(episode_rewards)

        if self.n_calls % self.log_freq == 0:
            print(self.n_calls)
            # Calculate mean episode reward and log it
            mean_reward = np.mean(self.rewards) if self.rewards else 0
            self.logger.record('train/mean_episode_reward', mean_reward)
            self.rewards = []  # Reset the rewards list after logging

        if self.n_calls % self.eval_freq == 0:
            # Evaluate the policy and log the mean reward of the validation set
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=20)
            print(mean_reward)
            self.logger.record('validation/mean_reward', mean_reward)

        return True


class ReduceLROnPlateauCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ReduceLROnPlateauCallback, self).__init__(verbose)
        self.best_mean_reward = -np.inf
        self.patience_counter = 0
        self.patience = 30  # Number of evaluations without improvement to wait before reducing the LR
        self.factor = 0.99  # Factor by which the learning rate will be reduced. new_lr = lr * factor.

    def _on_step(self) -> bool:
        if 'episode_reward' in self.locals:
            mean_reward = np.mean(self.locals['episode_reward'])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                # Reduce learning rate
                current_lr = self.model.lr_schedule(self.num_timesteps)
                new_lr = current_lr * self.factor
                self.model.lr_schedule = lambda _: new_lr
                if self.verbose > 0:
                    print(f"Reducing learning rate to {new_lr} due to plateau.")
                self.patience_counter = 0

        return True


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num times steps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

def convert_back(response, name_map):
    for k, v in reversed(name_map.items()):
        response = response.replace(k, v)
    return response

