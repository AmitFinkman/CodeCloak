import os

import numpy as np

import gymnasium as gym


"""Set of wrappers for normalizing actions and observations."""


class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def save_stats(self, filepath):
        """Saves the statistics to a file, creating the directory if it doesn't exist."""
        # Extract the directory path from the filepath
        directory = os.path.dirname(filepath)

        # Create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Save the statistics to the file
        np.savez(filepath, mean=self.mean, var=self.var, count=self.count)

    def load_stats(self, filepath):
        """Loads the statistics from a file."""
        with np.load(filepath) as data:
            self.mean = data['mean']
            self.var = data['var']
            self.count = data['count']


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation_7(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]

        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations,
        excluding the last two values"""
        # Update the running mean and variance using all observation values
        self.obs_rms.update(obs)

        # Copy the observation to avoid normalization of the last two values
        obs_normalized = np.copy(obs)

        # Adjust slicing to exclude the last two columns
        last_column_to_normalize = obs.shape[1] - 7
        obs_normalized[:, :last_column_to_normalize] = (obs[:, :last_column_to_normalize] - self.obs_rms.mean[
                                                                                            :last_column_to_normalize]) / np.sqrt(
            self.obs_rms.var[:last_column_to_normalize] + self.epsilon)

        return obs_normalized


class NormalizeObservation_3(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8, train=True):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        self.train = train

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]

        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs, info = self.env.reset(**kwargs)

        if self.is_vector_env:
            return self.normalize(obs), info
        else:
            return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations,
        excluding the last two values"""
        # Update the running mean and variance using all observation values
        if self.train:
            self.obs_rms.update(obs)

        # Copy the observation to avoid normalization of the last two values
        obs_normalized = np.copy(obs)

        # Adjust slicing to exclude the last two columns
        last_column_to_normalize = obs.shape[1] - 3
        obs_normalized[:, :last_column_to_normalize] = (obs[:, :last_column_to_normalize] - self.obs_rms.mean[
                                                                                            :last_column_to_normalize]) / np.sqrt(
            self.obs_rms.var[:last_column_to_normalize] + self.epsilon)

        return obs_normalized


class NormalizeMultipleRewards(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,

    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        # Initialize RunningMeanStd for each metric
        self.return_rms_metric1 = RunningMeanStd(shape=())
        self.return_rms_metric2 = RunningMeanStd(shape=())
        self.returns_metric1 = np.zeros(self.num_envs)
        self.returns_metric2 = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon


    def step(self, action):
        obs, (metric1, metric2), terminateds, truncateds, infos = self.env.step(action)

        # Update the returns for each metric
        self.returns_metric1 = self.returns_metric1 * self.gamma * (1 - terminateds) + metric1
        self.returns_metric2 = self.returns_metric2 * self.gamma * (1 - terminateds) + metric2

        # Normalize each metric separately
        norm_metric1 = self.normalize(metric1, self.return_rms_metric1)
        norm_metric2 = self.normalize(metric2, self.return_rms_metric2)

        # Combine the normalized metrics (example: simple sum)
        # combined_reward = 0.8 * norm_metric1 + 0.2 * norm_metric2
        combined_reward = 1.0 * norm_metric1 + 0.0 * norm_metric2



        return obs, combined_reward, terminateds, truncateds, infos

    def normalize(self, reward, return_rms):
        """Normalizes the reward using specified RunningMeanStd instance."""
        if not self.is_vector_env:
            reward = np.array([reward])
        normalized_reward = np.zeros_like(reward)
        for i, rew in enumerate(reward):
            return_rms.update(np.array([rew]))
            normalized_reward[i] = rew / np.sqrt(return_rms.var + self.epsilon)
        if not self.is_vector_env:
            normalized_reward = normalized_reward[0]
        return normalized_reward


class NormalizeReward(gym.core.Wrapper, gym.utils.RecordConstructorArgs):
    """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma * (1 - terminateds) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        """Normalizes each reward sample-wise."""
        normalized_rewards = np.zeros_like(rews)
        for i, reward in enumerate(rews):
            # Update statistics for each reward sample
            self.return_rms.update(np.array([reward]))
            # Normalize each reward sample
            normalized_rewards[i] = reward / np.sqrt(self.return_rms.var + self.epsilon)
        return normalized_rewards


