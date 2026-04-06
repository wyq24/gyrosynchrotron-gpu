import numpy as np
import torch
from spec_utils import simulate_spectrum_optimized

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

class ParameterNormalizer:
    """Handle parameter and observation normalization for SBI model."""

    def __init__(self, param_bounds, x_log_bounds=None, target_range=(-5, 5)):
        """
        Args:
            param_bounds: List of (min, max) for each parameter
            x_log_bounds: (min, max) bounds for log10(x) before normalization
            target_range: Target normalization range (default: (-5, 5))
        """
        self.param_bounds = np.array(param_bounds)
        self.param_min = self.param_bounds[:, 0]
        self.param_max = self.param_bounds[:, 1]
        self.target_min, self.target_max = target_range

        # Set x bounds for log10(x) clipping
        if x_log_bounds is None:
            # Default bounds - you may need to adjust these based on your data
            self.x_log_min, self.x_log_max = 4.0, 9.0
        else:
            self.x_log_min, self.x_log_max = x_log_bounds

    def normalize_params(self, params):
        """Normalize parameters to [-5, 5] range."""
        if isinstance(params, torch.Tensor):
            params = params.cpu().numpy()

        # Clip to bounds
        params_clipped = np.clip(params, self.param_min, self.param_max)

        # Min-max normalization to [0, 1]
        params_01 = (params_clipped - self.param_min) / (self.param_max - self.param_min)

        # Scale to target range
        params_norm = params_01 * (self.target_max - self.target_min) + self.target_min

        return params_norm

    def denormalize_params(self, params_norm):
        """Denormalize parameters from [-5, 5] back to original range."""
        if isinstance(params_norm, torch.Tensor):
            params_norm = params_norm.cpu().numpy()

        # Scale from target range to [0, 1]
        params_01 = (params_norm - self.target_min) / (self.target_max - self.target_min)

        # Scale to original range
        params = params_01 * (self.param_max - self.param_min) + self.param_min

        return params

    def normalize_observation(self, obs):
        """
        Normalize observation using the SAME approach as your training data:
        1. log10 transform
        2. Clip to bounds
        3. Min-max normalize to [0,1]
        4. Scale to target range [-5,5]
        """
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()

        # Step 1: log10 transform (avoid log(0))
        obs_log = np.log10(np.clip(obs, 1e-10, None))

        # Step 2: Clip to the same bounds used in training
        obs_log_clipped = np.clip(obs_log, self.x_log_min, self.x_log_max)

        # Step 3: Min-max normalization to [0, 1]
        obs_01 = (obs_log_clipped - self.x_log_min) / (self.x_log_max - self.x_log_min)

        # Step 4: Scale to target range [-5, 5]
        obs_norm = obs_01 * (self.target_max - self.target_min) + self.target_min

        return obs_norm

    def denormalize_observation(self, obs_norm):
        """
        Denormalize observation back to original brightness temperature scale.
        """
        if isinstance(obs_norm, torch.Tensor):
            obs_norm = obs_norm.cpu().numpy()

        # Scale from target range to [0, 1]
        obs_01 = (obs_norm - self.target_min) / (self.target_max - self.target_min)

        # Scale to log space
        obs_log = obs_01 * (self.x_log_max - self.x_log_min) + self.x_log_min

        # Convert back from log10
        obs = 10 ** obs_log

        return obs


class SBIModelWrapper:
    """Wrapper for SBI model that handles normalization and mask-aware inference."""

    def __init__(self, model_path, param_bounds, vary_indices, fixed_params, simulator_func, x_log_bounds=None):
        self.param_bounds = param_bounds
        self.vary_indices = vary_indices
        self.fixed_params = fixed_params
        self.simulator_func = simulator_func

        # Create normalizer for varying parameters only
        varying_bounds = [param_bounds[i] for i in vary_indices]
        self.normalizer = ParameterNormalizer(
            param_bounds=varying_bounds,
            x_log_bounds=x_log_bounds,
            target_range=(-5, 5)
        )

        # Load SBI model
        self.posterior, self.standardizer, self.F, self.extra_channels, self.config = self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained SBI model from updated training code with mask-aware support."""
        import torch
        import numpy as np
        from sbi.inference import SNPE_C
        from sbi.neural_nets import posterior_nn

        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        if isinstance(model_path, tuple) or isinstance(model_path, list):
            model_path = model_path[-1]
        checkpoint = torch.load(model_path, map_location=device)

        # Extract architecture info
        arch = checkpoint["arch"]
        F = arch["freq_len"]  # Number of features
        D = arch["param_dim"]  # Number of parameters
        extra_channels = arch.get("extra_channels", 0)  # Default to 0

        # Extract training configuration (for mask-aware models)
        config = checkpoint.get("config", {})
        mask_aware = config.get("mask_aware", False)
        sentinel_value = config.get("sentinel_value", -8.0)

        print(f"Loading model: F={F}, D={D}, extra_channels={extra_channels}")
        print(f"Model trained with mask-aware: {mask_aware}")
        if mask_aware:
            print(f"  - Augment mask prob: {config.get('augment_mask_prob', 'unknown')}")
            print(f"  - Sentinel value: {sentinel_value}")

        # Build prior
        param_lows = checkpoint["param_lows"]
        param_highs = checkpoint["param_highs"]
        low = torch.tensor(param_lows, dtype=torch.float32, device=device)
        high = torch.tensor(param_highs, dtype=torch.float32, device=device)
        prior = torch.distributions.Independent(
            torch.distributions.Uniform(low, high), 1
        )

        # Build encoder matching the training architecture
        from torch import nn

        class EfficientEncoder(nn.Module):
            """Match the encoder from training"""

            def __init__(self, input_dim: int, hidden_dim: int = 256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1),

                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1),

                    nn.Linear(hidden_dim, 128),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.net(x)

        # Create encoder
        encoder = EfficientEncoder(input_dim=F, hidden_dim=256)

        # Build density estimator
        density_builder = posterior_nn(
            model="nsf",
            hidden_features=arch["hidden_features"],
            num_transforms=arch["num_transforms"],
            embedding_net=encoder,
        )

        # Initialize SNPE
        inference = SNPE_C(prior=prior, density_estimator=density_builder, device=device)

        # Create dummy data to build the network structure
        dummy_theta = torch.randn(10, D, device=device)
        dummy_x = torch.randn(10, F, device=device)

        # Build network and load weights
        estimator = density_builder(dummy_theta.cpu(), dummy_x.cpu())
        estimator.load_state_dict(checkpoint['state_dict'])
        estimator = estimator.to(device)

        # Build posterior
        posterior = inference.build_posterior(estimator)

        # Create standardizer
        std_info = checkpoint["standardizer"]

        class Standardizer:
            def __init__(self, mean, std, mask_aware=False, sentinel_value=-8.0):
                self.mean = np.array(mean, dtype=np.float32)
                self.std = np.array(std, dtype=np.float32)
                self.mask_aware = mask_aware
                self.sentinel_value = sentinel_value

            def apply(self, x, mask_weights=None):
                """Apply standardization and soft clipping, optionally with masking"""
                # Standardize
                x_std = (x - self.mean) / self.std
                # Soft clip using tanh (matching training)
                x_clipped = np.tanh(x_std / 3.0) * 3.0

                # Apply mask if provided (for mask-aware inference)
                if mask_weights is not None:
                    x_tilde = x_clipped * mask_weights + self.sentinel_value * (1.0 - mask_weights)
                    return x_tilde

                return x_clipped

        standardizer = Standardizer(
            mean=std_info["mean"],
            std=std_info["std"],
            mask_aware=mask_aware,
            sentinel_value=sentinel_value
        )

        return posterior, standardizer, F, extra_channels, config

    def sample_posterior(self, observation, num_samples=1000, mask_weights=None, use_masking=False):
        """
        Sample from SBI posterior given an observation.

        Args:
            observation: Input observation
            num_samples: Number of posterior samples
            mask_weights: Optional mask weights for masked inference (1D array same length as observation)
            use_masking: Whether to apply masking during inference
        """
        # Normalize observation using the parameter normalizer
        obs_norm = self.normalizer.normalize_observation(observation)

        # Prepare observation for SBI
        obs_input = self.prepare_observation_for_sbi(obs_norm, mask_weights, use_masking)

        # Sample from posterior
        samples_norm = self.posterior.sample((num_samples,), x=obs_input)

        # Denormalize samples
        if isinstance(samples_norm, torch.Tensor):
            samples_norm = samples_norm.cpu().numpy()

        samples = self.normalizer.denormalize_params(samples_norm)

        return samples

    def prepare_observation_for_sbi(self, obs_norm, mask_weights=None, use_masking=False):
        """
        Prepare normalized observation for SBI model input.

        Args:
            obs_norm: Normalized observation
            mask_weights: Optional mask weights (1D array)
            use_masking: Whether to apply masking
        """
        import torch
        import numpy as np

        # Get device from posterior model
        try:
            model_device = next(self.posterior._neural_net.parameters()).device
        except:
            model_device = torch.device("cpu")

        # Apply masking if requested and model supports it
        if use_masking and mask_weights is not None:
            if not self.standardizer.mask_aware:
                print("Warning: Masking requested but model wasn't trained with mask-aware augmentation")

            # Ensure mask_weights is numpy array
            if isinstance(mask_weights, torch.Tensor):
                mask_weights = mask_weights.cpu().numpy()

            # Apply the training standardizer with masking
            obs_standardized = self.standardizer.apply(obs_norm, mask_weights)
        else:
            # Apply the training standardizer (with soft clipping, no masking)
            obs_standardized = self.standardizer.apply(obs_norm)

        # Add batch dimension
        obs_input = obs_standardized[None, :]  # Add batch dimension

        # Convert to tensor and move to model's device
        obs_tensor = torch.from_numpy(obs_input.astype(np.float32))
        obs_tensor = obs_tensor.to(model_device)

        return obs_tensor

    def log_probability(self, params, observation, mask_weights=None, use_masking=False):
        """
        Compute log probability of parameters given observation.

        Args:
            params: Parameters to evaluate
            observation: Input observation
            mask_weights: Optional mask weights for masked inference
            use_masking: Whether to apply masking during inference
        """
        # Normalize parameters and observation
        params_norm = self.normalizer.normalize_params(params)
        obs_norm = self.normalizer.normalize_observation(observation)

        # Prepare observation
        obs_input = self.prepare_observation_for_sbi(obs_norm, mask_weights, use_masking)

        # Convert params to tensor
        import torch
        if not isinstance(params_norm, torch.Tensor):
            params_norm = torch.from_numpy(params_norm.astype(np.float32))

        # Ensure on same device
        model_device = next(self.posterior._neural_net.parameters()).device
        params_norm = params_norm.to(model_device)

        # Compute log prob
        log_prob = self.posterior.log_prob(params_norm, x=obs_input)

        return log_prob.cpu().numpy() if isinstance(log_prob, torch.Tensor) else log_prob

    def create_frequency_mask(self, freq_length, mask_type="contiguous", **kwargs):
        """
        Create frequency masks for masked inference.

        Args:
            freq_length: Length of frequency axis
            mask_type: Type of mask ('contiguous', 'random', 'bandpass', 'custom')
            **kwargs: Additional arguments for mask creation

        Returns:
            mask_weights: 1D numpy array of weights [0, 1]
        """
        import numpy as np

        if mask_type == "contiguous":
            # Create contiguous band mask similar to training
            min_keep_frac = kwargs.get('min_keep_frac', 0.60)
            low_drop_max_frac = kwargs.get('low_drop_max_frac', 0.05)
            high_taper_max_frac = kwargs.get('high_taper_max_frac', 0.20)
            taper_range = kwargs.get('high_taper_weight_range', (0.30, 0.80))

            # Random parameters
            min_keep = max(1, int(min_keep_frac * freq_length))
            low_drop_max = int(low_drop_max_frac * freq_length)
            high_taper_max = int(high_taper_max_frac * freq_length)

            k_low = np.random.randint(0, max(1, low_drop_max + 1)) if low_drop_max > 0 else 0
            taper_len = np.random.randint(0, max(1, high_taper_max + 1)) if high_taper_max > 0 else 0
            taper_peak = np.random.uniform(*taper_range) if taper_len > 0 else 0.0

            max_keep = max(1, freq_length - k_low - taper_len)
            keep_len = np.random.randint(min_keep, max_keep + 1) if max_keep >= min_keep else max_keep
            start, end = k_low, k_low + keep_len

            w = np.zeros((freq_length,), dtype=np.float32)
            if end > start:
                w[start:end] = 1.0
            if taper_len > 0 and end < freq_length:
                t = np.linspace(1.0, 0.0, taper_len, dtype=np.float32)
                w[end:min(freq_length, end + taper_len)] = (taper_peak * t)[
                                                           :max(0, min(freq_length, end + taper_len) - end)]
            if end + taper_len < freq_length:
                w[end + taper_len:] = 0.02  # tiny floor

            return w

        elif mask_type == "random":
            # Random binary mask
            keep_frac = kwargs.get('keep_frac', 0.7)
            mask = np.random.rand(freq_length) < keep_frac
            return mask.astype(np.float32)

        elif mask_type == "bandpass":
            # Bandpass filter
            low_freq = kwargs.get('low_freq', 0.1)
            high_freq = kwargs.get('high_freq', 0.9)
            low_idx = int(low_freq * freq_length)
            high_idx = int(high_freq * freq_length)

            mask = np.zeros(freq_length, dtype=np.float32)
            mask[low_idx:high_idx] = 1.0
            return mask

        elif mask_type == "custom":
            # Custom mask provided by user
            custom_mask = kwargs.get('mask_weights')
            if custom_mask is None:
                raise ValueError("Custom mask type requires 'mask_weights' argument")
            return np.array(custom_mask, dtype=np.float32)

        else:
            raise ValueError(f"Unknown mask type: {mask_type}")

    def evaluate_with_different_masks(self, observation, params, num_masks=10, mask_type="contiguous"):
        """
        Evaluate log probability with different random masks to test robustness.
        Useful for models trained with mask-aware augmentation.

        Args:
            observation: Input observation
            params: Parameters to evaluate
            num_masks: Number of different masks to test
            mask_type: Type of masks to generate

        Returns:
            log_probs: List of log probabilities for different masks
            masks: List of masks used
        """
        log_probs = []
        masks = []

        # Evaluate with full spectrum (no mask)
        log_prob_full = self.log_probability(params, observation, use_masking=False)
        log_probs.append(log_prob_full)
        masks.append(np.ones(self.F, dtype=np.float32))

        # Evaluate with different masks
        if self.standardizer.mask_aware:
            for i in range(num_masks):
                mask = self.create_frequency_mask(self.F, mask_type=mask_type)
                log_prob_masked = self.log_probability(params, observation,
                                                       mask_weights=mask, use_masking=True)
                log_probs.append(log_prob_masked)
                masks.append(mask)
        else:
            print("Model not trained with mask-aware augmentation, skipping masked evaluation")

        return log_probs, masks

    def get_model_info(self):
        """Get information about the loaded model."""
        info = {
            'freq_length': self.F,
            'extra_channels': self.extra_channels,
            'mask_aware': self.config.get('mask_aware', False),
            'config': self.config
        }

        if info['mask_aware']:
            info.update({
                'augment_mask_prob': self.config.get('augment_mask_prob'),
                'sentinel_value': self.config.get('sentinel_value'),
                'min_keep_frac': self.config.get('min_keep_frac'),
                'low_drop_max_frac': self.config.get('low_drop_max_frac'),
                'high_taper_max_frac': self.config.get('high_taper_max_frac'),
            })

        return info

class SBIModelWrapper_non_mask:
    """Wrapper for SBI model that handles normalization."""

    def __init__(self, model_path, param_bounds, vary_indices, fixed_params, simulator_func, x_log_bounds=None):
        self.param_bounds = param_bounds
        self.vary_indices = vary_indices
        self.fixed_params = fixed_params
        self.simulator_func = simulator_func

        # Create normalizer for varying parameters only
        varying_bounds = [param_bounds[i] for i in vary_indices]
        self.normalizer = ParameterNormalizer(
            param_bounds=varying_bounds,
            x_log_bounds=x_log_bounds,
            target_range=(-5, 5)
        )

        # Load SBI model
        self.posterior, self.standardizer, self.F, self.extra_channels = self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained SBI model from updated training code."""
        import torch
        import numpy as np
        from sbi.inference import SNPE_C
        from sbi.neural_nets import posterior_nn

        # Get device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        if isinstance(model_path, tuple) or isinstance(model_path, list):
            model_path = model_path[-1]
        checkpoint = torch.load(model_path, map_location=device)

        # Extract architecture info
        arch = checkpoint["arch"]
        F = arch["freq_len"]  # Number of features
        D = arch["param_dim"]  # Number of parameters
        extra_channels = arch.get("extra_channels", 0)  # Default to 0

        # Build prior
        param_lows = checkpoint["param_lows"]
        param_highs = checkpoint["param_highs"]
        low = torch.tensor(param_lows, dtype=torch.float32, device=device)
        high = torch.tensor(param_highs, dtype=torch.float32, device=device)
        prior = torch.distributions.Independent(
            torch.distributions.Uniform(low, high), 1
        )

        # Build encoder matching the training architecture
        from torch import nn

        class EfficientEncoder(nn.Module):
            """Match the encoder from training"""

            def __init__(self, input_dim: int, hidden_dim: int = 256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1),

                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(0.1),

                    nn.Linear(hidden_dim, 128),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.net(x)

        # Create encoder
        encoder = EfficientEncoder(input_dim=F, hidden_dim=256)

        # Build density estimator
        density_builder = posterior_nn(
            model="nsf",
            hidden_features=arch["hidden_features"],
            num_transforms=arch["num_transforms"],
            embedding_net=encoder,
        )

        # Initialize SNPE
        inference = SNPE_C(prior=prior, density_estimator=density_builder, device=device)

        # Create dummy data to build the network structure
        dummy_theta = torch.randn(10, D, device=device)
        dummy_x = torch.randn(10, F, device=device)

        # Build network and load weights
        estimator = density_builder(dummy_theta.cpu(), dummy_x.cpu())
        estimator.load_state_dict(checkpoint['state_dict'])
        estimator = estimator.to(device)

        # Build posterior
        posterior = inference.build_posterior(estimator)

        # Create standardizer
        std_info = checkpoint["standardizer"]

        class Standardizer:
            def __init__(self, mean, std):
                self.mean = np.array(mean, dtype=np.float32)
                self.std = np.array(std, dtype=np.float32)

            def apply(self, x):
                """Apply standardization and soft clipping"""
                # Standardize
                x_std = (x - self.mean) / self.std
                # Soft clip using tanh (matching training)
                x_clipped = np.tanh(x_std / 3.0) * 3.0
                return x_clipped

        standardizer = Standardizer(
            mean=std_info["mean"],
            std=std_info["std"]
        )

        return posterior, standardizer, F, extra_channels

    def sample_posterior(self, observation, num_samples=1000):
        """Sample from SBI posterior given an observation."""
        # Normalize observation using the parameter normalizer
        obs_norm = self.normalizer.normalize_observation(observation)

        # Prepare observation for SBI
        obs_input = self.prepare_observation_for_sbi(obs_norm)

        # Sample from posterior
        samples_norm = self.posterior.sample((num_samples,), x=obs_input)

        # Denormalize samples
        if isinstance(samples_norm, torch.Tensor):
            samples_norm = samples_norm.cpu().numpy()

        samples = self.normalizer.denormalize_params(samples_norm)

        return samples

    def prepare_observation_for_sbi(self, obs_norm):
        """
        Prepare normalized observation for SBI model input.
        """
        import torch
        import numpy as np

        # Get device from posterior model
        try:
            model_device = next(self.posterior._neural_net.parameters()).device
        except:
            model_device = torch.device("cpu")

        # Apply the training standardizer (with soft clipping)
        obs_standardized = self.standardizer.apply(obs_norm)

        # Since training didn't use extra channels, we don't need mask/clip channels
        # Just add batch dimension
        obs_input = obs_standardized[None, :]  # Add batch dimension

        # Convert to tensor and move to model's device
        obs_tensor = torch.from_numpy(obs_input.astype(np.float32))
        obs_tensor = obs_tensor.to(model_device)

        return obs_tensor

    def log_probability(self, params, observation):
        """
        Compute log probability of parameters given observation.
        """
        # Normalize parameters and observation
        params_norm = self.normalizer.normalize_params(params)
        obs_norm = self.normalizer.normalize_observation(observation)

        # Prepare observation
        obs_input = self.prepare_observation_for_sbi(obs_norm)

        # Convert params to tensor
        import torch
        if not isinstance(params_norm, torch.Tensor):
            params_norm = torch.from_numpy(params_norm.astype(np.float32))

        # Ensure on same device
        model_device = next(self.posterior._neural_net.parameters()).device
        params_norm = params_norm.to(model_device)

        # Compute log prob
        log_prob = self.posterior.log_prob(params_norm, x=obs_input)

        return log_prob.cpu().numpy() if isinstance(log_prob, torch.Tensor) else log_prob


def create_normalized_simulator(simulator_func, param_bounds, vary_indices, fixed_params, normalizer):
    """Create simulator that works with normalized parameters for MCMC and returns normalized observations."""

    def normalized_simulator(params_norm):
        """Simulator that takes normalized varying parameters and returns normalized observations."""
        # Denormalize parameters
        params_denorm = normalizer.denormalize_params(params_norm)

        # Create full parameter vector
        full_params = np.array(fixed_params, copy=True)
        for i, idx in enumerate(vary_indices):
            full_params[idx] = params_denorm[i]

        # Run original simulator (returns brightness temperature)
        result = simulator_func(torch.tensor(full_params, dtype=torch.float32))

        # Normalize the observation using the SAME method as training
        result_norm = normalizer.normalize_observation(result)

        return result_norm

    return normalized_simulator


# def create_normalized_simulator_for_mcmc(simulator_func, param_bounds, vary_indices, fixed_params,
#                                          normalizer, standardizer):
#     """
#     Create simulator for MCMC that matches SBI's data processing.
#     MCMC parameters are in [-5, 5], simulator output should be z-score standardized.
#     """
#
#     def normalized_simulator(params_norm):
#         """Simulator that takes normalized varying parameters and returns standardized observations."""
#         # Denormalize parameters from [-5, 5] to original space
#         params_denorm = normalizer.denormalize_params(params_norm)
#
#         # Create full parameter vector
#         full_params = np.array(fixed_params, copy=True)
#         for i, idx in enumerate(vary_indices):
#             full_params[idx] = params_denorm[i]
#
#         # Run original simulator (returns brightness temperature)
#         result = simulator_func(torch.tensor(full_params, dtype=torch.float32))
#
#         if isinstance(result, torch.Tensor):
#             result = result.cpu().numpy()
#
#         # Apply THE SAME transformation as SBI training:
#         # 1. Log transform
#         result_log = np.log10(np.clip(result, 1e-10, None))
#
#         # 2. Clip to bounds (same as training)
#         result_log_clipped = np.clip(result_log, 4.0, 9.0)  # Adjust based on your training
#
#         # 3. Apply z-score standardization using training statistics
#         result_standardized = standardizer.apply(result_log_clipped[None, :])[0]
#
#         return result_standardized
#
#     return normalized_simulator


def log_likelihood_fn_normalized(params_norm, normalized_simulator, observation_norm, noise_level=0.1):
    """Log likelihood for normalized parameters and observations."""
    #try:
        # Run simulator with normalized parameters (returns normalized observations)
    simulation_norm = normalized_simulator(params_norm)

    # Convert to numpy if needed
    if isinstance(simulation_norm, torch.Tensor):
        simulation_norm = simulation_norm.cpu().numpy()
    if isinstance(observation_norm, torch.Tensor):
        observation_norm = observation_norm.cpu().numpy()

    # Both simulation and observation are now in the same normalized space [-5,5]
    # Calculate log likelihood directly
    residuals = observation_norm - simulation_norm
    log_like = -0.5 * np.sum((residuals / noise_level) ** 2)
    log_like -= 0.5 * len(observation_norm) * np.log(2 * np.pi * noise_level ** 2)

    return log_like
    # except Exception as e:
    #     logger.warning(f"Error in log_likelihood: {e}")
    #     return -np.inf


def log_prior_fn_normalized(params_norm):
    """Log prior for normalized parameters (uniform in [-5, 5])."""
    if not isinstance(params_norm, np.ndarray):
        params_norm = np.array(params_norm)

    # Check if all parameters are within [-5, 5]
    if np.any(params_norm < -5) or np.any(params_norm > 5):
        return -np.inf

    return 0.0


def log_probability_fn_normalized(params_norm, normalized_simulator, observation_norm, noise_level=0.1):
    """Combined log prior and log likelihood for normalized MCMC."""
    # Check prior
    log_prior = log_prior_fn_normalized(params_norm)
    if not np.isfinite(log_prior):
        return -np.inf

    # Compute likelihood
    log_like = log_likelihood_fn_normalized(params_norm, normalized_simulator, observation_norm, noise_level)

    return log_prior + log_like

def make_simulator_for_subset(simulator_8d, fixed_params, vary_indices):
    """
    返回一个新的模拟器函数 simulator_n(param_n)，只需要 param_n 是 n 维：
      - 先将 fixed_params 拷贝一份
      - 在 vary_indices 上用 param_n 的值覆盖
      - 再调用原8维模拟器 simulator_8d
    这里示例是单条输入逻辑，如需batched，可进一步修改。
    """
    def simulator_n(param_n):
        if isinstance(param_n, torch.Tensor):
            param_n = param_n.cpu().numpy()
        full_params = np.array(fixed_params, copy=True)
        for i, idx in enumerate(vary_indices):
            full_params[idx] = param_n[i]
        # 调用原8维模拟器
        out = simulator_8d(torch.tensor(full_params, dtype=torch.float32))
        return out
    return simulator_n

def simulator_8d(parameters):
    """
    Simulator function that takes parameter vector and returns observation.
    This follows the sbi required format.

    Parameters:
    -----------
    parameters : torch.Tensor
        Parameters for the model [area_asec2, depth_asec, Bx100G, T_MK, log_nth, log_nnth, delta, theta, Emin_keV, Emax_MeV]

    Returns:
    --------
    spectrum : torch.Tensor
        The simulated spectrum
    """
    # Global freqghz definition - you'll need to set this before using the simulator
    # global freqghz

    # Get the appropriate device
    device = get_device()

    # Convert tensor parameters to numpy for the underlying function
    if isinstance(parameters, torch.Tensor):
        params_np = parameters.cpu().numpy()
    else:
        params_np = np.array(parameters)

    # Call your existing spectrum generation function
    spectrum = simulate_spectrum_optimized(params_np)

    # Ensure output is a torch tensor on the appropriate device
    if not isinstance(spectrum, torch.Tensor):
        spectrum = torch.tensor(spectrum, dtype=torch.float32, device=device)
    else:
        spectrum = spectrum.to(device)

    return spectrum