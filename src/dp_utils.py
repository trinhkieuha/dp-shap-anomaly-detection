import tensorflow as tf
import dp_accounting
from absl import app
import math
import numpy as np
import tensorflow_probability as tfp

def add_gaussian_noise(tensor, sigma):
    """
    Adds Gaussian noise to a tensor.
    Parameters:
    - tensor: tf.Tensor, the input tensor to which noise will be added.
    - sigma: float, the standard deviation of the Gaussian noise.
    Returns:
    - tf.Tensor, the tensor with added Gaussian noise.
    """

    noise = tf.random.normal(tf.shape(tensor), stddev=sigma)
    
    return tensor + noise

def add_lap_noise(tensor, scale):
    """
    Adds Laplace noise to a tensor.
    Parameters:
    - tensor: tf.Tensor, the input tensor to which noise will be added.
    - scale: float, the scale of the Laplace noise.
    Returns:
    - tf.Tensor, the tensor with added Laplace noise.
    """

    dtype = tensor.dtype  # get the data type of the input tensor
    lap = tfp.distributions.Laplace(loc=tf.constant(0.0, dtype=dtype), scale=tf.constant(scale, dtype=dtype))
    noise = lap.sample(sample_shape=tf.shape(tensor))
    return tensor + noise

def compute_empirical_nsr(reference_scores, baseline_scores):
    """
    Computes the empirical noise-to-signal ratio (NSR) between two sets of reconstruction scores:
    - reference_scores: from a DP-SGD or noisy model
    - baseline_scores: from a non-private model

    Returns:
    - nsr: float, empirical noise-to-signal ratio
    """
    signal_std = np.std(baseline_scores)
    noise = reference_scores - baseline_scores
    noise_std = np.std(noise)

    nsr = noise_std / signal_std
    return nsr

def compute_T_from_nsr(signal_std, epsilon, nsr_target, mechanism="gaussian", delta=None):
    """
    Computes the sensitivity threshold T such that the resulting noise achieves the desired NSR.

    Parameters:
    - signal_std: float, standard deviation of the original (non-private) reconstruction errors
    - epsilon: float, DP epsilon
    - nsr_target: float, desired noise-to-signal ratio
    - mechanism: str, either 'laplace' or 'gaussian'
    - delta: float or None, required for the Gaussian mechanism

    Returns:
    - T: float, calibrated threshold for PTR
    """
    if mechanism == "laplace":
        scale = nsr_target * signal_std / np.sqrt(2)
        T = epsilon * scale  # Since scale = T / ε
    elif mechanism == "gaussian":
        assert delta is not None, "Delta must be specified for the Gaussian mechanism."
        sigma = nsr_target * signal_std
        T = sigma * epsilon / np.sqrt(2 * np.log(1.25 / delta))  # From σ = T √(2log(1.25/δ)) / ε
    else:
        raise ValueError("Unsupported mechanism: choose 'laplace' or 'gaussian'.")

    return T

def max_per_sample_sensitivity(d_real, d_binary, gamma=1.0):
    """
    Compute the theoretical max per-sample reconstruction loss for post-hoc DP.

    Parameters:
    - d_real: int, number of real-valued features
    - d_binary: int, number of binary-valued features
    - gamma: float in [0, 1], weight on MSE (1 - gamma for CE)

    Returns:
    - delta: float, max per-sample sensitivity (for both L1 and L2 if scalar output)
    """
    max_mse = 0.5 * d_real              # max 0.5 per real-valued dim
    max_ce  = np.log(2) * d_binary      # max log(2) per binary dim

    delta = gamma * max_mse + (1 - gamma) * max_ce
    return delta

class DPSGDSanitizer:
    def __init__(self, n, batch_size, target_epsilon, epochs, delta):
        """
        Initializes the DPSGDSanitizer with parameters for differential privacy.
        Parameters:
        - n: int, the total number of samples in the dataset.
        - batch_size: int, the size of each batch for training.
        - target_epsilon: float, the target epsilon for differential privacy.
        - epochs: int, the number of epochs for training.
        - delta: float, the target delta for differential privacy.
        """
        self.n = n
        self.batch_size = batch_size
        self.epochs = epochs
        self.accountant = self._make_accountant()
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.steps_taken = 0

        # Compute the sampling probability
        self.q = batch_size / n  # the sampling probability
        if self.q > 1:
            raise app.UsageError('n must be larger than the batch size.')

    def _make_accountant(self):
        """
        Creates an accountant for differential privacy accounting.
        Returns:
        - dp_accounting.rdp.RdpAccountant object
        """
        # Create a new RDP accountant with specified orders
        orders = np.linspace(1.1, 64.0, num=100)
        return dp_accounting.rdp.RdpAccountant(orders,
                                               neighboring_relation=dp_accounting.NeighboringRelation.REPLACE_ONE
                                               )
    
    def _make_event(self, sigma, steps=None):
        """
        Creates a Gaussian event for differential privacy accounting.
        Parameters:
        - sigma: float, the noise scale for the Gaussian mechanism.
        - steps: int, the number of steps completed so far.
        Returns:
        - dp_accounting.GaussianDpEvent object
        """
        if steps is None:
            steps = int(math.ceil(self.epochs * self.n / self.batch_size))
        dp_event_obj = dp_accounting.SelfComposedDpEvent(
            dp_accounting.SampledWithoutReplacementDpEvent(self.n, self.batch_size, dp_accounting.GaussianDpEvent(sigma)), steps
        )
        return dp_event_obj
    
    def compute_noise_from_eps(self):
        """
        Computes the noise scale for the Gaussian mechanism based on the target epsilon and delta.
        Returns:
        - float, the computed noise scale sigma.
        """
        # Calculate target noise multiplier
        target_noise = dp_accounting.calibrate_dp_mechanism(
            self._make_accountant, self._make_event, self.target_epsilon, self.delta,
            dp_accounting.LowerEndpointAndGuess(1, 2))

        return target_noise

    def compose_privacy_event(self, sigma, steps):
        """
        Composes the privacy event using the Gaussian mechanism and the RDP accountant.

        Parameters:
        - sigma: float, the noise scale for the Gaussian mechanism.
        - steps: int, the number of steps completed so far.
        Returns:
            dp_accounting.SampledWithoutReplacementDpEvent(self.n, self.batch_size, dp_accounting.GaussianDpEvent(sigma)), steps
        - float, the achieved epsilon for the given delta.
        """
        dp_event_obj = self._make_event(sigma, steps)
        self.accountant.compose(dp_event_obj)
        
        achieved_epsilon = self.accountant.get_epsilon(self.delta)

        return achieved_epsilon

    @tf.function
    def sanitize(self, grad, sigma, l2norm_pct=90.0):
        """
        Sanitizes the gradients using the Gaussian mechanism for differential privacy.
        Parameters:
        - per_sample_grads: tf.Tensor, the gradients to be sanitized.
        - sigma: float, the noise scale for the Gaussian mechanism.
        - l2norm_pct: float, the percentile for clipping the gradients.
        Returns:
        - tf.Tensor, the sanitized gradients.
        """    

        # Compute L2 norms for each per-example gradient
        flat_grads = [
            tf.reshape(g, (self.batch_size, -1))
            for g in grad
        ]
        squared_norms = [tf.reduce_sum(tf.square(fg), axis=1) for fg in flat_grads]
        total_squared = tf.add_n(squared_norms) # this has shape [batch]
        l2_norms = tf.sqrt(total_squared + 1e-6)
        
        # Compute the 90th percentile
        l2norm_bound = tf.minimum(tfp.stats.percentile(l2_norms, l2norm_pct, interpolation='nearest'), 5)

        # Clip the gradients
        sanitized_grad = tf.nest.map_structure(
            lambda g: add_gaussian_noise(
                tf.reduce_sum(g * tf.expand_dims(l2norm_bound, axis=tf.range(0, tf.rank(g))[:1]), axis=0),
                sigma * l2norm_bound
            ) / tf.cast(tf.shape(g)[0], tf.float32),
            grad
        )

        return sanitized_grad