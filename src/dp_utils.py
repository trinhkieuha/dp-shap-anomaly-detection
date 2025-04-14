import tensorflow as tf
import dp_accounting
from absl import app
import math
import numpy as np

def add_gaussian_noise(tensor, sigma):
    
    return tensor + tf.random.normal(tf.shape(tensor), stddev=sigma)

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
        self.accountant = self._make_accountant()
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.steps_taken = 0  # New attribute to track training steps

        # Compute the sampling probability
        self.q = batch_size / n  # the sampling probability
        if self.q > 1:
            raise app.UsageError('n must be larger than the batch size.')
        
        # Compute total training steps
        self.T = int(math.ceil(epochs * n / batch_size))
        
        # Check epsilon feasibility
        max_eps = self.q**2 * self.T
        if self.target_epsilon >= max_eps:
            raise ValueError(
                f"Target epsilon = {self.target_epsilon:.4f} exceeds the theoretical max "
                f"epsilon = {max_eps:.4f} for q = {self.q:.4f}, T = {self.T}."
            )

    def _make_accountant(self):
        """
        Creates an accountant for differential privacy accounting.
        Returns:
        - dp_accounting.rdp.RdpAccountant object
        """
        # Create a new RDP accountant with specified orders
        orders = np.linspace(1.1, 64.0, num=100)
        return dp_accounting.rdp.RdpAccountant(orders)
    
    def compute_noise(self):
        """
        Computes the noise scale for the Gaussian mechanism based on the target epsilon and delta.
        Returns:
        - float, the computed noise scale sigma.
        """
        # Calculate target noise multiplier
        target_noise = (1.5 * self.q * math.sqrt(self.T * math.log(1 / self.delta))) / self.target_epsilon

        return target_noise

    def compose_privacy_event(self, sigma, steps):
        """
        Composes the privacy event using the Gaussian mechanism and the RDP accountant.

        Parameters:
        - sigma: float, the noise scale for the Gaussian mechanism.
        - steps: int, the number of steps completed so far.
        """
        dp_event_obj = dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(self.q, dp_accounting.GaussianDpEvent(sigma)), steps
        )
        self.accountant.compose(dp_event_obj)
        
        achieved_epsilon = self.accountant.get_epsilon(self.delta)

        return achieved_epsilon

    def sanitize(self, per_sample_grads, sigma, l2norm_bound=0.1):
        """
        Sanitizes per-sample gradients using DP-SGD:
        - Clips each individual gradient to l2norm_bound
        - Averages the clipped gradients
        - Adds Gaussian noise to the average

        Parameters:
        - per_sample_grads: tf.Tensor of shape [batch_size, ...] — gradients for a single variable
        - sigma: float — noise multiplier
        - l2norm_bound: float — max L2 norm for clipping

        Returns:
        - tf.Tensor — sanitized gradient (same shape as one sample)
        """
        # Flatten each per-sample gradient for norm computation
        grads_flat = tf.reshape(per_sample_grads, [tf.shape(per_sample_grads)[0], -1])  # shape: [batch_size, num_params]
        norms = tf.norm(grads_flat, axis=1)  # shape: [batch_size]

        # Compute clipping factors
        scaling = tf.minimum(1.0, l2norm_bound / (norms + 1e-8))  # shape: [batch_size]
        scaling = tf.reshape(scaling, [-1] + [1] * (len(per_sample_grads.shape) - 1))  # broadcast shape

        # Clip
        grads_clipped = per_sample_grads * scaling  # shape: [batch_size, ...]

        # Average over batch
        avg_grad = tf.reduce_mean(grads_clipped, axis=0)  # shape: [...]

        # Add Gaussian noise
        noise_stddev = sigma * l2norm_bound
        noise = tf.random.normal(shape=tf.shape(avg_grad), stddev=noise_stddev)
        sanitized_grad = avg_grad + noise

        return sanitized_grad