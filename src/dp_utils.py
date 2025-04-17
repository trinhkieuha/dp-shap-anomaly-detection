import tensorflow as tf
import dp_accounting
from absl import app
import math
import numpy as np
import tensorflow_probability as tfp

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
        return dp_accounting.rdp.RdpAccountant(orders)
    
    def _make_event(self, sigma, steps=None):
        """
        Creates a Gaussian event for differential privacy accounting.
        Parameters:
        - sigma: float, the noise scale for the Gaussian mechanism.
        Returns:
        - dp_accounting.GaussianDpEvent object
        """
        if steps is None:
            steps = int(math.ceil(self.epochs * self.n / self.batch_size))
        dp_event_obj = dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(self.q, dp_accounting.GaussianDpEvent(sigma)), steps
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
        """
        dp_event_obj = self._make_event(sigma, steps)
        self.accountant.compose(dp_event_obj)
        
        achieved_epsilon = self.accountant.get_epsilon(self.delta)

        return achieved_epsilon

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

        # Compute the L2 norms of the gradients
        l2_norms = tf.norm(grad)
    
        # Compute the 90th percentile
        l2norm_bound = tf.minimum(tfp.stats.percentile(l2_norms, l2norm_pct, interpolation='nearest'), 5)

        # Clip the gradients
        grad = tf.clip_by_norm(grad, clip_norm=l2norm_bound)
        
        # Add noises
        sanitized_grad = add_gaussian_noise(grad, sigma * l2norm_bound) # Add noise

        return sanitized_grad