import tensorflow as tf
import dp_accounting
from absl import app
import math

def add_gaussian_noise(tensor, sigma):

    return tensor + tf.random.normal(tf.shape(tensor), stddev=sigma)

class DPSGDSanitizer:
    def __init__(self, n, batch_size, target_epsilon, epochs, delta, c1=1.0, c2=1.0):
        
        self.accountant = self._make_accountant()
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.c2 = c2

        # Check epsilon feasibility
        max_eps = c1 * self.q**2 * self.T
        if self.target_epsilon >= max_eps:
            raise ValueError(
                f"Target epsilon = {self.target_epsilon:.4f} exceeds the theoretical max "
                f"epsilon = {max_eps:.4f} for q = {self.q:.4f}, T = {self.T}."
            )

        # Compute the sampling probability
        self.q = batch_size / n  # the sampling probability
        if self.q > 1:
            raise app.UsageError('n must be larger than the batch size.')
        
        # Compute total training steps
        self.T = int(math.ceil(epochs * n / batch_size))

    def _make_accountant(self):
        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] +
            list(range(5, 64)) + [128, 256, 512])
        return dp_accounting.rdp.RdpAccountant(orders)
    
    def _compute_noise(self):
        
        # Calculate target noise multiplier
        target_noise = (self.c2 * self.q * math.sqrt(self.T * math.log(1 / self.delta))) / self.target_epsilon

        return target_noise

    def compose_privacy_event(self, sigma):
        
        # Create and compose DP event using dp_accounting
        dp_event_obj = dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(self.q, dp_accounting.GaussianDpEvent(sigma)), self.T
        )
        self.accountant.compose(dp_event_obj)
        
        # Sanity check the resulting epsilon
        achieved_epsilon = self.accountant.get_epsilon(self.delta)
        print(f'Composed DP event: achieved epsilon = {achieved_epsilon:.4f} for delta = {self.delta}')

    def sanitize(self, grad, l2norm_bound=0.1, c1=1.0, c2=1.0):
        """
        Sanitizes a gradient tensor using the Gaussian mechanism and records the privacy cost.

        Parameters:
        - grad: tf.Tensor, the gradient to be sanitized.
        - l2norm_bound: float, the maximum L2 norm to clip gradients to (default: 0.1).
        - c1: float, a constant to control the upper bound for epsilon
        - c2: float, a constant to control the lower bound for sigma

        Returns:
        - tf.Tensor, the sanitized gradient with added Gaussian noise.
        - float, the noise scale sigma used
        """

        # Calculate sigma using the desired epsilon and delta
        sigma = self._compute_noise(c1=c1, c2=c2)

        # Clip the gradients
        grad = tf.clip_by_norm(grad, clip_norm=l2norm_bound)
        
        # Add noises
        saned_grad = add_gaussian_noise(grad, sigma * l2norm_bound) # Add noise
        return saned_grad, sigma