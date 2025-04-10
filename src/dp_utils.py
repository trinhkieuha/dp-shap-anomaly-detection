import tensorflow as tf

def add_gaussian_noise(tensor, sigma):

    return tensor + tf.random.normal(tf.shape(tensor), stddev=sigma)

class GaussianSanitizer:
    def __init__(self, accountant):
        
        self.accountant = accountant

    def sanitize(self, grad, eps, delta, l2norm_bound=0.1):
        """
        Sanitizes a gradient tensor using the Gaussian mechanism and records the privacy cost.

        Parameters:
        - grad: tf.Tensor, the gradient to be sanitized.
        - eps: float, privacy parameter epsilon.
        - delta: float, privacy parameter delta.
        - l2norm_bound: float, the maximum L2 norm to clip gradients to (default: 0.1).

        Returns:
        - tf.Tensor, the sanitized gradient with added Gaussian noise.
        """

        # Calculate sigma using the desired epsilon and delta
        with tf.control_dependencies(
            [tf.Assert(tf.greater(eps, 0), ["eps needs to be greater than 0"]),
             tf.Assert(tf.greater(delta, 0), ["delta needs to be greater than 0"])]):
            sigma = tf.sqrt(2.0 * tf.math.log(1.25 / delta)) / eps

        # Clip the gradients
        grad = tf.clip_by_norm(grad, clip_norm=l2norm_bound)
        
        # Add noises
        num_examples = tf.slice(tf.shape(grad), [0], [1]) # Extract the number of examples
        privacy_accum_op = self.accountant.accumulate_privacy_spending(eps, delta, sigma, num_examples) # Track the consumed privacy budget
        with tf.control_dependencies([privacy_accum_op]):
            saned_grad = add_gaussian_noise(grad, sigma * l2norm_bound) # Add noise
        return saned_grad