# Import relevant packages
import pandas as pd
import os
import warnings
import numpy as np
import tensorflow as tf
import random
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from itertools import product
import matplotlib.pyplot as plt
from datetime import datetime
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from src.dp_utils import *
from src.utils import *
from joblib import Parallel, delayed
import glob

# Config
pd.set_option('display.max_columns', None) # Ensure all columns are displayed
warnings.filterwarnings("ignore")

class HybridLoss(tf.keras.losses.Loss):
    def __init__(self, model, real_cols, binary_cols, all_cols, lam, gamma, reduce=True):
        """
        Initializes the inputs to compute loss.

        Parameters:
        - model: tf.keras.Model, the full autoencoder model (used to access weights for L2 regularization)
        - real_cols: list of str, names of real-valued features
        - binary_cols: list of str, names of binary-valued features
        - all_cols: list or pd.Index of all feature names (used to index into x and x_hat)
        - lam: float, regularization coefficient
        - gamma: float, range [0, 1], weight of MSE
        - reduce: bool, if True returns scalar loss (mean over batch),
                  if False returns per-sample loss (for DP or anomaly scoring)
        """

        super().__init__()
        self.model = model
        self.real_cols = real_cols
        self.binary_cols = binary_cols
        self.all_cols = all_cols
        self.lam = lam
        self.gamma = gamma
        self.reduce = reduce

    def call(self, y_true, y_pred):
        col_to_idx = {col: i for i, col in enumerate(self.all_cols)}
        real_idx = [col_to_idx[c] for c in self.real_cols]
        binary_idx = [col_to_idx[c] for c in self.binary_cols]

        # MSE for real-valued columns
        x_real = tf.gather(y_true, real_idx, axis=1)
        x_hat_real = tf.gather(y_pred, real_idx, axis=1)
        mse_loss = tf.reduce_sum(0.5 * tf.square(x_real - x_hat_real), axis=1)  # [batch_size]

        # Cross-entropy for binary columns
        x_bin = tf.gather(y_true, binary_idx, axis=1)
        x_hat_bin = tf.gather(y_pred, binary_idx, axis=1)
        eps = 1e-7
        x_hat_bin = tf.clip_by_value(x_hat_bin, eps, 1 - eps)
        ce_loss = tf.reduce_sum(
            -x_bin * tf.math.log(x_hat_bin) - (1 - x_bin) * tf.math.log(1 - x_hat_bin), axis=1
        )  # [batch_size]

        # Compute reconstruction loss as the weighted sum of MSE and cross-entropy errors
        recon_loss = self.gamma * mse_loss + (1 - self.gamma) * ce_loss

        # L2 regularization term (shared across batch)
        l2_reg = tf.add_n([
            tf.reduce_sum(tf.square(w))
            for w in self.model.trainable_variables if 'kernel' in w.name
        ])
        reg_term = (self.lam / 2.0) * l2_reg

        if self.reduce:
            return tf.reduce_mean(recon_loss) + reg_term
        else:
            return recon_loss + reg_term
    
class AutoencoderTrainer:
    def __init__(self, 
                 input_dim,
                 real_cols,
                 binary_cols,
                 all_cols,
                 hidden_dims=[64, 32],
                 learning_rate=1e-3,
                 lam=1e-4,
                 gamma=0.2,
                 max_epochs=500,
                 patience_limit=10,
                 batch_size=64,
                 activation='relu',
                 dropout_rate=None,
                 verbose=True,
                 plot_losses=False,
                 dp_sgd=False,
                 post_hoc=False,
                 target_epsilon=1,
                 delta=1e-5,
                 l2norm_pct=90.0,
                 save_tracking=False,
                 version=None,
                 raise_convergence_error=True
                 ):
        """
        Initializes the trainer and constructs the encoder-decoder architecture.
        Parameters:
        - input_dim: int, the dimensionality of the input data (number of features).
        - real_cols: list of str, list of column names corresponding to real-valued features (used for MSE loss).
        - binary_cols: list of str, list of column names corresponding to binary features (used for cross-entropy loss).
        - all_cols: list or pd.Index, complete list of all input feature names; used for column indexing.
        - hidden_dims: list of int, default = [64, 32], sizes of the hidden layers for the encoder. The decoder will mirror this structure.
        - learning_rate: float, default = 1e-3, learning rate for the optimizer.
        - lam: float, default = 1e-4, L2 regularization coefficient.
        - gamma: float, range [0, 1], weight of MSE
        - max_epochs: int, default = 100, maximum number of epochs to train the autoencoder.
        - patience_limit: int, default = 10, number of epochs to wait for improvement in validation loss before early stopping.
        - batch_size: int, default = 64, mini-batch size used during training.
        - activation: str, default = 'relu', activation function to use in each hidden layer.
        - dropout_rate: float or None, default = None, if not None, applies dropout with the specified rate after each hidden layer.
        - verbose: boolean, default = True, if True, print log message.
        - plot_losses: boolean, default = False, if True, plot the trajectory of the losses
        - dp_sgd: boolean, default = False, if True, use DP-SGD for training.
        - post_hoc: boolean, default = False, if True, use post-hoc analysis for hyperparameter tuning.
        - target_epsilon: float, default = 0.1, target epsilon for differential privacy.
        - delta: float, default = 1e-5, target delta for differential privacy.
        - l2norm_pct: float, default = 90.0, percentile for clipping the gradients.
        - save_tracking: boolean, default = False, if True, save the privacy tracking information.
        - version: str, default = None, version identifier for the training run.
        - raise_convergence_error: boolean, default = True, if True, raise an error if the model does not converge.
        """
        self.input_dim = input_dim
        self.real_cols = real_cols
        self.binary_cols = binary_cols
        self.all_cols = all_cols
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.lam = lam
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.patience_limit = patience_limit
        self.batch_size = batch_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.verbose = verbose
        self.plot_losses = plot_losses
        self.dp_sgd = dp_sgd
        self.post_hoc = post_hoc
        if not dp_sgd and not post_hoc:
            self.target_epsilon = None
            self.delta = None
            self.l2norm_pct = None
        else:
            self.target_epsilon = target_epsilon
            self.delta = delta
            self.l2norm_pct = l2norm_pct
        self.save_tracking = save_tracking
        if save_tracking:
            self.version = version if version else datetime.now().strftime("%Y%m%d%H%M")
        self.raise_convergence_error = raise_convergence_error

        # Build model
        self.encoder, self.decoder = self._build_autoencoder()
        self.autoencoder = tf.keras.Sequential([self.encoder, self.decoder])

    def _build_autoencoder(self):
        """
        Constructs encoder and decoder networks with optional dropout.

        Returns:
        - encoder: tf.keras.Sequential, the encoder model, which compresses the input into a low-dimensional representation.
        - decoder: tf.keras.Sequential, the decoder model, which reconstructs the input from the encoded representation. 
        Ends with a sigmoid-activated output layer to produce values in [0, 1].
        """
        # --- Encoder ---
        encoder = tf.keras.Sequential(name='encoder')
        for h_dim in self.hidden_dims:
            encoder.add(tf.keras.layers.Dense(
                h_dim,
                activation=self.activation,
                kernel_initializer='glorot_uniform'
            ))
            if self.dropout_rate is not None and not np.isnan(self.dropout_rate):
                encoder.add(tf.keras.layers.Dropout(rate=self.dropout_rate))

        # --- Decoder hidden layers ---
        decoder_input = tf.keras.Input(shape=(self.hidden_dims[-1],), name='decoder_input')
        x = decoder_input
        for h_dim in reversed(self.hidden_dims[:-1]):
            x = tf.keras.layers.Dense(
                h_dim,
                activation=self.activation,
                kernel_initializer='glorot_uniform'
            )(x)
            if self.dropout_rate is not None and not np.isnan(self.dropout_rate):
                x = tf.keras.layers.Dropout(rate=self.dropout_rate)(x)

        # --- Output layers with ordered activations ---
        real_dim = len(self.real_cols)
        binary_dim = len(self.binary_cols)

        real_output = tf.keras.layers.Dense(
            real_dim, activation='linear', name='real_output'
        )(x)

        binary_output = tf.keras.layers.Dense(
            binary_dim, activation='sigmoid', name='binary_output'
        )(x)

        # Concatenate in specified order: [real, binary]
        # Map from column name to output tensor
        output_dict = {col: real_output[:, i] for i, col in enumerate(self.real_cols)}
        output_dict.update({col: binary_output[:, i] for i, col in enumerate(self.binary_cols)})

        # Stack columns in the order of self.all_cols
        ordered_outputs = [tf.expand_dims(output_dict[col], axis=-1) for col in self.all_cols]
        full_output = tf.keras.layers.Concatenate(name='decoder_output')(ordered_outputs)

        decoder = tf.keras.Model(
            inputs=decoder_input,
            outputs=full_output,
            name='decoder'
        )

        return encoder, decoder
    
    def _track_loss(self, train_loss_history, val_loss_history):
        """
        Plots the training and validation loss histories over the last 100 epochs.

        Parameters:
        - train_loss_history: list of float, training loss at each epoch.
        - val_loss_history: list of float, validation loss at each epoch.
        """
        plt.plot(train_loss_history[-100:], label='Train Loss')
        plt.plot(val_loss_history[-100:], label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def train(self, x_train, x_val):
        """
        Trains the autoencoder using standard SGD and early stopping.

        Parameters:
        - x_train: pd.DataFrame, normal training data
        - x_val: pd.DataFrame, validation data for early stopping

        Returns:
        - encoder, decoder: the trained encoder and decoder models
        """
        # Define optimizer and loss
        optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=self.learning_rate)
        loss_fn = HybridLoss(model=self.autoencoder,
                            real_cols=self.real_cols,
                            binary_cols=self.binary_cols,
                            all_cols=self.all_cols,
                            lam=self.lam,
                            gamma=self.gamma,
                            reduce=False) # Not reduce to calculate per-example loss

        # Prepare data for TensorFlow
        train_data = tf.data.Dataset.from_tensor_slices(x_train.values.astype(np.float32))
        train_data = train_data.cache().batch(self.batch_size).prefetch(tf.data.AUTOTUNE) # mini-batches
        val_data = tf.convert_to_tensor(x_val.values.astype(np.float32))

        # Initialize the training
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        patience = 0
        cooldown = 0
        best_weights = None
        train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

        # Initialize the DPSGD sanitizer
        if self.dp_sgd:
            sanitizer = DPSGDSanitizer(n=x_train.shape[0],
                                        batch_size=self.batch_size,
                                        target_epsilon=self.target_epsilon,
                                        epochs=self.max_epochs,
                                        delta=self.delta,
                                        )
            # Calculate sigma using the desired epsilon and delta
            noise_multiplier = sanitizer.compute_noise_from_eps()
            if self.verbose:
                print(f"Noise multiplier: {noise_multiplier:.4f}")
        
        # Prepare log file to save privacy accounting history (iteration-wise)
        if self.save_tracking:
            folder = "experiments/tracking"
            os.makedirs(folder, exist_ok=True)

            # Delete any existing files with the same version prefix
            pattern = os.path.join(folder, f"{self.version}_*.csv")
            for file in glob.glob(pattern):
                os.remove(file)

            if self.dp_sgd:
                log_path = os.path.join(folder, f"{self.version}_noise{noise_multiplier:.4f}.csv")
                with open(log_path, "w") as f:
                    f.write("epoch,spent_eps,train_loss,val_loss\n")
            else:
                if self.post_hoc:
                    log_path = os.path.join(folder, f"{self.version}_posthoc.csv")
                else:
                    log_path = os.path.join(folder, f"{self.version}_baseline.csv")
                with open(log_path, "w") as f:
                    f.write("epoch,train_loss,val_loss\n")

        # Define a TensorFlow function for each training step
        @tf.function
        def train_step(x_batch):
            with tf.GradientTape() as tape:
                recon = self.autoencoder(x_batch, training=True)
                per_batch_loss = loss_fn(x_batch, recon)
                mean_loss = tf.reduce_mean(per_batch_loss)
            grads = tape.gradient(mean_loss, self.autoencoder.trainable_variables)
            if self.dp_sgd:
                sanitized_grads = []
                for px_grad in grads:
                    sanitized_grad = sanitizer.sanitize(px_grad, sigma=noise_multiplier, l2norm_pct=self.l2norm_pct)
                    sanitized_grads.append(sanitized_grad)
                optimizer.apply_gradients(zip(sanitized_grads, self.autoencoder.trainable_variables))
            else:
                optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_variables))
            
            train_loss_tracker.update_state(mean_loss)

        # Loop over the epoch
        for epoch in range(self.max_epochs):

            # Reset metric states at the start of the epoch
            train_loss_tracker.reset_state()
            val_loss_tracker.reset_state()

            # Loop over mini-batches
            for x_batch in train_data:
                train_step(x_batch)
            
            # Validation loss
            val_recon = self.autoencoder(val_data, training=False)
            val_loss_tracker.update_state(tf.reduce_mean(loss_fn(val_data, val_recon)))

            # Compose the privacy event
            if self.dp_sgd:
                spent_eps = sanitizer.compose_privacy_event(noise_multiplier, epoch + 1)  
                if self.verbose:
                    if epoch % 10 == 0:
                        print(f'Composed DP event (after {epoch + 1} steps): achieved epsilon = {spent_eps:.4f} for delta = {self.delta}')
                if spent_eps > self.target_epsilon:
                    if self.verbose:
                        print(f"\tWARNING: achieved epsilon {spent_eps:.4f} exceeds target epsilon {self.target_epsilon:.4f}.")
                    break

            # Print the training and validation loss and privacy event
            if self.verbose:
                print(f"Epoch {epoch+1} → Train Loss: {train_loss_tracker.result():.4f}, Val Loss: {val_loss_tracker.result():.4f}")              
            
            # Update the tracking log with the current step
            if self.save_tracking:
                if self.dp_sgd:
                    with open(log_path, "a") as f:
                        f.write(f"{epoch + 1},{spent_eps:.4f},{train_loss_tracker.result():.4f},{val_loss_tracker.result():.4f}\n")
                else:
                    with open(log_path, "a") as f:
                        f.write(f"{epoch + 1},{train_loss_tracker.result():.4f},{val_loss_tracker.result():.4f}\n")
            
            # Append the losses to the history
            train_loss_history.append(train_loss_tracker.result().numpy())
            val_loss_history.append(val_loss_tracker.result().numpy())

            # Early stopping
            if best_val_loss - val_loss_tracker.result() > -1e-3:
                best_val_loss = val_loss_tracker.result()
                best_weights = self.autoencoder.get_weights()
                patience = 0
                cooldown = 0
            else:
                if cooldown > 0:
                    cooldown -= 1
                else:
                    patience += 1
                    if patience >= self.patience_limit:
                        if self.verbose:
                            print(f"Early stopping triggered at epoch {epoch+1}.")
                        break

        # Check convergence
        loss_history = np.array(val_loss_history)
        n = len(loss_history)
        tail_len = 20 #int(0.1 * n)

        # Ensure tail_len is at least 2 to compute slope
        if n < 2:
            raise ValueError("Too few points in the tail to compute slope.")

        # Compute the slope of the last 10% of the loss history
        y = loss_history[-tail_len:]
        x = np.arange(min(tail_len, n))

        # Fit a line to the last 10% of the loss history
        slope = np.polyfit(x, y, 1)[0]  # degree 1 polynomial fit  
        
        # Check if the slope is close to zero (indicating convergence)
        converged = (slope > -0.003) & (slope < 0.003)

        if not converged:
            if self.raise_convergence_error:
                # Raise error if the model did not converge
                raise ValueError(f"The model did not converge (slope = {slope}). Please check the training parameters.")
            else:
                print(f"The model did not converge (slope = {slope}). Please check the training parameters.")

        # Restore the best weights
        self.autoencoder.set_weights(best_weights)
        
        # Plot the trajectory of the losses
        if self.plot_losses:
            self._track_loss(train_loss_history, val_loss_history)
        
        return self.autoencoder

class AnomalyDetector:
    def __init__(self, model, real_cols, binary_cols, all_cols, lam=1e-4, gamma=0.2
                 , post_hoc=False, noise_mechanism='gaussian'
                 , target_epsilon=1, delta=1e-5):
        """
        Initializes the anomaly detector with a trained autoencoder model.

        Parameters:
        - model: tf.keras.Model, the trained autoencoder
        - real_cols: list of str, real-valued feature names
        - binary_cols: list of str, binary feature names
        - all_cols: list or pd.Index, all input feature names
        - lam: float, L2 regularization coefficient
        - gamma: float, range [0, 1], weight of MSE
        - post_hoc: bool, if True, use post-hoc analysis for hyperparameter tuning
        - noise_mechanism: str, noise mechanism for differential privacy ('gaussian'/'laplace')
        - target_epsilon: float, target epsilon for differential privacy
        - delta: float, target delta for differential privacy
        """
        self.model = model
        self.real_cols = real_cols
        self.binary_cols = binary_cols
        self.all_cols = all_cols
        self.lam = lam
        self.gamma = gamma
        self.post_hoc = post_hoc
        self.noise_mechanism = noise_mechanism
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.loss_fn = HybridLoss(model, real_cols, binary_cols, all_cols, lam, gamma, reduce=False)

    def _compute_anomaly_scores(self, x, **kwargs):
        """
        Computes anomaly scores, with optional post-hoc differential privacy.
        
        If self.post_hoc is True, this method internally computes the noise scale T and 
        uses the baseline (non-private) reconstruction scores, adding calibrated noise.
        Otherwise, it uses the model's reconstruction loss directly.

        Parameters:
        - x: pd.DataFrame or np.ndarray, input data
        - kwargs: if post_hoc=True, must include arguments for _compute_sensitivity

        Returns:
        - scores: np.ndarray of shape (n_samples,), anomaly scores (possibly perturbed)
        """
        # Standard scoring using trained model
        x_tensor = tf.convert_to_tensor(x.values.astype(np.float32))
        x_hat = self.model(x_tensor, training=False)
        scores = self.loss_fn.call(x_tensor, x_hat).numpy()

        if self.post_hoc:
            # Compute the noise scale
            if self.noise_mechanism == "laplace":
                noise = add_lap_noise(tf.convert_to_tensor(scores, dtype=tf.float32), kwargs['noise_multiplier'])
            elif self.noise_mechanism == "gaussian":
                if self.delta is None:
                    raise ValueError("Delta must be set for Gaussian mechanism.")
                noise = add_gaussian_noise(tf.convert_to_tensor(scores, dtype=tf.float32), kwargs['noise_multiplier'])
            else:
                raise ValueError(f"Unsupported noise mechanism: {self.noise_mechanism}")

            scores = noise.numpy()

        else:
            if np.isscalar(scores):
                scores = np.array([scores])
            noise_multiplier = 0

        return scores
        
    def _detect(self, scores, threshold=0.01):
        """
        Detects anomalies in the input based on the reconstruction loss.

        Parameters:
        - scores: np.ndarray, reconstruction losses (anomaly scores)
        - threshold: float, manually specified threshold for anomaly detection

        Returns:
        - y_pred: np.ndarray of shape (n_samples,), binary predictions
        """
        y_pred = (scores > threshold).astype(int)
        return y_pred

    def _evaluate(self, y_pred, y_true, scores):
        """
        Evaluates anomaly detection performance on a labeled test set.

        Parameters:
        - y_pred: np.ndarray of shape (n_samples,), binary predictions
        - y_true: np.ndarray, ground truth labels (0 = normal, 1 = anomaly)
        - scores: np.ndarray of shape (n_samples,), reconstruction losses

        Returns:
        - metrics: dict with accuracy, precision, recall, F1, and AUC
        """
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        auc = roc_auc_score(y_true, scores)
        accuracy = accuracy_score(y_true, y_pred)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
    
    
class AutoencoderTuner:
    def __init__(self, x_train, x_train_val,
                 x_val, y_val,
                 real_cols, binary_cols, all_cols,
                 activation='relu',
                 max_epochs=None,
                 patience_limit=10,
                 version=None,
                 dp_sgd=False,
                 target_epsilon=1,
                 delta=1e-5,
                 continue_run=False,
                 post_hoc=False,
                 noise_mechanism='gaussian',
                 bo_estimator='GP'):
        """
        Initializes the hyperparameter tuner for the autoencoder.

        Parameters:
        - x_train: pd.DataFrame, training data (normal-only)
        - x_train_val: pd.DataFrame, the validation part of the training data (normal-only)
        - x_val: pd.DataFrame, validation data (mixed, labeled)
        - y_val: np.ndarray, ground truth labels for validation set
        - real_cols: list of str, real-valued feature names
        - binary_cols: list of str, binary-valued feature names
        - all_cols: list or pd.Index of all input feature names
        - activation: str, activation function used in the autoencoder
        - max_epochs: int, maximum number of epochs for training
        - patience_limit: int, early stopping patience threshold
        - version: str, version identifier for the tuning run
        - dp_sgd: bool, if True, use differential privacy during training
        - target_epsilon: float, target epsilon for differential privacy
        - delta: float, target delta for differential privacy
        - continue_run: bool, if True, continue from the last checkpoint
        - post_hoc: bool, if True, use post-hoc analysis for hyperparameter tuning
        - noise_mechanism: str, noise mechanism for differential privacy ('gaussian'/'laplace')
        - bo_estimator: str, Bayesian optimization estimator ('GP' or 'ET')
        """
        self.x_train = x_train
        self.x_train_val = x_train_val
        self.x_val = x_val
        self.y_val = y_val
        self.real_cols = real_cols
        self.binary_cols = binary_cols
        self.all_cols = all_cols
        self.activation = activation
        self.max_epochs = max_epochs
        self.patience_limit = patience_limit
        self.dp_sgd = dp_sgd
        self.target_epsilon = target_epsilon
        self.delta = delta
        if not version:
            self.version = datetime.now().strftime("%Y%m%d%H%M")
        else:
            self.version = version
        self.continue_run = continue_run
        self.post_hoc = post_hoc
        self.noise_mechanism = noise_mechanism
        self.bo_estimator = bo_estimator
    
    def _train_model(self, params, final=False, dp_sgd=False):
        """
        Trains the autoencoder model with the specified hyperparameters.
        Parameters:
        - params: dict, hyperparameters for the autoencoder
        - final: bool, if True, trains the final model with the best hyperparameters
        - dp_sgd: bool, if True, uses differential privacy during training
        - post_hoc: bool, if True, uses post-hoc analysis for hyperparameter tuning
        Returns:
        - model: trained autoencoder model
        """
        if final:
            privacy_tracking = True
            verbose = True
            raise_convergence_error = False
        else:
            privacy_tracking = False
            verbose = False
            raise_convergence_error = True

        if self.max_epochs:
            params['max_epochs'] = self.max_epochs

        trainer = AutoencoderTrainer(
            input_dim=self.x_train.shape[1],
            real_cols=self.real_cols,
            binary_cols=self.binary_cols,
            all_cols=self.all_cols,
            activation=self.activation,
            patience_limit=self.patience_limit,
            verbose=verbose,
            dp_sgd=dp_sgd,
            post_hoc=self.post_hoc,
            target_epsilon=self.target_epsilon,
            delta=self.delta,
            save_tracking=privacy_tracking,
            version=self.version,
            raise_convergence_error=raise_convergence_error,
            **params
        )
        
        return trainer.train(self.x_train, self.x_train_val)

    def _evaluate_model(self, metric, model, params, final=False, noise_multiplier=None):
        """
        Evaluates the trained autoencoder model on the validation set.
        Parameters:
        - model: trained autoencoder model
        - metric: str, performance metric to evaluate (e.g., 'auc', 'f1_score')
        - params: dict, hyperparameters for the autoencoder
        - final: bool, if True, saves the final reconstruction scores
        - noise_multiplier: float, noise multiplier for differential privacy
        Returns:
        - threshold: float, optimal threshold for anomaly detection
        - eval_scores: dict, evaluation metrics (accuracy, precision, recall, f1, auc)
        """
        
        detector = AnomalyDetector(model, self.real_cols, self.binary_cols, self.all_cols, lam=params["lam"], gamma=params["gamma"]
                                   , post_hoc=self.post_hoc, noise_mechanism=self.noise_mechanism,
                                   target_epsilon=self.target_epsilon, delta=self.delta)
        if self.post_hoc:
            scores = detector._compute_anomaly_scores(x=self.x_val,
                                                      noise_multiplier=noise_multiplier)
            if final == True:
                os.makedirs(os.path.dirname("experiments/scores/posthoc_dp/"), exist_ok=True)
                score_path = f"experiments/scores/posthoc_dp/{self.version}_noise{noise_multiplier}.feather"
        else:
            scores = detector._compute_anomaly_scores(x=self.x_val)
            if final == True:
                if self.dp_sgd:
                    os.makedirs(os.path.dirname("experiments/scores/dpsgd/"), exist_ok=True)
                    score_path = f"experiments/scores/dpsgd/{self.version}.feather"
                else:
                    os.makedirs(os.path.dirname("experiments/scores/baseline/"), exist_ok=True)
                    score_path = f"experiments/scores/baseline/{self.version}.feather"

        # Save the final reconstruction scores
        if final:
            pd.DataFrame(scores, columns=['score']).to_feather(score_path)

        eval_scores = []
        for q in np.linspace(0.70, 0.95, 5):
            threshold = np.quantile(scores, q)
            y_pred = detector._detect(scores, threshold)
            eval_scores.append((threshold, detector._evaluate(y_pred, self.y_val, scores)))
        if metric == 'auc':
            metric = 'f1_score' # if the chosen metric for hyperparameter tuning is 'auc,' use f1 for threshold instead since auc will be the same
        max_score = max(eval_scores, key=lambda x: x[1][metric])
        return max_score[0], max_score[1]
    
    @staticmethod
    def run_trial(seed, config, tuner_self, metric, noise_multiplier=None):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Train the model
        if tuner_self.post_hoc:
            # Train the baseline model with the current configuration
            model = tuner_self._train_model(config, final=False, dp_sgd=False)

        else: # For DP-SGD and baseline
            # Train the specified type of model with the current configuration
            model = tuner_self._train_model(config, final=False, dp_sgd=tuner_self.dp_sgd)
            
        # Evaluate the model
        threshold, scores = tuner_self._evaluate_model(metric, model, config, noise_multiplier=noise_multiplier)

        clear_tf_memory()
        del model

        return scores[metric]

    def bo_tune(self, param_space, metric="auc", n_calls=30, random_starts=5, eval_num=3, warm_start=True):
        """
        Performs Bayesian Optimization to tune hyperparameters using skopt.

        Parameters:
        - param_space: dict, keys are parameter names and values are lists of candidate values.
        - metric: str, performance metric to optimize (default = "auc").
        - n_calls: int, total number of hyperparameter evaluations (random + model-based).
        - random_starts: int, number of initial random evaluations before using surrogate model.
        - warm_start: bool, whether to continue from previous trials (default = True).

        Returns:
        - final_model: trained autoencoder with best config.
        - best_config: dict, best hyperparameter set including best threshold.
        - final_score: dict, evaluation results of the final model.
        - pd.DataFrame: complete results table.
        """

        # Compute the sensitivity value T for the noise scale    
        if self.post_hoc:
            # Load the sensitivity file
            sensitivity_df = pd.read_csv("experiments/perf_summary/sensitivity.csv")

            # Get the sensitivity value for the epsilon, delta, and noise mechanism
            T = sensitivity_df.loc[
                (sensitivity_df['epsilon'] == self.target_epsilon) &
                (sensitivity_df['delta'] == self.delta) &
                (sensitivity_df['noise_mechanism'] == self.noise_mechanism) &
                (sensitivity_df['metric'] == metric),
                'sensitivity'
            ].values[0]
            # Compute the noise scale
            if self.noise_mechanism == "laplace":
                noise_multiplier = T / self.target_epsilon
                print(f"Using noise scale = {T} for Laplace mechanism.")
            elif self.noise_mechanism == "gaussian":
                if self.delta is None:
                    raise ValueError("Delta must be set for Gaussian mechanism.")
                noise_multiplier = T * np.sqrt(2 * np.log(1.25 / self.delta)) / self.target_epsilon
                print(f"Using noise sigma = {T} for Gaussian mechanism.")
            else:
                raise ValueError(f"Unsupported noise mechanism: {self.noise_mechanism}")
        else:
            noise_multiplier = None

        # Objective function for skopt: maps hyperparameter config to negative metric score
        def objective(params):
            config = dict(zip(param_space.keys(), params))  # Build dictionary from params
            try:
                seeds = [1234 + i for i in range(eval_num)]
                results = Parallel(n_jobs=2)(
                    delayed(AutoencoderTuner.run_trial)(seed, config, self, metric, noise_multiplier) for seed in seeds
                )
                results = [r for r in results if r is not None]
                if not results or len(results) == 0:
                    return 1.0  # All failed
                mean_score = np.mean(results)
                return -mean_score
            except Exception as e:
                print("Failed config:", config)
                print("Error:", e)
                return 1.0
            
        # Prepare log file to save trial history (iteration-wise)
        if self.dp_sgd:
            model_type = "dpsgd"
        elif self.post_hoc:
            model_type = "posthoc_dp"
        else:
            model_type = "baseline"
        log_path = f"experiments/hyperparam_tune/{model_type}/bayes_{metric}_{self.version}.csv"
        os.makedirs(os.path.dirname(log_path), exist_ok=True) # Create directory if it doesn't exist

        # Define skopt-compatible hyperparameter search space
        skopt_space = []
        for k, v in param_space.items():
            if all(isinstance(i, float) for i in v):
                skopt_space.append(Real(min(v), max(v), name=k))
            elif all(isinstance(i, int) for i in v):
                skopt_space.append(Integer(min(v), max(v), name=k))
            else:
                # Handle categorical lists (e.g., hidden_dims like [64, 32])
                safe_values = [tuple(i) if isinstance(i, list) else i for i in v]
                skopt_space.append(Categorical(safe_values, name=k, transform="onehot"))

        # Initialize the Bayesian optimizer
        if self.continue_run:
            random_starts = 0
        elif warm_start == True:
            perf_sum = pd.read_csv(f"experiments/perf_summary/{model_type}_val_results.csv")
            metric_match = perf_sum[perf_sum["tuned_by"].str.lower().str.replace("-", "_") == metric]
            if self.dp_sgd or self.post_hoc:
                metric_match = metric_match[metric_match["epsilon"] == self.target_epsilon]
                metric_match = metric_match[metric_match["delta"] == self.delta]
                if self.post_hoc:
                    metric_match = metric_match[metric_match["noise_mechanism"] == self.noise_mechanism]
            if len(metric_match) > 0:
                prev_version = metric_match["version"].tolist()[0]
                if os.path.exists(f"experiments/hyperparam_tune/{model_type}/bayes_{metric}_{prev_version}.csv"):
                    k = random_starts
                    random_starts = 0
                else:
                    prev_version = None
            else:
                prev_version = None
        
        # Initialize the Bayesian optimizer
        optimizer = Optimizer(dimensions=skopt_space,
                            n_initial_points=random_starts,
                            base_estimator=self.bo_estimator,
                            acq_func="EI",                # to promote exploration
                            acq_func_kwargs={"xi": 1} # to promote exploration
                            )
        
        # Initialize log file and evaluated set if checkpoint doesn't exist
        completed_trials = set()
        if self.continue_run and os.path.exists(log_path) and len(pd.read_csv(log_path)) > 0:
            df_log = pd.read_csv(log_path)

            for _, row in df_log.iterrows():
                params = []
                for k in param_space.keys():
                    val = row[k]
                    params.append(parse_param_value(val, k, param_space))
                loss = -float(row[metric])
                if len(df_log) < n_calls:
                    try:
                        optimizer.tell(params, loss)
                    except Exception as e:
                        print("Failed to tell optimizer:", e)
                        # Remove the row from the log file
                        df_log.drop(index=row.name, inplace=True)
                        df_log.to_csv(log_path, index=False)
                        continue
                completed_trials.add(tuple(params))
            # Get the best config and score from log
            best_row = df_log.loc[df_log[metric].idxmax()]
            best_config = {}
            for k in param_space.keys():
                val = best_row[k]
                parsed_val = parse_param_value(val, k, param_space)
                best_config[k] = list(parsed_val) if isinstance(param_space[k][0], list) else parsed_val
            best_score = float(best_row[metric])
        else:
            # Create the log file with updated header
            with open(log_path, "w") as f:
                f.write(",".join(list(param_space.keys()) + [metric]) + "\n")
            best_score = -float("inf")     # Track the best validation score found
            best_config = None             # Track the best hyperparameter configuration

        # If warm_start is True, load the previous best configurations
        if warm_start and not self.continue_run and prev_version:
            # Load the previous best configurations
            top_configs_df = pd.read_csv(f"experiments/hyperparam_tune/{model_type}/bayes_{metric}_{prev_version}.csv").sort_values(metric, ascending=False).head(k)

            for i, row in top_configs_df.iterrows():
                x0 = []
                for k in param_space.keys():
                    val = parse_param_value(row[k], k, param_space)
                    x0.append(val)
                y0 = objective(x0)  # Skopt uses loss, not score
                top_configs_df.at[i, metric] = -y0
                try:
                    optimizer.tell(x0, y0)
                    completed_trials.add(tuple(x0))

                    # Log the warm-start trial to current log file
                    safe_values = []
                    for k in param_space.keys():
                        val = str(parse_param_value(row[k], k, param_space))
                        if "," in val or '"' in val:
                            val = '"' + val.replace('"', '""') + '"'  # Escape internal quotes
                        safe_values.append(val)

                    with open(log_path, "a") as f:
                        f.write(",".join(safe_values) + f",{-y0}\n")
                except Exception as e:
                    print("Failed warm-start config:", x0)
                    print("Error:", e)

            # Get the best config and score from log
            best_row = top_configs_df.loc[top_configs_df[metric].idxmax()]
            best_config = {}
            for k in param_space.keys():
                val = best_row[k]
                parsed_val = parse_param_value(val, k, param_space)
                best_config[k] = list(parsed_val) if isinstance(param_space[k][0], list) else parsed_val
            best_score = float(best_row[metric])

        # Perform the optimization loop
        for i in range(n_calls - len(completed_trials)):
            next_params = optimizer.ask()  # Suggest next set of hyperparameters

            # Evaluate the proposed parameters and get corresponding loss
            loss = objective(next_params)

            optimizer.tell(next_params, loss)  # Update optimizer with the new observation
            score = -loss                      # Convert loss back to a score

            trial_config = dict(zip(param_space.keys(), next_params))  # Map params to dict

            # Append current trial to results and write to log
            safe_values = []
            for k in param_space.keys():
                val = str(trial_config[k])
                if "," in val or '"' in val:
                    val = '"' + val.replace('"', '""') + '"'  # Escape internal quotes
                safe_values.append(val)

            with open(log_path, "a") as f:
                f.write(",".join(safe_values) + f",{score}\n")

            # Update best config if this trial performs better
            if score > best_score:
                best_score = score
                best_config = trial_config

        # Convert tuple-valued parameters (e.g. hidden_dims) back to list for training
        decoded_final = {
            k: list(v) if isinstance(v, tuple) and isinstance(param_space[k][0], list) else v
            for k, v in best_config.items()
        }

        # Retrain and evaluate the best configuration
        print("Start training the final model with best config...")
        
        # Train the model
        if self.post_hoc: # For post-hoc model
            # Train the baseline model with the current configuration
            final_model = self._train_model(decoded_final, final=True, dp_sgd=False)

        else: # For DP-SGD and baseline
            # Train the specified type of model with the current configuration
            final_model = self._train_model(decoded_final, final=True, dp_sgd=self.dp_sgd)
            
        # Evaluate the model
        final_eval = self._evaluate_model(metric, final_model, best_config, final=True, noise_multiplier=noise_multiplier)
        best_config["threshold"] = final_eval[0]  # Add threshold to the config
        final_score = final_eval[1]               # Store the full metric dictionary

        return final_model, best_config, final_score