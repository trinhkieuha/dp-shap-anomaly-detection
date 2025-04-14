# Import relevant packages
import pandas as pd
import os
import warnings
import numpy as np
from IPython.display import display
import tensorflow as tf
import random
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from itertools import product
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from datetime import datetime
from skopt import Optimizer
from skopt.space import Real, Integer, Categorical
from src.dp_utils import DPSGDSanitizer

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
                 max_epochs=100,
                 patience_limit=10,
                 batch_size=64,
                 activation='relu',
                 dropout_rate=None,
                 verbose=True,
                 plot_losses=False,
                 dp_sgd=False,
                 target_epsilon=1,
                 delta=1e-5,
                 l2norm_bound=1.0):
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
        - target_epsilon: float, default = 0.1, target epsilon for differential privacy.
        - delta: float, default = 1e-5, target delta for differential privacy.
        - l2norm_bound: float, default = 1.0, maximum L2 norm to clip gradients to (used in DP-SGD).
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
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.l2norm_bound = l2norm_bound

        # Set seeds
        self.seed = 42
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

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
                                        delta=self.delta)
            # Calculate sigma using the desired epsilon and delta
            noise_multiplier = sanitizer.compute_noise()
            print(f"Noise multiplier: {noise_multiplier:.4f}")

        # Define a TensorFlow function for each training step
        @tf.function
        def train_step(x_batch, steps):                        
            # Clip gradients if using DP-SGD
            if self.dp_sgd:
                # Compute per-example gradients
                @tf.function
                def grad_fn(x_i):
                    with tf.GradientTape() as tape_i:
                        x_i = tf.expand_dims(x_i, axis=0)
                        recon_i = self.autoencoder(x_i, training=True)
                        loss_i = tf.reduce_sum(loss_fn(x_i, recon_i))  # scalar loss
                    grads = tape_i.gradient(loss_i, self.autoencoder.trainable_variables)
                    grads = tf.nest.map_structure(
                        lambda g, v: tf.zeros_like(v) if g is None else g,
                        grads,
                        self.autoencoder.trainable_variables
                    )
                    return grads
                # Compute per-example gradients
                per_variable_grads = tf.vectorized_map(grad_fn, x_batch)  # [batch_size, var_shape...]

                sanitized_grads = []
                # Loop over each gradient
                for var_index, grads_i in enumerate(per_variable_grads):
                    sanitized_grad = sanitizer.sanitize(grads_i, sigma=noise_multiplier, l2norm_bound=self.l2norm_bound)
                    sanitized_grads.append(sanitized_grad)

                # Apply sanitized gradients
                optimizer.apply_gradients(zip(sanitized_grads, self.autoencoder.trainable_variables))
                
                # Compute mean loss for logging
                recon = self.autoencoder(x_batch, training=False)
                mean_loss = tf.reduce_mean(loss_fn(x_batch, recon))

            else:
                # Apply standard SGD
                with tf.GradientTape() as tape:
                    recon = self.autoencoder(x_batch, training=True) # forward pass
                    # Compute the loss
                    per_batch_loss = loss_fn(x_batch, recon)
                    # Compute the mean loss over the batch
                    mean_loss = tf.reduce_mean(per_batch_loss)
                # Backward pass and update weights
                grads = tape.gradient(mean_loss, self.autoencoder.trainable_variables)
                # Apply gradients
                optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_variables))
                
            # Update the training loss metric
            train_loss_tracker.update_state(mean_loss)

        # Loop over the epoch
        for epoch in range(self.max_epochs):

            # Reset metric states at the start of the epoch
            train_loss_tracker.reset_state()
            val_loss_tracker.reset_state()

            # Loop over mini-batches
            for x_batch in train_data:
                train_step(x_batch, epoch + 1)
            
            # Compose the privacy event
            if self.dp_sgd:
                spent_eps = sanitizer.compose_privacy_event(noise_multiplier, epoch + 1)
                if self.verbose and epoch % 10 == 0:
                    print(f'Composed DP event (after {epoch + 1} steps): achieved epsilon = {spent_eps:.4f} for delta = {self.delta}')
                
            # Validation loss
            val_recon = self.autoencoder(val_data, training=False)
            val_loss_tracker.update_state(tf.reduce_mean(loss_fn(val_data, val_recon)))

            if self.verbose:
                if self.dp_sgd and spent_eps > self.target_epsilon:
                    print(f"\tWARNING: achieved epsilon {spent_eps:.4f} exceeds target epsilon {self.target_epsilon:.4f}.")
                print(f"Epoch {epoch+1} → Train Loss: {train_loss_tracker.result():.4f}, Val Loss: {val_loss_tracker.result():.4f}")
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

        self.autoencoder.set_weights(best_weights)
        
        # Plot the trajectory of the losses
        if self.plot_losses:
            self._track_loss(train_loss_history, val_loss_history)
        
        return self.autoencoder

class AnomalyDetector:
    def __init__(self, model, real_cols, binary_cols, all_cols, lam=1e-4, gamma=0.2):
        """
        Initializes the anomaly detector with a trained autoencoder model.

        Parameters:
        - model: tf.keras.Model, the trained autoencoder
        - real_cols: list of str, real-valued feature names
        - binary_cols: list of str, binary feature names
        - all_cols: list or pd.Index, all input feature names
        - lam: float, L2 regularization coefficient
        - gamma: float, range [0, 1], weight of MSE
        """
        self.model = model
        self.real_cols = real_cols
        self.binary_cols = binary_cols
        self.all_cols = all_cols
        self.lam = lam
        self.gamma = gamma
        self.loss_fn = HybridLoss(model, real_cols, binary_cols, all_cols, lam, gamma, reduce=False)

    def _compute_anomaly_scores(self, x):
        """
        Computes per-sample reconstruction losses using HybridLoss.

        Parameters:
        - x: pd.DataFrame or np.ndarray, input data

        Returns:
        - scores: np.ndarray, reconstruction losses (anomaly scores)
        """
        x_tensor = tf.convert_to_tensor(x.values.astype(np.float32))
        x_hat = self.model(x_tensor, training=False)
        per_sample_loss = self.loss_fn.call(x_tensor, x_hat)
        if len(per_sample_loss.shape) == 0:
            return np.array([per_sample_loss.numpy()])
        return per_sample_loss.numpy()

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
                 max_epochs=100,
                 patience_limit=10,
                 version=None):
        """
        Initializes the hyperparameter tuner for the autoencoder.

        Parameters:
        - x_train: pd.DataFrame, training data (normal-only)
        - x_train_val: pd.DataFrame, training + validation data (used during model fitting)
        - x_val: pd.DataFrame, validation data (mixed, labeled)
        - y_val: np.ndarray, ground truth labels for validation set
        - real_cols: list of str, real-valued feature names
        - binary_cols: list of str, binary-valued feature names
        - all_cols: list or pd.Index of all input feature names
        - activation: str, activation function used in the autoencoder
        - max_epochs: int, maximum number of training epochs
        - patience_limit: int, early stopping patience threshold
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
        if not version:
            self.version = datetime.now().strftime("%Y%m%d%H%M")
        else:
            self.version = version
    
    def _train_model(self, params):
        """
        Trains the autoencoder model with the specified hyperparameters.
        Parameters:
        - params: dict, hyperparameters for the autoencoder
        Returns:
        - model: trained autoencoder model
        """

        trainer = AutoencoderTrainer(
            input_dim=self.x_train.shape[1],
            real_cols=self.real_cols,
            binary_cols=self.binary_cols,
            all_cols=self.all_cols,
            activation=self.activation,
            max_epochs=self.max_epochs,
            patience_limit=self.patience_limit,
            verbose=False,
            **params
        )
        return trainer.train(self.x_train_val, self.x_val)

    def _evaluate_model(self, model, metric, lam=1e-4, gamma=0.2):
        """
        Evaluates the trained autoencoder model on the validation set.
        Parameters:
        - model: trained autoencoder model
        - metric: str, performance metric to evaluate (e.g., 'auc', 'f1_score')
        - lam: float, L2 regularization coefficient
        - gamma: float, weight of MSE
        Returns:
        - threshold: float, optimal threshold for anomaly detection
        - eval_scores: dict, evaluation metrics (accuracy, precision, recall, f1, auc)
        """
        # Compute reconstruction scores
        detector = AnomalyDetector(model, self.real_cols, self.binary_cols, self.all_cols, lam=lam, gamma=gamma)
        scores = detector._compute_anomaly_scores(self.x_val)
        eval_scores = []
        for q in np.linspace(0.70, 0.95, 5):
            threshold = np.quantile(scores, q)
            y_pred = detector._detect(scores, threshold)
            eval_scores.append((threshold, detector._evaluate(y_pred, self.y_val, scores)))
        if metric == 'auc':
            metric = 'f1_score' # if the chosen metric for hyperparameter tuning is 'auc,' use f1 for threshold instead since auc will be the same
        max_score = max(eval_scores, key=lambda x: x[1][metric])
        return max_score[0], max_score[1]

    def sequential_tune(self, param_grid, metric="auc", random_size=5):
        """
        Performs sequential hyperparameter tuning over the parameter grid.

        Parameters:
        - param_grid: dict, keys are parameter names, values are lists of candidate values
        - metric: str, performance metric used to evaluate models (default "auc")
        - random_size: int, number of context samples to use for evaluation (default 5)

        Returns:
        - best_model: trained autoencoder model with best validation score
        - best_params: dict, best-performing hyperparameters including threshold
        - final_score: dict, evaluation metrics of best model
        """
        # Set random seed for reproducibility
        rng = random.Random(123)
        keys = list(param_grid.keys())
        best_params = {}
        final_score = None

        # Set up logging path for saving tuning results
        checkpoint_path = f"experiments/hyperparam_tune/baseline/seq_{metric}_{self.version}.csv"
        
        # Initialize log file and evaluated set if checkpoint doesn't exist
        if not os.path.exists(checkpoint_path):
            with open(checkpoint_path, "w") as f:
                f.write("order,tuned_param,value,metric,mean_score\n")
            evaluated = dict()
        else:
            # Load previously evaluated configurations
            prev_df = pd.read_csv(checkpoint_path)
            evaluated = {
                (row['tuned_param']
                 , str(row['value'])
                 ): row['mean_score']
                for _, row in prev_df.iterrows()
            }
            print(evaluated)
            # Retrieve best values per parameter from past results
            grouped = prev_df.groupby('tuned_param')
            for param, group in grouped:
                best_value = group.loc[group['mean_score'].idxmax(), 'value']
                best_params[param] = best_value

        # Loop through each hyperparameter key to tune
        for i, key in enumerate(keys):
            print(f"\nTuning: {key}")
            scores_for_key = []

            # Build the contextual space excluding the current key
            context_keys = [k for k in keys if k != key]
            context_space = {
                k: [best_params[k]] if k in best_params else param_grid[k]
                for k in context_keys
            }

            # Generate a random subset of context configurations
            all_contexts = list(product(*context_space.values()))
            rng.shuffle(all_contexts)
            sampled_contexts = all_contexts[:min(random_size, len(all_contexts))]

            # Evaluate each candidate value of the current hyperparameter
            for value in param_grid[key]:
                print("\tValue:", value)

                key_value = (key, str(value) if value else 'nan')
                if key_value in evaluated.keys():
                    print(f"\tSkipping previously evaluated: {value}")
                    scores_for_key.append((value, evaluated[key_value]))
                    continue

                trial_scores = []

                # Evaluate each value in sampled contexts
                for context_values in sampled_contexts:
                    config = {k: v for k, v in zip(context_keys, context_values)}
                    config[key] = value  # Insert the value being tuned

                    # Train and evaluate model
                    model = self._train_model(config)
                    eval_rslt = self._evaluate_model(model, metric, lam=config['lam'], gamma=config['gamma'])
                    scores = eval_rslt[1]
                    config["threshold"] = eval_rslt[0]
                    trial_scores.append(scores[metric])

                    # Free memory
                    K.clear_session()
                    del model

                # Compute mean validation score for the current value
                mean_score = np.mean(trial_scores)
                scores_for_key.append((value, mean_score))
                print("\t\tMean score:", mean_score)

                # Log to file
                safe_value = str(value).replace('"', '""')  # Escape any existing double quotes
                with open(checkpoint_path, "a") as f:
                    f.write(f'{i + 1},{key},"{safe_value}",{metric},{mean_score}\n')

            # Choose the value with the best average performance
            best_value = max(scores_for_key, key=lambda x: x[1])[0]
            del scores_for_key
            best_params[key] = best_value
            print(f"→ Best {key}: {best_value}")

        # Train and evaluate final model using the best hyperparameters
        final_model = self._train_model(best_params)
        final_eval = self._evaluate_model(final_model, metric, lam=best_params['lam'], gamma=best_params['gamma'])
        best_params["threshold"] = final_eval[0]
        final_score = final_eval[1]

        return final_model, best_params, final_score
    
    def bo_tune(self, param_space, metric="auc", n_calls=30, random_starts=5):
        """
        Performs Bayesian Optimization to tune hyperparameters using skopt.

        Parameters:
        - param_space: dict, keys are parameter names and values are lists of candidate values.
        - metric: str, performance metric to optimize (default = "auc").
        - n_calls: int, total number of hyperparameter evaluations (random + model-based).
        - random_starts: int, number of initial random evaluations before using surrogate model.

        Returns:
        - final_model: trained autoencoder with best config.
        - best_config: dict, best hyperparameter set including best threshold.
        - final_score: dict, evaluation results of the final model.
        - pd.DataFrame: complete results table.
        """

        # Objective function for skopt: maps hyperparameter config to negative metric score
        def objective(params):
            config = dict(zip(param_space.keys(), params))  # Build dictionary from params
            try:
                model = self._train_model(config)  # Train autoencoder with proposed hyperparameters
                threshold, scores = self._evaluate_model(model, metric, lam=config['lam'], gamma=config['gamma'])  # Evaluate on validation set
                K.clear_session()  # Clear TensorFlow session to free memory
                del model
                return -scores[metric]  # skopt minimizes, so we negate the metric
            except Exception as e:
                print("Failed config:", config)
                print("Error:", e)
                return 1.0  # High penalty loss for failed trials

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
                skopt_space.append(Categorical(safe_values, name=k, transform="identity"))

        # Initialize the Bayesian optimizer
        optimizer = Optimizer(dimensions=skopt_space,
                            n_initial_points=random_starts,
                            random_state=123)

        best_score = -float("inf")     # Track the best validation score found
        best_config = None             # Track the best hyperparameter configuration

        # Prepare log file to save trial history (iteration-wise)
        log_path = f"experiments/hyperparam_tune/baseline/bayes_{metric}_{self.version}.csv"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as f:
            f.write(",".join(list(param_space.keys()) + [metric]) + "\n")

        # Perform the optimization loop
        for i in range(n_calls):
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
                    val = '"' + val.replace('"', '""') + '"'  # Escape internal quotes and wrap in quotes
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
        final_model = self._train_model(decoded_final)
        final_eval = self._evaluate_model(final_model, metric, lam=best_config['lam'], gamma=best_config['gamma'])
        best_config["threshold"] = final_eval[0]  # Add threshold to the config
        final_score = final_eval[1]               # Store the full metric dictionary

        return final_model, best_config, final_score