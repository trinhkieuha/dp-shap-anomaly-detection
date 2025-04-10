# Import relevant packages
import pandas as pd
import warnings
import numpy as np
from IPython.display import display
import tensorflow as tf
import random
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
from itertools import product
import matplotlib.pyplot as plt
from collections import defaultdict


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
                 plot_losses=False):
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
            if self.dropout_rate is not None:
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
            if self.dropout_rate is not None:
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
        train_data = train_data.batch(self.batch_size).prefetch(tf.data.AUTOTUNE) # mini-batches
        val_data = tf.convert_to_tensor(x_val.values.astype(np.float32))

        # Initialize the training
        train_loss_history = []
        val_loss_history = []
        best_val_loss = float('inf')
        patience = 0
        best_weights = None
        train_loss_tracker = tf.keras.metrics.Mean(name="train_loss")
        val_loss_tracker = tf.keras.metrics.Mean(name="val_loss")

        # Define a TensorFlow function for each training step
        @tf.function
        def train_step(x_batch):
            with tf.GradientTape() as tape:
                recon = self.autoencoder(x_batch, training=True)
                per_batch_loss = loss_fn(x_batch, recon)
                mean_loss = tf.reduce_mean(per_batch_loss)
            grads = tape.gradient(mean_loss, self.autoencoder.trainable_variables)
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

            if self.verbose:
                print(f"Epoch {epoch+1} → Train Loss: {train_loss_tracker.result():.4f}, Val Loss: {val_loss_tracker.result():.4f}")
            train_loss_history.append(train_loss_tracker.result().numpy())
            val_loss_history.append(val_loss_tracker.result().numpy())

            # Early stopping
            if val_loss_tracker.result() < best_val_loss:
                best_val_loss = val_loss_tracker.result()
                best_weights = self.autoencoder.get_weights()
                patience = 0
            else:
                patience += 1
                if patience >= self.patience_limit:
                    if self.verbose:
                        print("Early stopping triggered.")
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
                 patience_limit=10):
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

    def grid_tune(self, param_grid, metric="auc"):
        """
        Performs grid search over the parameter grid.
 
        Parameters:
        - param_grid: dict, keys are parameter names, values are lists of candidate values
 
        Returns:
        - best_model: trained autoencoder model with best validation score
        - best_params: dict, hyperparameters corresponding to best model
        - best_metric_val: float, best score on validation set
        - results_df: pd.DataFrame, contains all hyperparameter results
        """
        results = []
        best_metric_val = -np.inf
        best_model = None
        best_params = None
 
        keys, values = zip(*param_grid.items())
        for combination in product(*values):
            params = dict(zip(keys, combination))
            print(f"\nTraining with: {params}")
 
            model = self._train_model(params)
            metric_val = self._evaluate_model(model, metric)
            print(f"  {metric.capitalize()} = {metric_val:.4f}")
 
            if metric_val > best_metric_val:
                best_metric_val = metric_val
                best_model = model
                best_params = params.copy()
 
            results.append({**params, metric: metric_val})
 
        print("\nBest parameters found:")
        for k, v in best_params.items():
            print(f"- {k}: {v}")
        print(f"Best validation {metric.capitalize()}: {best_metric_val:.4f}")
 
        results_df = pd.DataFrame(results)
        return best_model, best_params, best_metric_val, results_df
    
    def sequential_tune(self, param_grid, metric="auc", random_size=5):
        random.seed(123)
        keys = list(param_grid.keys())
        best_params = {}
        result_records = []
 
        for i, key in enumerate(keys):
            print(f"\nTuning: {key}")
            scores_for_key = []
 
            # Define the search space excluding the current key
            context_keys = [k for k in keys if k != key]
            context_space = {
                k: [best_params[k]] if k in best_params else param_grid[k]
                for k in context_keys
            }
 
            # Generate shared context configs
            all_contexts = list(product(*context_space.values()))
            random.shuffle(all_contexts)
            sampled_contexts = all_contexts[:min(random_size, len(all_contexts))]  # sample up to 5
 
            # Build full config for each value of the current key
            for value in param_grid[key]:
                print("\tValue:", value)
                trial_scores = []
 
                for context_values in sampled_contexts:
                    # Generate the config dictionary
                    config = {k: v for k, v in zip(context_keys, context_values)}
                    config[key] = value # Add tuned hyperparameter
                    # Train and evaluate model
                    model = self._train_model(config)
                    eval_rslt = self._evaluate_model(model, metric)
                    scores = eval_rslt[1] # Model performance metrics
                    config["threshold"] = eval_rslt[0] # Add best threshold results
                    # Append the chosen metric value
                    trial_scores.append(scores[metric])
                    print("\t\t", config, "-", scores[metric])
                    # Append the tuned results to result_records 
                    result_records.append({'order': i + 1, 'tuned_param': key, **config, **scores})

                # Calculate the mean score
                mean_score = np.mean(trial_scores)
                scores_for_key.append((value, mean_score))
 
            # Select best value for this hyperparameter
            best_value = max(scores_for_key, key=lambda x: x[1])[0]
            best_params[key] = best_value
            print(f"→ Best {key}: {best_value}")
 
        # Final model with tuned parameters
        final_model = self._train_model(best_params)
        final_eval = self._evaluate_model(final_model, metric)
        best_params["threshold"] = final_eval[0]
        final_score = final_eval[1]
 
        return final_model, best_params, final_score, pd.DataFrame(result_records)
    
    def _train_model(self, params):
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

    def _evaluate_model(self, model, metric):
        detector = AnomalyDetector(model, self.real_cols, self.binary_cols, self.all_cols)
        scores = detector._compute_anomaly_scores(self.x_val)
        eval_scores = []
        for q in np.linspace(0.70, 0.95, 5):
            threshold = np.quantile(scores, q)
            y_pred = detector._detect(scores, threshold)
            eval_scores.append((threshold, detector._evaluate(y_pred, self.y_val, scores)))
        max_score = max(eval_scores, key=lambda x: x[1][metric])
        return max_score[0], max_score[1]