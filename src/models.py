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

# Config
pd.set_option('display.max_columns', None) # Ensure all columns are displayed
warnings.filterwarnings("ignore")

class HybridLoss(tf.keras.losses.Loss):
    def __init__(self, model, real_cols, binary_cols, all_cols, lam, reduce=True):
        """
        Initializes the inputs to compute loss.

        Parameters:
        - model: tf.keras.Model, the full autoencoder model (used to access weights for L2 regularization)
        - real_cols: list of str, names of real-valued features ℛ
        - binary_cols: list of str, names of binary-valued features ℬ
        - all_cols: list or pd.Index of all feature names (used to index into x and x_hat)
        - lam: float, regularization coefficient (λ)
        - reduce: bool, if True returns scalar loss (mean over batch),
                  if False returns per-sample loss (for DP or anomaly scoring)
        """

        super().__init__()
        self.model = model
        self.real_cols = real_cols
        self.binary_cols = binary_cols
        self.all_cols = all_cols
        self.lam = lam
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

        recon_loss = mse_loss + ce_loss  # [batch_size]

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
                 max_epochs=100,
                 patience_limit=10,
                 batch_size=64,
                 activation='relu',
                 dropout_rate=None,
                 verbose=True):
        """
        Initializes the trainer and constructs the encoder-decoder architecture.
        Parameters:
        - input_dim: int, the dimensionality of the input data (number of features).
        - real_cols: list of str, list of column names corresponding to real-valued features (used for MSE loss).
        - binary_cols: list of str, list of column names corresponding to binary features (used for cross-entropy loss).
        - all_cols: list or pd.Index, complete list of all input feature names; used for column indexing.
        - hidden_dims: list of int, default = [64, 32], sizes of the hidden layers for the encoder. The decoder will mirror this structure.
        - activation: str, default = 'relu', activation function to use in each hidden layer.
        - dropout_rate: float or None, default = None, if not None, applies dropout with the specified rate after each hidden layer.
        - learning_rate: float, default = 1e-3, learning rate for the optimizer.
        - lam: float, default = 1e-4, L2 regularization coefficient.
        - batch_size: int, default = 64, mini-batch size used during training.
        - max_epochs: int, default = 100, maximum number of epochs to train the autoencoder.
        - patience_limit: int, default = 10, number of epochs to wait for improvement in validation loss before early stopping.
        """
        self.input_dim = input_dim
        self.real_cols = real_cols
        self.binary_cols = binary_cols
        self.all_cols = all_cols
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.lam = lam
        self.max_epochs = max_epochs
        self.patience_limit = patience_limit
        self.batch_size = batch_size
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.verbose = verbose

        # Set seeds
        self.seed = 42
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Build model
        self.encoder, self.decoder = self.build_autoencoder()
        self.autoencoder = tf.keras.Sequential([self.encoder, self.decoder])

    def build_autoencoder(self):
        """
        Constructs encoder and decoder networks with optional dropout.

        Returns:
        - encoder: tf.keras.Sequential, the encoder model, which compresses the input into a low-dimensional representation.
        - decoder: tf.keras.Sequential, the decoder model, which reconstructs the input from the encoded representation. 
        Ends with a sigmoid-activated output layer to produce values in [0, 1].
        """
        encoder = tf.keras.Sequential()
        for h_dim in self.hidden_dims:
            encoder.add(tf.keras.layers.Dense(h_dim
                                              , activation=self.activation
                                              , kernel_initializer='glorot_uniform'))
            if self.dropout_rate is not None:
                encoder.add(tf.keras.layers.Dropout(rate=self.dropout_rate))

        decoder = tf.keras.Sequential()
        for h_dim in reversed(self.hidden_dims[:-1]):
            decoder.add(tf.keras.layers.Dense(h_dim
                                              , activation=self.activation
                                              , kernel_initializer='glorot_uniform'))
            if self.dropout_rate is not None:
                decoder.add(tf.keras.layers.Dropout(rate=self.dropout_rate))

        decoder.add(tf.keras.layers.Dense(self.input_dim, activation='sigmoid', kernel_initializer='glorot_uniform'))

        return encoder, decoder

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
                            lam=self.lam)

        # Compile model
        self.autoencoder.compile(optimizer=optimizer, loss=loss_fn)

        # Early stopping callback
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience_limit,
            restore_best_weights=True,
            verbose=1 if self.verbose else 0
        )

        # Fit model
        self.autoencoder.fit(
            x=x_train.values.astype(np.float32),
            y=x_train.values.astype(np.float32),
            validation_data=(x_val.values.astype(np.float32), x_val.values.astype(np.float32)),
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            shuffle=True,
            callbacks=[early_stop],
            verbose=1 if self.verbose else 0
        )

        return self.autoencoder

class AnomalyDetector:
    def __init__(self, model, real_cols, binary_cols, all_cols, lam=1e-4):
        """
        Initializes the anomaly detector with a trained autoencoder model.

        Parameters:
        - model: tf.keras.Model, the trained autoencoder
        - real_cols: list of str, real-valued feature names
        - binary_cols: list of str, binary feature names
        - all_cols: list or pd.Index, all input feature names
        - lam: float, L2 regularization coefficient
        """
        self.model = model
        self.real_cols = real_cols
        self.binary_cols = binary_cols
        self.all_cols = all_cols
        self.lam = lam
        self.loss_fn = HybridLoss(model, real_cols, binary_cols, all_cols, lam, reduce=False)

    def compute_anomaly_scores(self, x):
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

    def detect(self, scores, threshold=0.01):
        """
        Detects anomalies in the input based on the reconstruction loss.

        Parameters:
        - scores: np.ndarray, reconstruction losses (anomaly scores)
        - threshold: float, manually specified threshold for anomaly detection

        Returns:
        - y_pred: np.ndarray of shape (n_samples,), binary predictions
        - scores: np.ndarray of shape (n_samples,), reconstruction losses
        """
        y_pred = (scores > threshold).astype(int)
        return y_pred

    def evaluate(self, y_pred, y_true, scores):
        """
        Evaluates anomaly detection performance on a labeled test set.

        Parameters:
        - y_pred: np.ndarray of shape (n_samples,), binary predictions
        - y_true: np.ndarray, ground truth labels (0 = normal, 1 = anomaly)
        - scores: np.ndarray of shape (n_samples,), reconstruction losses

        Returns:
        - metrics: dict with precision, recall, F1, and AUC
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
    def __init__(self, x_train, x_train_val, x_val, y_val, real_cols, binary_cols, all_cols, verbose=True):
        """
        Initializes the hyperparameter tuner for the autoencoder.

        Parameters:
        - x_train: pd.DataFrame, training data (normal-only)
        - x_val: pd.DataFrame, validation data (mixed, labeled)
        - y_val: np.ndarray, ground truth labels for validation set
        - real_cols: list of str, real-valued feature names
        - binary_cols: list of str, binary-valued feature names
        - all_cols: list or pd.Index of all input feature names
        """
        self.x_train = x_train
        self.x_train_val = x_train_val
        self.x_val = x_val
        self.y_val = y_val
        self.real_cols = real_cols
        self.binary_cols = binary_cols
        self.all_cols = all_cols
        self.verbose = verbose

    def tune(self, param_grid, metric="auc"):
        """
        Performs grid search over the parameter grid.

        Parameters:
        - param_grid: dict, keys are parameter names, values are lists of candidate values

        Returns:
        - best_model: trained autoencoder model with best AUC
        - best_params: dict, hyperparameters corresponding to best model
        - best_auc: float, AUC score on validation set
        """
        results = []
        best_metric_val = -np.inf
        best_model = None
        best_params = None

        # Grid search over remaining parameters
        keys, values = zip(*param_grid.items())
        for combination in product(*values):
            params = dict(zip(keys, combination))
            print(f"\nTraining with: {params}")

            # Train autoencoder
            trainer = AutoencoderTrainer(
                input_dim=self.x_train.shape[1],
                real_cols=self.real_cols,
                binary_cols=self.binary_cols,
                all_cols=self.all_cols,
                hidden_dims=params.get('hidden_dims', [64]),
                learning_rate=params.get('learning_rate', 1e-2),
                lam=params.get('lam', 1e-3),
                max_epochs=params.get('max_epochs', 100),
                patience_limit=params.get('patience_limit', 10),
                batch_size=params.get('batch_size', 64),
                activation=params.get('activation', 'relu'),
                dropout_rate=params.get('dropout_rate', None),
                verbose=self.verbose
            )
            model = trainer.train(self.x_train, self.x_train_val)

            # Calculate scores
            detector = AnomalyDetector(
                    model=model,
                    real_cols=self.real_cols,
                    binary_cols=self.binary_cols,
                    all_cols=self.all_cols,
                    lam=params['lam']
                )
            scores = detector.compute_anomaly_scores(self.x_val)

            # Evaluate over threshold grid
            for q in np.linspace(0.70, 0.95, 5):
                threshold = np.quantile(scores, q)
                y_pred = detector.detect(scores, threshold)
                eval_scores = detector.evaluate(y_pred, self.y_val, scores)
                metric_val = eval_scores[metric]

                print(f"  Threshold {threshold:.4f} → {metric.capitalize()} = {metric_val:.4f}")

                if metric_val > best_metric_val:
                    best_metric_val = metric_val
                    best_model = model
                    best_params = params.copy()
                    best_params['threshold'] = threshold
                
                results.append({**params, 'threshold': threshold, **eval_scores})

        print("\nBest parameters found:")
        for k, v in best_params.items():
            print(f"- {k}: {v}")
        print(f"Best validation {metric.capitalize()}: {best_metric_val:.4f}")

        results_df = pd.DataFrame(results)

        return best_model, best_params, best_metric_val, results_df