"""
Model zoo for the comparative study.

Six model families give the paper a genuinely useful comparison axis:
  1. Naive persistence     (unscaled — correct this time, see note below)
  2. Linear Regression     (linear baseline)
  3. Random Forest         (bagged trees)
  4. XGBoost               (boosted trees)
  5. LSTM                  (recurrent deep learning)
  6. GRU                   (recurrent deep learning, fewer params than LSTM)
  7. Transformer (encoder) (attention-based, standard in recent literature)
  8. Stacked Ensemble      (meta-learner over 3-7) — often the actual best
     performer and a natural "contribution" for the paper.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Input, Dropout, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D, Add
)
from .. import config


# ----------------------------------------------------------------------
# Naive baseline
# ----------------------------------------------------------------------
def naive_persistence(y_raw_unscaled: np.ndarray, split_start_idx: int, n_test: int) -> np.ndarray:
    """
    Predicts tomorrow's return = today's actual (unscaled) log return.
    IMPORTANT: must use the raw unscaled log return series, never the
    MinMax-scaled feature matrix — that mismatch produces nonsensical
    MSE/R^2 (a bug worth flagging in your paper's methodology footnote
    if you ever see it, since it's an easy mistake to make).
    """
    return y_raw_unscaled[split_start_idx: split_start_idx + n_test]


# ----------------------------------------------------------------------
# Classical ML baselines (operate on flattened 2D features)
# ----------------------------------------------------------------------
def fit_linear_regression(X_train_2d, y_train):
    """
    Uses Ridge (L2-regularized) rather than plain OLS.
    Flattening a 60-timestep x N-feature lookback window into one long
    vector produces hundreds of highly correlated columns (rolling
    indicators overlap heavily across adjacent timesteps). Plain
    LinearRegression on that input is a classic multicollinearity
    blowup — near-singular design matrix, wildly unstable coefficients,
    and catastrophic out-of-sample error. Ridge's L2 penalty keeps
    coefficients bounded and turns this into a legitimate, stable linear
    baseline. This is standard practice for flattened-sequence linear
    baselines and is worth stating explicitly in the paper's methodology.
    """
    model = Ridge(alpha=10.0)
    model.fit(X_train_2d, y_train)
    return model


def fit_random_forest(X_train_2d, y_train):
    # n_jobs capped at 4 rather than -1 (all cores): each parallel worker
    # needs its own scratch buffers, and n_jobs=-1 on a machine with many
    # logical cores multiplies peak memory usage substantially. This showed
    # up as a hard process crash (not a catchable Python exception) on IBM
    # after ~17 stocks of a long session, most likely from cumulative
    # system-wide memory pressure combined with this per-fold thread fan-out.
    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=config.RANDOM_SEED, n_jobs=4
    )
    model.fit(X_train_2d, y_train)
    return model


def fit_xgboost(X_train_2d, y_train):
    # Same reasoning as fit_random_forest above: capped n_jobs to reduce
    # peak memory from parallel histogram-building buffers.
    model = XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=config.RANDOM_SEED, n_jobs=4
    )
    model.fit(X_train_2d, y_train)
    return model


# ----------------------------------------------------------------------
# Deep sequence models (operate on 3D sequences: [samples, timesteps, features])
# ----------------------------------------------------------------------
def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model


def build_gru(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model


def _transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout=0.1):
    attn_out = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization(epsilon=1e-6)(x)

    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(x.shape[-1])(ff)
    x = Add()([x, ff])
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


def build_transformer(input_shape, head_size=32, num_heads=4, ff_dim=64, num_blocks=2):
    """
    Small transformer encoder for return sequence regression.
    Positional info is implicit via the fixed lookback ordering; add a
    learned positional embedding if you want to be extra rigorous.
    """
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = _transformer_encoder_block(x, head_size, num_heads, ff_dim)
    x = GlobalAveragePooling1D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return model


def fit_sequence_model(build_fn, X_train_3d, y_train, epochs, batch_size=None, verbose=0):
    batch_size = batch_size or config.BATCH_SIZE
    model = build_fn((X_train_3d.shape[1], X_train_3d.shape[2]))
    model.fit(X_train_3d, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model


# ----------------------------------------------------------------------
# Stacked ensemble (meta-learner)
# ----------------------------------------------------------------------
def fit_stacked_ensemble(base_predictions_train: dict, y_train, base_predictions_test: dict):
    """
    Ridge meta-learner over out-of-fold-style base predictions.
    base_predictions_train / base_predictions_test: dict[model_name] -> np.array
    In a strict setup, base_predictions_train should come from cross-validated
    (out-of-fold) predictions on the training set, not in-sample fitted values,
    to avoid the meta-learner overfitting to base-model in-sample accuracy.
    For simplicity here we use a held-out slice of the train window as the
    meta-training set — see train_pipeline.py for how folds are carved out.
    """
    meta_X_train = np.column_stack(list(base_predictions_train.values()))
    meta_X_test = np.column_stack(list(base_predictions_test.values()))

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_X_train, y_train)
    return meta_model.predict(meta_X_test), meta_model