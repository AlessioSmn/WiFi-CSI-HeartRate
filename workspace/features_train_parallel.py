import os
import ast
import gc
import random
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, savgol_filter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore

# =======================
# CONFIG
# =======================

RAW_DATA_PATH = "data/raw_data.csv"                     # dataset path

CSI_DATA_LENGTH = 384                                   # 192 subcarrier × I/Q
N_SUB = CSI_DATA_LENGTH // 2                            # subcarriers

SAMPLING_FREQUENCY = 20                                 # sampling frequency (Hz)
WINDOW_LENGTH = 100                                     # amount of samples per window
LEARNING_RATE = 0.001                                   # learning rate of the lstm model
VAL_THRESHOLD = 0.01                                    # threshold for the early stopping for the validation metric                  
VALIDATION_SPLIT = 0.4                                  # fraction of data to be used for the validation

MODEL_PATH = f"models/csi_hr_{WINDOW_LENGTH}.keras"     # where to save the model
CONTINUE_MODEL = None                                   # path of the model to use for the training continuation
BATCH_SIZE = 64                                         # batch size for the training
TRAIN_FOR_TFLITE = True                                 # compile the model to be compatible with tflite

USE_CHUNKING = True                                     # balance the distribution of the training and validation set via chunking
N_CHUNKS = 20                                           # how many chunks should be created and distributed between the two datasets
DO_BALANCING = False                                    # balance the weights according to the training distribution

# =======================
# TF GPU SAFE CONFIG
# =======================

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# =======================
# SIGNAL UTILITIES
# =======================

def iq_to_complex_matrix(csi_raw_series):
    """
    convert the csi array to the array of I/Q components of each subcarrier
    csi_raw_series: iterable containing the csi array
    returns: ndarray (T, 192) complex64
    """
    valid_rows = []
    for i, row in enumerate(csi_raw_series):
        # check the type and length
        if isinstance(row, list) and len(row) == CSI_DATA_LENGTH:
            valid_rows.append(np.array(row, dtype=np.float32))
        else:
            print(f"Riga {i} scartata, lunghezza={len(row) if isinstance(row, list) else type(row)}")

    if len(valid_rows) == 0:
        raise ValueError("Nessuna riga CSI valida trovata!")

    # forma (T, 384)
    csi = np.stack(valid_rows, axis=0)

    I = csi[:, 0::2]
    Q = csi[:, 1::2]
    return I + 1j * Q



def butter_bandpass_filter(x, fs, lowcut=0.8, highcut=2.17, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, x)


# =======================
# FEATURE EXTRACTION
# =======================

def extract_features(df):
    """
    Compute X and y for the given dataframe
    Return:
        X: (N, W, N_SUBCARRIERS) float32
        y: (N,) float32 or None
    """

    # -----------------------
    # CLEAN INPUT
    # -----------------------
    cols = ["csi_raw", "AVG BPM"]
    df = df[cols].dropna()

    # drop csi arrays of the wrong length
    df = df[df["csi_raw"].apply(lambda x: isinstance(x, list) and len(x) == CSI_DATA_LENGTH)]

    if len(df) < WINDOW_LENGTH:
        return None, None

    # -----------------------
    # CSI → COMPLEX
    # -----------------------
    A_complex = iq_to_complex_matrix(df["csi_raw"])
    A_complex = A_complex.astype(np.complex64)

    # -----------------------
    # AMPLITUDE
    # -----------------------
    A_amp = np.abs(A_complex).astype(np.float32)
    del A_complex
    gc.collect()

    # -----------------------
    # DC REMOVAL (vectorized)
    # -----------------------
    kernel = np.ones(WINDOW_LENGTH, dtype=np.float32) / WINDOW_LENGTH
    mean_dc = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"),
        axis=0,
        arr=A_amp
    )
    A_dc = A_amp - mean_dc
    del A_amp, mean_dc
    gc.collect()

    # -----------------------
    # BANDPASS (parallel)
    # -----------------------
    A_pulse = np.stack(
        Parallel(n_jobs=-1, prefer="processes")(
            delayed(butter_bandpass_filter)(
                A_dc[:, k], SAMPLING_FREQUENCY
            )
            for k in range(N_SUB)
        ),
        axis=1
    ).astype(np.float32)

    del A_dc
    gc.collect()

    # -----------------------
    # SAVITZKY–GOLAY (parallel)
    # -----------------------
    A_smooth = np.stack(
        Parallel(n_jobs=-1)(
            delayed(savgol_filter)(
                A_pulse[:, k], 15, 3
            )
            for k in range(N_SUB)
        ),
        axis=1
    ).astype(np.float32)

    del A_pulse
    gc.collect()

    # -----------------------
    # WINDOWING (zero copy)
    # -----------------------
    X = np.lib.stride_tricks.sliding_window_view(
        A_smooth,
        window_shape=(WINDOW_LENGTH, N_SUB)
    )[:, 0, :, :]

    # -----------------------
    # NORMALIZATION
    # -----------------------
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-6
    X = ((X - mean) / std).astype(np.float32)

    del A_smooth
    gc.collect()

    # -----------------------
    # HR WINDOW
    # -----------------------
    hr = df["AVG BPM"].to_numpy(dtype=np.float32)
    y = np.convolve(hr, np.ones(WINDOW_LENGTH) / WINDOW_LENGTH, mode="valid")
    y = y.astype(np.float32)

    return X, y


def extract_features_with_stratified_chunks(df, chunk_size, train_ratio=0.8):
    """
    Create the training and validation datasets by balancing the distributions using different chunks
    df: raw dataframe
    chunk_size: rows per chunk
    train_ratio: fraction of data for the training phase

    Returns:
        X_train, y_train, X_val, y_val
    """

    # -----------------------
    # DIVIDE IN CONTIGUOUS CHUNKS
    # -----------------------
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        if len(chunk) >= WINDOW_LENGTH:
            chunks.append(chunk)

    # -----------------------
    # MEAN TARGET FOR EACH CHUNK
    # -----------------------
    chunk_stats = []
    for chunk in chunks:
        hr_mean = chunk["AVG BPM"].mean()
        chunk_stats.append((chunk, hr_mean))

    # -----------------------
    # SORT BY MEAN TARGET
    # -----------------------
    chunk_stats.sort(key=lambda x: x[1])

    # -----------------------
    # ASSEGNA CHUNK A TRAIN/VAL IN MODO STRATIFICATO
    # -----------------------
    n_train = int(train_ratio * len(chunk_stats))
    train_chunks = []
    val_chunks = []

    # distribute chunks
    for i, (chunk, _) in enumerate(chunk_stats):
        if i % int(1 / (1 - train_ratio)) == 0:
            val_chunks.append(chunk)
        else:
            train_chunks.append(chunk)

    # -----------------------
    # COMPUTE WINDOWS FOR EACH CHUNK
    # -----------------------
    def process_chunks(chunk_list):
        X_list, y_list = [], []
        for c in chunk_list:
            Xc, yc = extract_features(c)
            if Xc is not None:
                X_list.append(Xc)
                y_list.append(yc)
        if len(X_list) == 0:
            return None, None
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        return X_all, y_all

    X_train, y_train = process_chunks(train_chunks)
    X_val, y_val = process_chunks(val_chunks)

    # check the distribution of the datasets
    print("---------- DATASETS DISTRIBUTION ----------")
    print("train:")
    print(y_train.mean(), y_train.std())
    print("val:")
    print(y_val.mean(), y_val.std())

    return X_train, y_train, X_val, y_val


# =======================
# MAIN
# =======================

if __name__ == "__main__":

    # load raw data
    print("Loading CSV...")
    df = pd.read_csv(RAW_DATA_PATH)
    df["csi_raw"] = df["csi_raw"].apply(ast.literal_eval)

    print("Extracting features...")
    if USE_CHUNKING:
        len_chunk = len(df) // N_CHUNKS
        X_train, y_train, X_val, y_val = extract_features_with_stratified_chunks(df, len_chunk, 0.7)
    else:
        X, y = extract_features(df)
        split = int(0.7 * len(X))
        X_train = X[:split]
        y_train = y[:split]
        X_val = X[split:]
        y_val = y[split:]

    print("X_train:", X_train.shape)
    print("y_train:", None if y_train is None else y_train.shape)
    print("X_train shape:", X_train.shape)
    print("Estimated size (GB):", X_train.nbytes / 1e9)


    # =======================
    # BUILD MODEL
    # =======================
    inputs = keras.Input(shape=(WINDOW_LENGTH, N_SUB))
    if not TRAIN_FOR_TFLITE:
        x = keras.layers.LSTM(64, return_sequences=True)(inputs)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.LSTM(32)(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(16, activation="relu")(x)
    else:
        x = keras.layers.LSTM(64, return_sequences=True, unroll=True)(inputs)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.LSTM(32, unroll=True)(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(1)(x)

    model = None
    if CONTINUE_MODEL is not None:
        print("Resuming training...")
        model = keras.models.load_model(CONTINUE_MODEL)
    else:
        model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(LEARNING_RATE),
        loss="mae",
        metrics=["mae"]
    )
    model.summary()

    # =======================
    # TRAINING
    # =======================

    # save best model during training
    checkpoint = ModelCheckpoint(
        filepath=f"models/csi_hr_best_{WINDOW_LENGTH}.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # stop if for 30 epoch no improvement is made
    early_stop = EarlyStopping(
        monitor='val_mae',  # la metrica che vuoi monitorare
        patience=30,  # quante epoche consecutive senza miglioramento
        restore_best_weights=True
    )

    # stop if the validation metric threshold is reached
    class StopCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs and logs.get("val_loss", 1.0) <= VAL_THRESHOLD:
                print("Reached threshold. Stopping.")
                self.model.stop_training = True

    # compute at training time the rmse converted in BPM
    class RMSEBPMCallback(tf.keras.callbacks.Callback):
        def __init__(self, y_mean, y_std, X_val, y_val_norm):
            super().__init__()
            self.y_mean = y_mean
            self.y_std = y_std
            self.X_val = X_val
            self.y_val_norm = y_val_norm

        def on_epoch_end(self, epoch, logs=None):
            y_pred_norm = self.model.predict(self.X_val, verbose=0)
            # BPM conversion
            y_pred_bpm = y_pred_norm * self.y_std + self.y_mean
            y_true_bpm = self.y_val_norm * self.y_std + self.y_mean
            rmse_bpm = np.sqrt(np.mean((y_pred_bpm - y_true_bpm) ** 2))

            # pearson test
            from scipy.stats import pearsonr
            print(pearsonr(self.y_val_norm.squeeze(), y_pred_norm.squeeze()))
            print(f"Epoch {epoch + 1}: RMSE validation ≈ {rmse_bpm:.3f} BPM, \t mean: {self.y_mean:.3f} std: {self.y_std:.3f}")


    # balancing by weights
    weights = None
    if DO_BALANCING:
        hist, bin_edges = np.histogram(y_train, bins=20)
        bin_indices = np.digitize(y_train, bin_edges[:-1])
        weights = 1.0 / hist[bin_indices - 1]
        weights = weights / np.mean(weights)

    # normalization of y
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-6
    y_norm = (y_train - y_mean) / y_std

    y_val_norm = (y_val - y_mean) / y_std

    # training
    model.fit(
        X_train, y_norm,
        validation_data=(X_val, y_val_norm),
        sample_weight=weights,
        batch_size=BATCH_SIZE,
        epochs=5000,
        validation_split=VALIDATION_SPLIT,
        callbacks=[checkpoint, StopCallback(), early_stop, RMSEBPMCallback(y_mean, y_std, X_val, y_val_norm)],
        verbose=2
    )

    # save final model
    model.save(MODEL_PATH)
    print("Training complete.")
