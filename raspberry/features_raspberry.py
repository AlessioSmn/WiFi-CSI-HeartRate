import gc
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter


# =======================
# SIGNAL UTILITIES
# =======================

def iq_to_complex_matrix(csi_raw_series, n_sub):
    """
    csi_raw_series: iterable di liste lunghezza 384
    restituisce: ndarray (T, 192) complesso64
    """
    csi = np.stack(csi_raw_series.to_numpy(), axis=0).astype(np.float32)
    csi = csi.reshape(-1, n_sub, 2)
    return csi[..., 0] + 1j * csi[..., 1]



def butter_bandpass_filter(x, fs, lowcut=0.8, highcut=2.17, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, x)


# =======================
# FEATURE EXTRACTION
# =======================

def extract_features(df, csi_data_length, sampling_frequency, window_length):
    """
    Return:
        X: (N, W, 192) float32
        y: (N,) float32 or None
    """
    n_subcarriers = csi_data_length // 2

    # -----------------------
    # CLEAN INPUT
    # -----------------------
    df = df[["csi_raw"]].dropna()

    if len(df) < window_length:
        return None


    # PULSE-FI PROCESSING

    # -----------------------
    # 1. CSI → COMPLEX → AMPLITUDE
    # -----------------------
    A_complex = iq_to_complex_matrix(df["csi_raw"], n_subcarriers)
    A_complex = A_complex.astype(np.complex64)
    A_amp = np.abs(A_complex).astype(np.float32)
    del A_complex
    gc.collect()

    # -----------------------
    # 2. DC REMOVAL (vectorized)
    # -----------------------
    kernel = np.ones(window_length, dtype=np.float32) / window_length
    mean_dc = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="same"),
        axis=0,
        arr=A_amp
    )
    A_dc = A_amp - mean_dc

    del A_amp, mean_dc
    gc.collect()

    # -----------------------
    # 3. BANDPASS (parallel)
    # -----------------------
    A_pulse = butter_bandpass_filter(A_dc, sampling_frequency)

    del A_dc
    gc.collect()

    # -----------------------
    # 4. SAVITZKY–GOLAY (parallel)
    # -----------------------
    A_smooth = savgol_filter(A_pulse, 15, 3)

    del A_pulse
    gc.collect()

    # -----------------------
    # 5. WINDOWING (zero copy) → NORMALIZATION
    # -----------------------
    X = np.lib.stride_tricks.sliding_window_view(
        A_smooth,
        window_shape=(window_length, n_subcarriers)
    )[:, 0, :, :]

    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-6
    X = ((X - mean) / std).astype(np.float32)

    del A_smooth
    gc.collect()

    return X
