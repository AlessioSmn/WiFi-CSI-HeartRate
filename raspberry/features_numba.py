import gc
import numpy as np
import numba as nb
from scipy.signal import butter


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
    return (csi[..., 0] + 1j * csi[..., 1]).astype(np.complex64)


# =======================
# NUMBA DSP KERNELS
# =======================

@nb.njit(parallel=True, fastmath=True)
def remove_dc_parallel(A, win):
    """Moving-average DC removal per subcarrier"""
    T, C = A.shape
    out = np.empty_like(A)

    for c in nb.prange(C):
        s = 0.0
        for i in range(T):
            s += A[i, c]
            if i >= win:
                s -= A[i - win, c]

            if i >= win - 1:
                mean = s / win
            else:
                mean = s / (i + 1)

            out[i, c] = A[i, c] - mean

    return out


@nb.njit(parallel=True, fastmath=True)
def bandpass_iir_parallel(A, b, a):
    """Biquad IIR bandpass per subcarrier (Direct Form II)"""
    T, C = A.shape
    out = np.zeros_like(A)

    for c in nb.prange(C):
        z1 = 0.0
        z2 = 0.0

        for n in range(T):
            x = A[n, c]

            y = b[0] * x + z1
            z1 = b[1] * x - a[1] * y + z2
            z2 = b[2] * x - a[2] * y

            out[n, c] = y

    return out


@nb.njit(parallel=True, fastmath=True)
def smooth_parallel(A):
    """FIR smoothing tipo Savitzky–Golay (kernel fisso len=15)"""
    T, C = A.shape
    out = np.empty_like(A)

    k = np.array([
        -3, -2, -1, 0, 1, 2, 3,
        4, 3, 2, 1, 0, -1, -2, -3
    ], dtype=np.float32)

    k = k / np.sum(np.abs(k))
    half = len(k) // 2

    for c in nb.prange(C):
        for i in range(T):
            s = 0.0
            for j in range(len(k)):
                idx = i + j - half
                if idx < 0:
                    idx = 0
                elif idx >= T:
                    idx = T - 1
                s += A[idx, c] * k[j]
            out[i, c] = s

    return out


@nb.njit(fastmath=True)
def normalize_windows(X):
    """Normalizzazione per finestra: X (N, W, C)"""
    N, W, C = X.shape

    for n in range(N):
        mean = 0.0
        std = 0.0

        for i in range(W):
            for c in range(C):
                mean += X[n, i, c]
        mean /= (W * C)

        for i in range(W):
            for c in range(C):
                diff = X[n, i, c] - mean
                std += diff * diff

        std = np.sqrt(std / (W * C)) + 1e-6

        for i in range(W):
            for c in range(C):
                X[n, i, c] = (X[n, i, c] - mean) / std

    return X


# =======================
# BUTTER COEFFICIENTS (outside numba)
# =======================

def butter_coeff(fs, low=0.8, high=2.17, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return b.astype(np.float32), a.astype(np.float32)


# =======================
# FEATURE EXTRACTION
# =======================

def extract_features(df, csi_data_length, sampling_frequency, window_length):
    """
    Return:
        X: (N, W, 192) float32
    """
    n_subcarriers = csi_data_length // 2

    # -----------------------
    # CLEAN INPUT
    # -----------------------
    df = df[["csi_raw"]].dropna()

    if len(df) < window_length:
        return None

    # -----------------------
    # 1. CSI → AMPLITUDE
    # -----------------------
    A_complex = iq_to_complex_matrix(df["csi_raw"], n_subcarriers)
    A_amp = np.abs(A_complex).astype(np.float32)
    del A_complex
    gc.collect()

    # -----------------------
    # 2. DC REMOVAL
    # -----------------------
    A_dc = remove_dc_parallel(A_amp, window_length)
    del A_amp
    gc.collect()

    # -----------------------
    # 3. BANDPASS IIR
    # -----------------------
    b, a = butter_coeff(sampling_frequency)
    A_pulse = bandpass_iir_parallel(A_dc, b, a)
    del A_dc
    gc.collect()

    # -----------------------
    # 4. SMOOTH
    # -----------------------
    A_smooth = smooth_parallel(A_pulse)
    del A_pulse
    gc.collect()

    # -----------------------
    # 5. WINDOWING + NORMALIZATION
    # -----------------------
    X = np.lib.stride_tricks.sliding_window_view(
        A_smooth,
        window_shape=(window_length, n_subcarriers)
    )[:, 0, :, :].astype(np.float32)

    del A_smooth
    gc.collect()

    X = normalize_windows(X)

    return X
