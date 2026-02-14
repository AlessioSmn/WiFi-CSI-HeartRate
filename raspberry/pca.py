#region ======= Imports =======

import os
import time
import numpy as np
from mpque import MPDeque
import matplotlib.pyplot as plt
from scipy.stats import trim_mean
from scipy.fft import rfft, rfftfreq
from log import print_log, LVL_DBG, LVL_INF, LVL_ERR, DebugLevel
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks, welch
from scipy.signal import detrend

from collections import deque

#endregion

#region ======= Debug and utility function =======

def Hz_to_BPM(hz: float) -> float:
    return hz * 60.0

def print_log_loc(s: str, log_level: DebugLevel = DebugLevel.INFO):
    s_loc = f"[DSP ] {s}"
    print_log(s_loc, log_level)

def save_spectrum(signal, fs, chan_idx, stage, kind="fft", timestamp=0):
    N = len(signal)

    plt.figure()

    if kind == "fft":
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(N, d=1/fs)
        mag = np.abs(fft_vals) / N
        plt.plot(freqs, mag)
        ylabel = "Amplitude"

    elif kind == "psd":
        nperseg = min(1024, len(signal))
        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

        if np.all(psd <= 0):
            plt.plot(freqs, psd)
            ylabel = "PSD (zero)"
        else:
            plt.semilogy(freqs, psd)
            ylabel = "PSD [V²/Hz]"

    else:
        return

    plt.xlabel("Frequency [Hz]")
    plt.ylabel(ylabel)
    plt.xlim(0, 5)

    # Vertical light-gray lines every 0.5 Hz
    for x in np.arange(0.5, 5.0, 0.5):
        plt.axvline(x, color="0.85", linewidth=0.8)

    plt.grid(True, axis="y")

    fname = f"./img/{stage}/{kind}/{timestamp}_chan{chan_idx}_{stage}_{kind}.jpg"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

def debug_spectrums(original_matrix, processed_matrix, fs, top_idx=None):

    os.makedirs("./img", exist_ok=True)
    os.makedirs("./img/proc", exist_ok=True)
    os.makedirs("./img/proc/fft", exist_ok=True)
    os.makedirs("./img/proc/psd", exist_ok=True)
    os.makedirs("./img/raw", exist_ok=True)
    os.makedirs("./img/raw/fft", exist_ok=True)
    os.makedirs("./img/raw/psd", exist_ok=True)
    timestamp = int(time.time())

    original_matrix = np.asarray(original_matrix)
    processed_matrix = np.asarray(processed_matrix)

    _, channels = original_matrix.shape

    if top_idx is None:
        chan_list = range(channels)
    else:
        chan_list = top_idx

    log_path = f"./img/log_{timestamp}.txt"
    with open(log_path, "w") as f:
        f.write("channel_idx,power_raw,power_proc\n")

        for i in chan_list:
            raw = original_matrix[:, i]
            proc = processed_matrix[:, i]

            # Spettri RAW
            save_spectrum(raw, fs, i, stage="raw", kind="fft", timestamp=timestamp)
            save_spectrum(raw, fs, i, stage="raw", kind="psd", timestamp=timestamp)

            # Spettri PROCESSED
            save_spectrum(proc, fs, i, stage="proc", kind="fft", timestamp=timestamp)
            save_spectrum(proc, fs, i, stage="proc", kind="psd", timestamp=timestamp)

            power_raw = float(np.mean(raw ** 2))
            power_proc = float(np.mean(proc ** 2))

            f.write(f"{i},{power_raw:.6e},{power_proc:.6e}\n")

    print(f"[DEBUG] Spectrums saved in ./img/**")
    print(f"[DEBUG] Log: {log_path}")

#endregion

#region ======= Parameters =======

# Sampling frequency
PAR_SAMP_FREQ_HZ = 40

# Bandpass filter order
PAR__BP_ORDER = 3

# Bandpass filter extra margin
PAR__BP_ALLOW_HZ = 0

PAR__PCA_CARRIERS = 3

# Minimum HR
PAR__HR_MIN_BPM = 50

# Maximum HR
PAR__HR_MAX_BPM = 200

#endregion

#region ======= CSI Data functions =======

def resample_window(csi_data_window_counters):
    """
    Resample a CSI window so that samples are aligned on a continuous counter axis.

    The function reconstructs missing samples by linear interpolation.
    If some counters are missing in the input sequence, they are detected,
    printed to stdout, and filled accordingly.

    Parameters
    ----------
    csi_data_window_counters : list of tuples
        Each element is a tuple:
            (counter, csi_vector)

        - counter (int): monotonically increasing sample counter
        - csi_vector (array-like): CSI values for all subcarriers

    Returns
    -------
    np.ndarray
        Rebuilt CSI matrix with shape:
            (number_of_expected_samples, number_of_subcarriers)

        Missing samples are reconstructed via linear interpolation.
    """

    # Extract counters and CSI signals
    counters = np.array([x[0] for x in csi_data_window_counters])
    signals = np.array([x[1] for x in csi_data_window_counters])

    # Build the expected continuous counter sequence
    full_counters = np.arange(
        counters[0],
        counters[-1] + 1
    )

    # Detect missing counters
    num_missing_counters = len(full_counters) - len(counters)
    # missing_counters = np.setdiff1d(full_counters, counters)

    # Print missing counters, if any
    if num_missing_counters > 0:
        print_log_loc(f"Resampling: {num_missing_counters} missing counters", LVL_DBG)

    # Number of samples after reconstruction
    n_samples = len(full_counters)

    # Number of CSI subcarriers
    n_subcarr = signals.shape[1]

    # Allocate rebuilt CSI matrix
    rebuilt = np.zeros((n_samples, n_subcarr))

    # Interpolate each subcarrier independently
    for ch in range(n_subcarr):
        rebuilt[:, ch] = np.interp(
            full_counters,   # target counters
            counters,        # available counters
            signals[:, ch]   # known CSI samples
        )

    return rebuilt

def parse_csi_amplitudes(values):
    """
    Given a flat list of ints [re0, im0, re1, im1, ...],
    returns a numpy array of amplitudes.

    :param values: list of ints
    :return: numpy array of amplitudes
    """
    if len(values) % 2 != 0:
        raise ValueError("CSI values must have even length (pairs of re/im)")

    # Reshape into (imag, real) pairs
    complex_pairs = np.array(values).reshape(-1, 2)

    # Convert to complex numbers: real + j*imag
    csi_complex = complex_pairs[:,1] + 1j * complex_pairs[:,0]

    # Compute amplitudes
    amplitudes = np.abs(csi_complex)
    return amplitudes

#endregion

#region ======= DSP functions (single step) =======

def butter_bandpass_filter(
        signal: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: float,
        order: int = 3
    ) -> np.ndarray:
    """
    Pulse Extraction: 3rd-order Butterworth bandpass (default order=3).
    Uses zero-phase filtering (filtfilt).
    lowcut/highcut in Hz. fs is sampling frequency (Hz).
    """
    x = np.asarray(signal, dtype=float).copy()
    if x.size == 0:
        return x
    nyq = 0.5 * fs
    if not (0 < lowcut < highcut < nyq):
        raise ValueError(f"Invalid bandpass: low={lowcut}, high={highcut}, Nyquist={nyq}")
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    # filtfilt for zero-phase
    return filtfilt(b, a, x)

def get_autocorr_estimation(signal, fs, hr_min, hr_max):
    """
    Estimates Heart Rate using Autocorrelation method.
    It looks for the first major peak in the autocorrelation function 
    within the lag range corresponding to hr_min and hr_max.
    """
    # Calculate autocorrelation
    # We use 'full' mode and take the second half
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]
    
    # Define lag search range
    # Lag = fs / frequency
    min_lag = int(fs / (hr_max / 60.0))
    max_lag = int(fs / (hr_min / 60.0))
    
    # Safety check
    if max_lag >= len(corr):
        max_lag = len(corr) - 1
        
    # Extract the region of interest
    roi = corr[min_lag:max_lag]
    lags = np.arange(min_lag, max_lag)
    
    if len(roi) == 0:
        return 0.0, 0.0
    
    # Find peaks in the autocorrelation ROI
    peaks, props = find_peaks(roi, height=0)
    
    if len(peaks) == 0:
        # If no distinct peaks, take the max (fallback)
        best_idx = np.argmax(roi)
    else:
        # Take the highest peak in the ROI
        best_idx = peaks[np.argmax(props['peak_heights'])]
        
    best_lag = lags[best_idx]
    
    # Convert lag to Hz
    est_freq = fs / best_lag
    confidence = roi[best_idx] / (corr[0] + 1e-9) # Normalize by energy at lag 0
    
    return est_freq, confidence

#endregion

#region ======= Main Function =======

def estimate_hr_freq_PCA_STORY(
        signal_matrix: list[list[float]],
        fs: float,
        hr_history: list[float],
        top_components: int = 5,
        hr_min: float = 45,
        hr_max: float = 200,
        par_bp_order: int = 4,
        par_bp_hr_allowance: float = 0.0
    ) -> float:
    
    DEBUG_ON = True 
    timestamp = int(time.time())

    hr_min_Hz = hr_min / 60.0
    hr_max_Hz = hr_max / 60.0

    # Convert list to numpy array
    signal_matrix = np.array(signal_matrix)

    # --- 1. Preprocessing ---
    processed_channels = []
    
    for chan in signal_matrix.T:
        # Remove linear trend
        chan_detrended = detrend(chan)
        
        # Bandpass filter
        # Note: If HR is expected to be high, ensure lowcut is > 0.8 Hz to remove breathing harmonics
        chan = butter_bandpass_filter(
            signal=chan_detrended,
            fs=fs,
            lowcut=hr_min_Hz - par_bp_hr_allowance,
            highcut=hr_max_Hz + par_bp_hr_allowance,
            order=par_bp_order
        )
        processed_channels.append(chan)

    # --- 2. PCA (Dimensionality Reduction) ---
    X = np.array(processed_channels).T
    
    # Standardization
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0 # Avoid div by zero
    X_centered = (X - X_mean) / X_std

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    n_components = min(top_components, 5, U.shape[1])
    pca_components = U[:, :n_components]

    # --- 3. Method A: Spectral Analysis (FFT) ---
    n_padded = 8192 
    avg_spectrum = np.zeros(n_padded // 2 + 1)
    
    # Start from index 0. Filtered signal shouldn't have breathing dominance.
    start_comp_idx = 0 

    for i in range(start_comp_idx, n_components):
        comp_signal = pca_components[:, i]
        
        # Hanning Window to reduce leakage
        window = np.hanning(len(comp_signal))
        fft_vals = np.abs(np.fft.rfft(comp_signal * window, n=n_padded))
        
        if np.max(fft_vals) > 0:
            fft_vals = fft_vals / np.max(fft_vals)
        avg_spectrum += fft_vals

    freqs = np.fft.rfftfreq(n_padded, d=1/fs)
    mask = (freqs >= hr_min_Hz) & (freqs <= hr_max_Hz)
    valid_freqs = freqs[mask]
    valid_spectrum = avg_spectrum[mask]

    fft_freq = 0.0
    if len(valid_spectrum) > 0:
        # Use simple argmax for FFT base candidate
        # We refine this logic below with history, but first we need a raw candidate
        peak_idx = np.argmax(valid_spectrum)
        fft_freq = valid_freqs[peak_idx]

    # --- 4. Method B: Autocorrelation (Time Domain) ---
    # We apply autocorrelation on the first valid PCA component (most energetic)
    # or on a weighted sum of components. Here we use PC0 for simplicity.
    
    # Use Autocorrelation on the first PCA component as a cross-check
    autocorr_freq, autocorr_conf = get_autocorr_estimation(
        pca_components[:, 0], fs, hr_min, hr_max
    )

    # --- 5. HYBRID DECISION LOGIC ---
    
    final_freq = 0.0
    
    bpm_fft = fft_freq * 60
    bpm_auto = autocorr_freq * 60
    
    diff_bpm = abs(bpm_fft - bpm_auto)
    
    # Threshold for agreement (e.g., 12 BPM)
    agreement_threshold = 12.0 
    
    is_agreement = diff_bpm < agreement_threshold
    
    # Retrieve history reference
    ref_hr_hz = np.median(hr_history) if len(hr_history) > 0 else 0.0
    ref_bpm = ref_hr_hz * 60
    
    # print(f"[DSP] FFT: {bpm_fft:.1f} | Auto: {bpm_auto:.1f} (Conf: {autocorr_conf:.2f}) | Diff: {diff_bpm:.1f}")

    if is_agreement:
        # High confidence: both methods agree
        # Average them for robustness
        final_freq = (fft_freq + autocorr_freq) / 2.0
    else:
        # Disagreement: Resolve using History or Confidence
        if ref_hr_hz > 0:
            # Check which one is closer to history
            dist_fft = abs(bpm_fft - ref_bpm)
            dist_auto = abs(bpm_auto - ref_bpm)
            
            if dist_fft < dist_auto:
                final_freq = fft_freq
            else:
                final_freq = autocorr_freq
        else:
            # No history: Trust Autocorrelation if confidence is high, else FFT
            # Autocorrelation is generally more robust to "fake" harmonics than FFT
            if autocorr_conf > 0.5:
                final_freq = autocorr_freq
            else:
                final_freq = fft_freq

    # --- DEBUG PLOT ---
    if DEBUG_ON:
        os.makedirs("./img/decision", exist_ok=True)
        plt.figure(figsize=(10, 5))
        
        # Plot FFT Spectrum
        plt.plot(valid_freqs * 60, valid_spectrum / np.max(valid_spectrum), label="FFT Spectrum", color='blue')
        
        # Plot vertical lines for estimates
        plt.axvline(bpm_fft, color='cyan', linestyle='--', label=f"FFT Cand: {bpm_fft:.1f}")
        plt.axvline(bpm_auto, color='orange', linestyle='--', label=f"Auto Cand: {bpm_auto:.1f}")
        
        if ref_hr_hz > 0:
            plt.axvline(ref_bpm, color='green', linewidth=2, alpha=0.5, label=f"History: {ref_bpm:.1f}")
            
        plt.axvline(final_freq * 60, color='red', linewidth=3, label=f"FINAL: {final_freq*60:.1f}")
        
        plt.title(f"Hybrid Decision (Diff: {diff_bpm:.1f} BPM)")
        plt.xlabel("BPM")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./img/decision/{timestamp}_hybrid.jpg")
        plt.close()

    return final_freq

#endregion

#region ======= DSP process =======

def PROC_DSP(mpdeque_dsp: MPDeque, mpdeque_hr: MPDeque, log_to_file: bool = False, log_filename: str = "log.log"):

    print_log_loc("Process alive", LVL_INF)
    
    if log_to_file:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_filename, "a") as f:
            f.write(f"\n[{timestamp}] NEW SESSION\n")
            f.write(f"Parameters: BP_Order={PAR__BP_ORDER}, PCA={PAR__PCA_CARRIERS}\n")

    # --- History Buffer ---
    # Maxlen 10 significa circa 10 secondi di memoria (se 1 stima/sec).
    # Se il cuore cambia drasticamente, ci vorranno ~5 secondi per adattarsi.
    history_buffer = deque(maxlen=10) 

    while True:
        csi_data_window = mpdeque_dsp.popright(block=True)
        print_log_loc("Processing csi data window", LVL_DBG)

        # Convertiamo il deque in lista per passarlo alla funzione (che usa numpy)
        history_list = list(history_buffer)

        new_hr_hz = estimate_hr_freq_PCA_STORY(
            signal_matrix=csi_data_window,
            fs=PAR_SAMP_FREQ_HZ,
            hr_history=history_list, # <--- Passiamo la lista
            top_components=PAR__PCA_CARRIERS,
            hr_min=PAR__HR_MIN_BPM,
            hr_max=PAR__HR_MAX_BPM,
            par_bp_order=PAR__BP_ORDER,
            par_bp_hr_allowance=PAR__BP_ALLOW_HZ
        )

        if new_hr_hz > 0:
            # Aggiungiamo al buffer. Se è pieno, il più vecchio viene rimosso automaticamente.
            history_buffer.append(new_hr_hz)
            
            # --- Output Smoothing (Opzionale) ---
            # Per l'output finale (mpdeque_hr) possiamo mandare la media smussata
            # oppure il valore raw appena calcolato. 
            # Mandare la mediana del buffer è ultra-stabile.
            mean_hr = np.mean(history_buffer)
            mediam_hr = np.median(history_buffer)
                
            mpdeque_hr.appendleft(mean_hr)
            
            print(f"Est: {new_hr_hz*60:.1f} BPM \t/ Mean: {mean_hr*60:.1f} \t/Median: {mediam_hr*60:.1f}")
            
            if log_to_file:
                with open(log_filename, "a") as f:
                    f.write(f"{Hz_to_BPM(mean_hr):.2f} BPM / {mean_hr:.3f} Hz\n")
#endregion