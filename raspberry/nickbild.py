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

# Savitzky-Golay polynomial order
PAR__SG_POLYORDER = 3

# Savitzky-Golay window length
PAR__SG_WINLEN = 15

# Minimum HR
PAR__HR_MIN_BPM = 48

# Maximum HR
PAR__HR_MAX_BPM = 160


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

def savitzky_golay_smooth(signal: np.ndarray,
                          window_length: int = 15,
                          polyorder: int = 3) -> np.ndarray:
    """
    Pulse Shaping: Savitzky-Golay smoothing (preserve waveform shape).
    Ensures window_length is odd and less than signal length.
    If signal too short, returns original signal.
    """
    x = np.asarray(signal, dtype=float).copy()
    n = x.size
    if n == 0:
        return x
    wl = int(window_length)
    if wl % 2 == 0:
        wl += 1
    if wl < (polyorder + 2):
        raise ValueError("window_length too small for polyorder")
    if wl >= n:
        # fallback: choose the largest odd window smaller than n
        wl_candidate = n - 1
        if wl_candidate % 2 == 0:
            wl_candidate -= 1
        if wl_candidate < (polyorder + 2):
            # can't apply SG; return original
            return x
        wl = wl_candidate
    return savgol_filter(x, wl, polyorder)

#endregion

#region ======= Main function =======

def estimate_hr_freq(
        signal_matrix: list[list[float]],
        fs: float,
        hr_min: float = 45,
        hr_max: float = 200,
        par_bp_order: int = 3,
        par_bp_hr_allowance: float = 0.2,
        par_sg_order: int = 3,
        par_sg_winlen: int = 15,
    ) -> float:
    
    DEBUG_ON = True

    hr_min_Hz = hr_min / 60.0
    hr_max_Hz = hr_max / 60.0

    # Convert to numpy
    signal_matrix = np.array(signal_matrix)
    samples, channels = signal_matrix.shape

    # Store processed channels and their powers
    processed_amps = []

    # Process each AMPLITUDE ARRAY
    for amp_array in signal_matrix:

        # Band-pass
        amp_array_bbpf = butter_bandpass_filter(
            signal=amp_array,
            fs=fs,
            lowcut=hr_min_Hz - par_bp_hr_allowance,
            highcut=hr_max_Hz + par_bp_hr_allowance,
            order=par_bp_order
        )

        amp_array_shaped = savitzky_golay_smooth(
            amp_array_bbpf, 
            window_length=par_sg_winlen, 
            polyorder=par_sg_order
        )

        # Store filtered channel
        processed_amps.append(amp_array_shaped)

    # Convert back to np array
    processed_amps = np.array(processed_amps)

    # ---------------------------------------------------------
    # DEBUG PLOT (Equispaced Channels & Zoomed FFT)
    # ---------------------------------------------------------
    if DEBUG_ON:
        # Numero di plot da generare (es. 3: primo, medio, ultimo)
        N_plots = 3
        
        # Se abbiamo meno canali di N_plots, usali tutti
        if channels < N_plots:
            selected_indices = np.arange(channels)
        else:
            # Calcola indici equispaziati (es. 0, 25, 50)
            selected_indices = np.linspace(0, channels - 1, N_plots, dtype=int)

        fig, axes = plt.subplots(len(selected_indices), 2, figsize=(12, 4 * len(selected_indices)))
        
        # Gestione caso singolo plot (array 1D -> 2D)
        if len(selected_indices) == 1: 
            axes = np.expand_dims(axes, axis=0)

        for plot_idx, chan_idx in enumerate(selected_indices):
            
            # --- Recupero Dati ---
            # signal_matrix è (Samples, Subcarriers), processed_channels_T è (Subcarriers, Samples)
            # Prendiamo il segnale FILTRATO (processed)
            shaped_signal = processed_amps[chan_idx, :]

            # --- Calcolo FFT ---
            # Applico finestra per ridurre spectral leakage
            window = np.hanning(len(shaped_signal))
            fft_shaped = np.abs(np.fft.rfft(shaped_signal * window))
            freqs = np.fft.rfftfreq(len(shaped_signal), d=1/fs)

            # Normalizzazione (evita divisione per zero)
            if np.max(fft_shaped) > 0: 
                fft_shaped /= np.max(fft_shaped)

            # --- Plot Time Domain (Sinistra) ---
            ax_time = axes[plot_idx, 0]
            ax_time.plot(shaped_signal, label=f'Subcarrier {chan_idx} (Filt)', color='tab:orange', linewidth=1.5)
            ax_time.set_title(f'Subcarrier {chan_idx} - Time Domain')
            ax_time.set_xlabel('Samples')
            ax_time.set_ylabel('Amplitude')
            ax_time.grid(True, alpha=0.3)
            ax_time.legend(loc='upper right')

            # --- Plot Frequency Domain (Destra) ---
            ax_freq = axes[plot_idx, 1]
            ax_freq.plot(freqs, fft_shaped, label='FFT (Shaped)', color='tab:green')
            
            # Highlight del picco massimo nel range di interesse
            mask = (freqs >= 0.75) & (freqs <= 3.5)
            if np.any(mask):
                max_idx_range = np.argmax(fft_shaped[mask])
                # Converti indice maschera in indice globale
                real_idx = np.where(mask)[0][max_idx_range] 
                peak_freq = freqs[real_idx]
                ax_freq.plot(peak_freq, fft_shaped[real_idx], 'rx', label=f'Peak: {peak_freq*60:.1f} BPM')

            ax_freq.set_title(f'Subcarrier {chan_idx} - Frequency Domain')
            ax_freq.set_xlabel('Frequency (Hz)')
            ax_freq.set_ylabel('Norm. Magnitude')
            
            # --- ZOOM RICHIESTO: 0.75 Hz - 3.5 Hz ---
            ax_freq.set_xlim(0.75, 3.5) 
            ax_freq.grid(True, alpha=0.3, which='both')
            ax_freq.minorticks_on()
            ax_freq.legend(loc='upper right')

        plt.tight_layout()
        plt.show()
        # plt.savefig(f"./img/debug_channels_{timestamp}.jpg")

    return 0.0

#endregion

#region ======= DSP process =======

def PROC_DSP(mpdeque_dsp: MPDeque, mpdeque_hr: MPDeque, log_to_file: bool = False, log_filename: str = "log.log"):

    print_log_loc("Process alive", LVL_INF)
    
    while True:

        # Get csi data window
        csi_data_window = mpdeque_dsp.popright(block=True)

        print_log_loc("Processing csi data window", LVL_DBG)

        # Estimate HR
        hr_hz = estimate_hr_freq(
            signal_matrix=csi_data_window,
            fs=PAR_SAMP_FREQ_HZ,
            hr_min=PAR__HR_MIN_BPM,
            hr_max=PAR__HR_MAX_BPM,
            par_bp_order=PAR__BP_ORDER,
            par_bp_hr_allowance=PAR__BP_ALLOW_HZ,
            par_sg_winlen=PAR__SG_WINLEN,
            par_sg_order=PAR__SG_POLYORDER
        )

        # Enqueue
        mpdeque_hr.appendleft(hr_hz)

        print_log_loc("Estimate calculated and returned", LVL_DBG)

#endregion