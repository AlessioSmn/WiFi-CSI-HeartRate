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

# Aggregation method
PAR__AGGR_METHOD = 'mean'

# Bandpass filter order
PAR__BP_ORDER = 3

# Bandpass filter extra margin
PAR__BP_ALLOW_HZ = 0

""" SG Non usato
# Savitzky-Golay polynomial order
PAR__SG_POLYORDER = 4
# Savitzky-Golay window length
PAR__SG_WINLEN = 5
"""

# Number of carriers to use (by power in band)
PAR__TOP_CARRIERS = 20

PAR__PCA_CARRIERS = 3

# Minimum HR
PAR__HR_MIN_BPM = 50

# Maximum HR
PAR__HR_MAX_BPM = 200

# Minimum width for a peak to be selected (after FFT)
PAR__FT_BAND_WIDTH = 0.075 # Hz

# Percentage of peak power
PAR__FT_BAND_THRESH = 0.3

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

def dominant_wide_frequency(
        signal: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: float,
        min_bw_hz: float = 0.0,
        threshold: float = 0.5
    ) -> float:
    """
    Compute the dominant frequency of a real-valued signal within a specified frequency band,
    using integrated energy over bandwidth to penalize narrow spurious peaks.

    :param signal: Input time-domain signal (1D array of samples).
    :type signal: np.ndarray
    :param lowcut: Lower bound of the frequency band (Hz). Frequencies below this are ignored.
    :type lowcut: float
    :param highcut: Upper bound of the frequency band (Hz). Frequencies above this are ignored.
    :type highcut: float
    :param fs: Sampling frequency of the signal (Hz).
    :type fs: float
    :param min_bw_hz: Minimum bandwidth (Hz) required to consider a peak valid.
    :type min_bw_hz: float
    :param threshold: Percentage of the peak power to be considered the same band.
    :type threshold: float
    :return: Dominant frequency (Hz) within the specified band.
    :rtype: float
    """

    # === Step 0: Calculate number of samples
    samples = len(signal)

    # === Step 1: Frequency axis
    freqs = rfftfreq(samples, d=1/fs)

    # Mask frequency band
    mask = (freqs >= lowcut) & (freqs <= highcut)
    freq_band = freqs[mask]

    # === Step 2: FFT magnitude
    fft_vals = np.abs(rfft(signal))
    fft_band = fft_vals[mask]

    # Fallback: identical to classic version
    if min_bw_hz <= 0:
        return freq_band[np.argmax(fft_band)]

    # Frequency resolution
    df = freq_band[1] - freq_band[0]

    # Sort indices by descending amplitude
    sorted_idx = np.argsort(fft_band)[::-1]

    # === Step 3: Reject only too-narrow peaks
    for i in sorted_idx:
        peak_val = fft_band[i]

        # Left bound
        l = i
        while l > 0 and fft_band[l] > threshold:
            l -= 1

        # Right bound
        r = i
        while r < len(fft_band) - 1 and fft_band[r] > threshold:
            r += 1

        bandwidth = (r - l) * df
        if bandwidth >= min_bw_hz:
            return freq_band[i]

    # Fallback: highest peak
    return freq_band[np.argmax(fft_band)]

#endregion

#region ======= Main function =======

def estimate_hr_freq_PCA(
        signal_matrix: list[list[float]],
        fs: float,
        top_components: int = 10,
        hr_min: float = 45,
        hr_max: float = 200,
        par_bp_order: int = 3,
        par_bp_hr_allowance: float = 0.2
    ) -> float:
    
    """
    Estimate the heart rate (HR) from multi-channel CSI amplitude data.

    The function performs the following steps:
    1. Band-pass filtering of each channel within an extended HR range.
    2. Savitzky-Golay smoothing to reduce noise while preserving waveform shape.
    3. Selection of the top channels based on signal power.
    4. Extraction of the dominant frequency for each selected channel.
    5. Calculation of HR in beats per minute (BPM) as the mean of selected channels.

    :param signal_matrix: 2D array of CSI amplitudes, shape (samples, subcarriers). 
                          Each row corresponds to a time sample, each column to a subcarrier.
    :type signal_matrix: list[list[float]]
    :param fs: Sampling frequency of the signal in Hz.
    :type fs: float
    :param top_carriers: Number of strongest subcarriers to use for HR estimation. 
                         Default is 10.
    :type top_carriers: int
    :param aggr_method: Aggregation method for the extracted frequencies.
                         Can be one of 'mean', 'median', 'trimmed'. Default is 'mean'.
    :type aggr_method: str
    :param hr_min: Minimum expected heart rate in BPM. Default is 45 BPM.
    :type hr_min: float
    :param hr_max: Maximum expected heart rate in BPM. Default is 200 BPM.
    :type hr_max: float
    :param par_bp_order: order of Butterworth bandpass filter. Default is 3.
    :type par_bp_order: int
    :param par_bp_hr_allowance:  hertz of additional allowance of Butterworth bandpass filter. Default is 0.2 Hertz.
    :type par_bp_hr_allowance: float
    :param par_sg_order: polyorder of Saviztky-Golay filter. Default is 3.
    :type par_sg_order: int
    :param par_sg_winlen: window length of Saviztky-Golay filter. Default is 15.
    :type par_sg_winlen: int
    :param par_ft_band: Minimum bandwidth to select a peak. Default is 0.1 Hz
    :type par_ft_band: float
    :param par_ft_thresh: Percentage of the peak power to be considered in the same band. Default is 0.5 (50%)
    :type par_ft_thresh: float
    :return: Estimated heart rate in Hertz, averaged over selected channels.
    :rtype: float
    """
    DEBUG_ON = True
    timestamp = int(time.time())

    hr_min_Hz = hr_min / 60.0
    hr_max_Hz = hr_max / 60.0

    # Convert to numpy
    signal_matrix = np.array(signal_matrix)

    # Store processed channels and their powers
    processed_channels = []

    # Process each channel
    for chan in signal_matrix.T:
        # 1. Detrending
        chan_detrended = detrend(chan)
        
        # 3. Band-pass
        chan = butter_bandpass_filter(
            signal=chan_detrended,
            fs=fs,
            lowcut=hr_min_Hz - par_bp_hr_allowance,
            highcut=hr_max_Hz + par_bp_hr_allowance,
            order=par_bp_order
        )

        # Store filtered channel
        processed_channels.append(chan)

    # --- 2. PCA (Principal Component Analysis) via SVD ---

    X = np.array(processed_channels).T 
    
    # Standardizzazione
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0 
    X_centered = (X - X_mean) / X_std

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Prendiamo le prime componenti
    n_components = min(top_components, U.shape[1])
    pca_components = U[:, :n_components]

    # --- 3. SPECTRAL AVERAGING ---
    
    n_padded = 8192
    avg_spectrum = np.zeros(n_padded // 2 + 1)
    
    # Ignoriamo la prima componente (indice 0) perché contiene quasi sempre solo respiro e movimento macroscopico.
    start_comp_idx = 1 if n_components > 1 else 0

    for i in range(start_comp_idx, n_components):
        comp_signal = pca_components[:, i]
        
        # --- DEBUG: Salva lo spettro di ogni componente PCA ---
        if DEBUG_ON:
            # Creiamo le cartelle se non esistono (la save_spectrum lo fa per i canali, ma qui usiamo "pca")
            os.makedirs("./img/pca/fft", exist_ok=True)
            save_spectrum(comp_signal, fs, chan_idx=i, stage="pca", kind="fft", timestamp=timestamp)
        
        # FFT
        fft_vals = np.abs(np.fft.rfft(comp_signal, n=n_padded))
        
        # Normalizzazione: impedisce che una componente rumorosa domini
        peak_val = np.max(fft_vals)
        if peak_val > 0:
            fft_vals = fft_vals / peak_val
            
        avg_spectrum += fft_vals

    # --- 4. STIMA FREQUENZA ---
    
    freqs = np.fft.rfftfreq(n_padded, d=1/fs)
    
    mask = (freqs >= hr_min_Hz) & (freqs <= hr_max_Hz)
    valid_freqs = freqs[mask]
    valid_spectrum = avg_spectrum[mask]

    if len(valid_spectrum) == 0:
        return 0.0
    
    # Frequency Weighted Peak Search
    #weighted_spectrum = valid_spectrum * (valid_freqs ** 0.5)
    weighted_spectrum = valid_spectrum

    # --- PEAK FINDING (Cruciale) ---
    # Invece di argmax (che prende il punto più alto in assoluto, anche se è un picco di rumore isolato),
    # usiamo find_peaks per trovare "colline" reali.
    
    # Normalizziamo lo spettro tra 0 e 1 per facilitare i parametri
    if np.max(weighted_spectrum) > 0:
        weighted_spectrum /= np.max(weighted_spectrum)

    # Prominence: quanto il picco "svetta" rispetto alla valle vicina.
    # Un picco cardiaco reale ha una forma definita. Il rumore è frastagliato.
    peaks, properties = find_peaks(weighted_spectrum, prominence=0.05, height=0.1)
    
    if len(peaks) > 0:
        # Se troviamo picchi validi, prendiamo quello con l'ampiezza (o prominenza) maggiore
        # Preferiamo la prominenza per robustezza contro il rumore 1/f residuo
        best_peak_idx = peaks[np.argmax(properties["prominences"])]
        best_freq = valid_freqs[best_peak_idx]
    else:
        # Fallback se non troviamo picchi chiari: torniamo al massimo assoluto
        peak_idx = np.argmax(weighted_spectrum)
        best_freq = valid_freqs[peak_idx]
    
    # --- DEBUG: PLOT FINALE DECISIVO ---
    if DEBUG_ON:
        os.makedirs("./img/decision", exist_ok=True)
        plt.figure()
        plt.plot(valid_freqs * 60, weighted_spectrum)
        plt.axvline(best_freq * 60, color='r', linestyle='--')
        plt.title(f"Decision: {best_freq*60:.1f} BPM")
        plt.xlabel("BPM")
        plt.grid(True)
        plt.savefig(f"./img/decision/{timestamp}_final.jpg")
        plt.close()
    
    return best_freq

def estimate_hr_freq_PCA_STORY(
        signal_matrix: list[list[float]],
        fs: float,hr_history: list[float],
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

    # Convert to numpy
    signal_matrix = np.array(signal_matrix)

    # --- 1. Preprocessing ---
    processed_channels = []
    for chan in signal_matrix.T:
        chan_detrended = detrend(chan)
        
        
        chan = butter_bandpass_filter(
            signal=chan_detrended,
            fs=fs,
            lowcut=max(0.1, hr_min_Hz - par_bp_hr_allowance),
            highcut=hr_max_Hz + par_bp_hr_allowance,
            order=par_bp_order
        )
        processed_channels.append(chan)

    # --- 2. PCA ---
    X = np.array(processed_channels).T 
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0 
    X_centered = (X - X_mean) / X_std

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    n_components = min(top_components, 5, U.shape[1])
    pca_components = U[:, :n_components]

    # --- 3. Spectral Averaging ---
    n_padded = 8192 
    avg_spectrum = np.zeros(n_padded // 2 + 1)
    
    # Ignora la prima componente (respiro forte)
    start_comp_idx = 1 if n_components > 1 else 0

    for i in range(start_comp_idx, n_components):
        comp_signal = pca_components[:, i]
        fft_vals = np.abs(np.fft.rfft(comp_signal, n=n_padded))
        if np.max(fft_vals) > 0:
            fft_vals = fft_vals / np.max(fft_vals)
        avg_spectrum += fft_vals

    # --- 4. Analisi Spettro Finale ---
    freqs = np.fft.rfftfreq(n_padded, d=1/fs)
    mask = (freqs >= hr_min_Hz) & (freqs <= hr_max_Hz)
    valid_freqs = freqs[mask]
    valid_spectrum = avg_spectrum[mask]

    if len(valid_spectrum) == 0:
        return 0.0
    
    # Pesatura soft per favorire le alte frequenze (come prima)
    weighted_spectrum = valid_spectrum * (valid_freqs ** 0.5)
    # weighted_spectrum = valid_spectrum
    
    if np.max(weighted_spectrum) > 0:
        weighted_spectrum /= np.max(weighted_spectrum)

    
    # --- 5. PEAK TRACKING CON STORICO (History) ---
    
    peaks, properties = find_peaks(weighted_spectrum, prominence=0.03, height=0.1)
    best_freq = 0.0
    best_peak_idx = 0
    ref_hr_hz = 0.0
    
    if len(peaks) > 0:
        cand_freqs = valid_freqs[peaks]
        cand_proms = properties["prominences"]
        
        if len(hr_history) > 0:
            ref_hr_hz = np.median(hr_history)
            
            # Sigma base
            if len(hr_history) > 3:
                hist_std = np.std(hr_history)
                sigma = max(0.4, hist_std * 2) 
            else:
                sigma = 0.6 

            scores = []
            for i, f in enumerate(cand_freqs):
                dist = abs(f - ref_hr_hz)
                
                # --- LOGICA HARMONIC JUMP ---
                # Se il candidato è circa il DOPPIO della storia (es. Storia=70, Cand=140),
                # è molto probabile che la storia fosse agganciata a un'armonica inferiore.
                # In questo caso, IGNORIAMO la distanza.
                is_double = (1.8 * ref_hr_hz) <= f <= (2.2 * ref_hr_hz)
                
                if is_double:
                    # Bonus massiccio: trattalo come se fosse vicino alla media
                    weight = 1.0 
                else:
                    weight = np.exp(- (dist**2) / (2 * sigma**2))
                
                # Score finale
                scores.append(cand_proms[i] * (weight + 0.3))
            
            best_idx = np.argmax(scores)
        else:
            best_idx = np.argmax(cand_proms)
            
        best_freq = cand_freqs[best_idx]
        best_peak_idx = peaks[best_idx]
        
    else:
        peak_idx = np.argmax(weighted_spectrum)
        best_freq = valid_freqs[peak_idx]
        best_peak_idx = peak_idx
        ref_hr_hz = np.median(hr_history) if len(hr_history) > 0 else 0.0

    # --- DEBUG PLOT ---
    if DEBUG_ON:
        os.makedirs("./img/decision", exist_ok=True)
        plt.figure(figsize=(10, 5))
        plt.plot(valid_freqs * 60, weighted_spectrum, label="Spectrum (Linear Weighted)")
        
        if ref_hr_hz > 0:
            plt.axvline(ref_hr_hz * 60, color='green', linestyle='--', linewidth=2, label=f"Hist: {ref_hr_hz*60:.1f}")

        plt.plot(best_freq * 60, weighted_spectrum[best_peak_idx], 'rx', markersize=12, markeredgewidth=3, label=f"Est: {best_freq*60:.1f}")
        
        plt.title(f"Decision: {best_freq*60:.1f} BPM")
        plt.xlabel("BPM")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"./img/decision/{timestamp}_linear.jpg")
        plt.close()

    return best_freq

def estimate_hr_freq(
        signal_matrix: list[list[float]],
        fs: float,
        top_carriers: int = 10,
        aggr_method: str = 'mean',
        hr_min: float = 45,
        hr_max: float = 200,
        par_bp_order: int = 3,
        par_bp_hr_allowance: float = 0.2,
        par_sg_order: int = 3,
        par_sg_winlen: int = 15,
        par_ft_band: float = 0.1,
        par_ft_thresh: float = 0.5
    ) -> float:
    
    """
    Estimate the heart rate (HR) from multi-channel CSI amplitude data.

    The function performs the following steps:
    1. Band-pass filtering of each channel within an extended HR range.
    2. Savitzky-Golay smoothing to reduce noise while preserving waveform shape.
    3. Selection of the top channels based on signal power.
    4. Extraction of the dominant frequency for each selected channel.
    5. Calculation of HR in beats per minute (BPM) as the mean of selected channels.

    :param signal_matrix: 2D array of CSI amplitudes, shape (samples, subcarriers). 
                          Each row corresponds to a time sample, each column to a subcarrier.
    :type signal_matrix: list[list[float]]
    :param fs: Sampling frequency of the signal in Hz.
    :type fs: float
    :param top_carriers: Number of strongest subcarriers to use for HR estimation. 
                         Default is 10.
    :type top_carriers: int
    :param aggr_method: Aggregation method for the extracted frequencies.
                         Can be one of 'mean', 'median', 'trimmed'. Default is 'mean'.
    :type aggr_method: str
    :param hr_min: Minimum expected heart rate in BPM. Default is 45 BPM.
    :type hr_min: float
    :param hr_max: Maximum expected heart rate in BPM. Default is 200 BPM.
    :type hr_max: float
    :param par_bp_order: order of Butterworth bandpass filter. Default is 3.
    :type par_bp_order: int
    :param par_bp_hr_allowance:  hertz of additional allowance of Butterworth bandpass filter. Default is 0.2 Hertz.
    :type par_bp_hr_allowance: float
    :param par_sg_order: polyorder of Saviztky-Golay filter. Default is 3.
    :type par_sg_order: int
    :param par_sg_winlen: window length of Saviztky-Golay filter. Default is 15.
    :type par_sg_winlen: int
    :param par_ft_band: Minimum bandwidth to select a peak. Default is 0.1 Hz
    :type par_ft_band: float
    :param par_ft_thresh: Percentage of the peak power to be considered in the same band. Default is 0.5 (50%)
    :type par_ft_thresh: float
    :return: Estimated heart rate in Hertz, averaged over selected channels.
    :rtype: float
    """
    DEBUG_ON = True

    hr_min_Hz = hr_min / 60.0
    hr_max_Hz = hr_max / 60.0

    # Convert to numpy
    signal_matrix = np.array(signal_matrix)
    samples, channels = signal_matrix.shape

    # Store processed channels and their powers
    processed_channels = []
    variances = []
    channel_power = []

    # Process each channel
    for chan in signal_matrix.T:

        # Band-pass
        chan = butter_bandpass_filter(
            signal=chan,
            fs=fs,
            lowcut=hr_min_Hz - par_bp_hr_allowance,
            highcut=hr_max_Hz + par_bp_hr_allowance,
            order=par_bp_order
        )

        # Store filtered channel
        processed_channels.append(chan)

        # Selection
        variances.append(np.var(chan))

    # Convert back to np array
    processed_channels = np.array(processed_channels).T

    # Select top_carriers channels by power
    # top_carriers = min(top_carriers, len(channel_power))
    # top_idx = np.argsort(channel_power)[-top_carriers:]
    top_carriers = min(top_carriers, len(variances))
    top_idx = np.argsort(variances)[-top_carriers:] #selection based on dynamic energy
    strong_signal_matrix = processed_channels[:, top_idx]
        
    # 3. Spectral Averaging con Zero Padding
    n_padded = 4096 # Alta risoluzione per l'interpolazione FFT
    avg_spectrum = np.zeros(n_padded // 2 + 1)
    
    for chan in strong_signal_matrix.T:
        # Calcola FFT con zero padding
        fft_vals = np.abs(np.fft.rfft(chan, n=n_padded))
        # Somma allo spettro medio
        avg_spectrum += fft_vals

    # 4. Trova il picco sullo spettro medio
    freqs = np.fft.rfftfreq(n_padded, d=1/fs)
    
    # Maschera per il range cardiaco
    mask = (freqs >= hr_min_Hz) & (freqs <= hr_max_Hz)
    
    valid_freqs = freqs[mask]
    valid_spectrum = avg_spectrum[mask]
    
    if len(valid_spectrum) == 0:
        return 0.0

    # Picco massimo
    peak_idx = np.argmax(valid_spectrum)
    dominant_freq = valid_freqs[peak_idx]

    return dominant_freq

#endregion

#region ======= DSP process =======

def PROC_DSP_OLD(mpdeque_dsp: MPDeque, mpdeque_hr: MPDeque, log_to_file: bool = False, log_filename: str = "log.log"):

    print_log_loc("Process alive", LVL_INF)
    
    if log_to_file:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_filename, "a") as f:
            f.write(f"\n[{timestamp}] NEW SESSION\n")
            f.write(" --- Parameters:\n")
            # f.write(f"Sample window length = [{window_len}]\n")
            # f.write(f"Iterations per estimate = [{iter_per_estimate}]\n")
            f.write(f"Bandpass - Order = [{PAR__BP_ORDER}]\n")
            f.write(f"Bandpass - HR allowance = [{PAR__BP_ALLOW_HZ}] (Hertz)\n")
            # f.write(f"Savitzky-Golay - Polyomial order = [{PAR__SG_POLYORDER}]\n")
            # f.write(f"Savitzky-Golay - Window length = [{PAR__SG_WINLEN}]\n")
            f.write(f"FFT - Peak band = [{PAR__FT_BAND_WIDTH}]\n")
            f.write(f"FFT - Band threshold = [{PAR__FT_BAND_THRESH}]\n")
            f.write(f"HR Min = [{PAR__HR_MIN_BPM}]\n")
            f.write(f"HR Max = [{PAR__HR_MAX_BPM}]\n")
            f.write(f"Top carriers = [{PAR__TOP_CARRIERS}]\n")
            f.write(f"Aggregation method = [{PAR__AGGR_METHOD}]\n")
            f.write(" --- Results: (BPM / Hz)\n")

    while True:

        # Get csi data window
        csi_data_window = mpdeque_dsp.popright(block=True)

        print_log_loc("Processing csi data window", LVL_DBG)

        # Estimate HR
        hr_hz = estimate_hr_freq_PCA(
            signal_matrix=csi_data_window,
            fs=PAR_SAMP_FREQ_HZ,
            top_components=PAR__PCA_CARRIERS,
            hr_min=PAR__HR_MIN_BPM,
            hr_max=PAR__HR_MAX_BPM,
            par_bp_order=PAR__BP_ORDER,
            par_bp_hr_allowance=PAR__BP_ALLOW_HZ
        )
        """
        hr_hz = estimate_hr_freq(
            signal_matrix=csi_data_window,
            fs=PAR_SAMP_FREQ_HZ,
            top_carriers=PAR__TOP_CARRIERS,
            aggr_method=PAR__AGGR_METHOD,
            hr_min=PAR__HR_MIN_BPM,
            hr_max=PAR__HR_MAX_BPM,
            par_bp_order=PAR__BP_ORDER,
            par_bp_hr_allowance=PAR__BP_ALLOW_HZ,
            par_sg_order=PAR__SG_POLYORDER,
            par_sg_winlen=PAR__SG_WINLEN,
            par_ft_band=PAR__FT_BAND_WIDTH,
            par_ft_thresh=PAR__FT_BAND_THRESH
        )
        """

        # Enqueue
        mpdeque_hr.appendleft(hr_hz)

        print_log_loc("Estimate calculated and returned", LVL_DBG)

        if log_to_file:
            with open(log_filename, "a") as f:
                f.write(f"{Hz_to_BPM(hr_hz):.2f} BPM / {hr_hz:.3f} Hz\n")


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
            if len(history_buffer) >= 3:
                stable_output = np.median(history_buffer)
            else:
                stable_output = new_hr_hz
                
            mpdeque_hr.appendleft(stable_output)
            
            print(f"Est: {new_hr_hz*60:.1f} BPM (Stable: {stable_output*60:.1f})")
            
            if log_to_file:
                with open(log_filename, "a") as f:
                    f.write(f"{Hz_to_BPM(stable_output):.2f} BPM / {stable_output:.3f} Hz\n")
#endregion