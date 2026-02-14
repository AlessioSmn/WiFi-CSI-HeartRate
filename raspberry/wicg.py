#region ======= Imports =======

import os
import time
import numpy as np
from mpque import MPDeque
import matplotlib.pyplot as plt
from scipy.stats import trim_mean
from scipy.fft import rfft, rfftfreq
from scipy.linalg import hankel, svd
from scipy.sparse.linalg import svds
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
PAR__SAMP_FREQ_HZ = 40

# Bandpass filter order
PAR__BP_ORDER = 3

# Bandpass filter extra margin
PAR__BP_ALLOW_HZ = 0

PAR__PCA_CARRIERS = 3

# Minimum HR
PAR__HR_MIN_BPM = 48
PAR__HR_MIN_HZ = PAR__HR_MIN_BPM / 60.0
# Maximum HR
PAR__HR_MAX_BPM = 160
PAR__HR_MAX_HZ = PAR__HR_MAX_BPM / 60.0

#endregion

#region ======= CSI Data functions =======

def resample_window_complex(csi_data_window_counters):
    """
    Resample a CSI window (COMPLEX data) aligned on a continuous counter axis.
    Handles Real and Imaginary parts separately for interpolation.
    """
    # Extract counters and CSI signals (Complex)
    counters = np.array([x[0] for x in csi_data_window_counters])
    signals = np.array([x[1] for x in csi_data_window_counters]) # This is now complex

    # Build the expected continuous counter sequence
    full_counters = np.arange(counters[0], counters[-1] + 1)

    # Detect missing counters (logging opzionale)
    num_missing_counters = len(full_counters) - len(counters)
    if num_missing_counters > 0:
        print_log_loc(f"Resampling: {num_missing_counters} missing counters", LVL_DBG)

    n_samples = len(full_counters)
    n_subcarr = signals.shape[1]

    # Allocate rebuilt CSI matrix as COMPLEX
    rebuilt = np.zeros((n_samples, n_subcarr), dtype=np.complex128)

    # Interpolate each subcarrier
    for ch in range(n_subcarr):
        # Estrai la serie temporale per la sottoportante corrente
        y_vals = signals[:, ch]
        
        # Interpolazione separata Reale e Immaginaria
        y_real = np.interp(full_counters, counters, np.real(y_vals))
        y_imag = np.interp(full_counters, counters, np.imag(y_vals))
        
        # Ricombina
        rebuilt[:, ch] = y_real + 1j * y_imag

    return rebuilt

def parse_csi_complex(values):
    """
    Given a flat list of ints [re0, im0, re1, im1, ...],
    returns a numpy array of COMPLEX numbers.

    :param values: list of ints
    :return: numpy array of complex128 (Re + j*Im)
    """
    if len(values) % 2 != 0:
        raise ValueError("CSI values must have even length (pairs of re/im)")

    # Reshape into (N, 2) where col 0 is Real, col 1 is Imag
    complex_pairs = np.array(values).reshape(-1, 2)

    # Convert to complex numbers: real + j*imag
    # NOTA: Ho corretto l'ordine rispetto al tuo codice originale per rispettare lo standard [Re, Im]
    csi_complex = complex_pairs[:, 0] + 1j * complex_pairs[:, 1]
    
    return csi_complex

def get_microseconds_epoch():
    return time.time_ns() // 1_000

#endregion

#region ======= DSP functions (single step) =======

def butter_bandpass_filter(
        signal: np.ndarray,
        lowcut: float = PAR__HR_MIN_HZ,
        highcut: float = PAR__HR_MAX_HZ,
        fs: float = PAR__SAMP_FREQ_HZ
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
    b, a = butter(PAR__BP_ORDER, [low, high], btype="band")
    # filtfilt for zero-phase
    return filtfilt(b, a, x)

def wicg_pca_denoise(complex_matrix: np.ndarray) -> np.ndarray:
    """
    Implements the Ambient Noise Removal from WiCG (Section 4.1).
    Performs PCA on the subcarrier dimension separately for I and Q components.
    
    :param complex_matrix: (Samples x Subcarriers) complex CSI data
    :return: Reconstructed complex CSI data
    """
    # complex_matrix shape: [Time, Subcarriers]
    # WiCG lavora sulla dimensione delle sottoportanti (CFR)[cite: 1759].
    
    # 1. Separazione I (Real) e Q (Imag)
    I_comp = np.real(complex_matrix)
    Q_comp = np.imag(complex_matrix)
    
    def reconstruct_component(X):
        # X shape: [Time, Subcarriers]
        # WiCG Equation (10): Covariance along subcarrier dimension
        # Transpose to [Subcarriers, Time] per il calcolo standard della covarianza spaziale
        X_T = X.T 
        
        # Center the data (mean of subcarriers at different time points) [cite: 1816]
        mean_vec = np.mean(X_T, axis=1, keepdims=True)
        X_centered = X_T - mean_vec
        
        # Covariance Matrix (K x K where K is subcarriers)
        # C = (1 / (K-1)) * X_centered @ X_centered.T
        
        # SVD (PCA)
        # U contiene gli autovettori.
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Select Top 1 Principal Component [cite: 1901]
        # "Select the top 1 eigenvector to form the transformation matrix V1"
        # Ricostruzione: Project e Reconstruct
        # Nota: In SVD (X = U S Vt), le componenti principali sono in U.
        # Ricostruiamo usando solo la prima componente.
        
        S_new = np.zeros_like(S)
        S_new[0] = S[0] # Keep only first eigenvalue
        
        # Reconstruct: X_rec = U * S_new * Vt + Mean
        X_rec_T = U @ np.diag(S_new) @ Vt + mean_vec
        
        return X_rec_T.T # Return to [Time, Subcarriers]

    I_rec = reconstruct_component(I_comp)
    Q_rec = reconstruct_component(Q_comp)
    
    # Ricombina in complesso [cite: 1794]
    return I_rec + 1j * Q_rec

def wicg_calculate_her(
        signal: np.ndarray,
        hr_min_hz: float = PAR__HR_MIN_HZ,
        hr_max_hz: float = PAR__HR_MAX_HZ,
        fs: float = PAR__SAMP_FREQ_HZ
    ) -> float:
    """
    Calculates Heartbeat Energy Ratio (HER) for candidate selection (Section 5.1).
    [cite: 1982-1984]
    """
    # 1. FFT
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(n, d=1/fs)
    
    # 2. Define masks
    # "Get rid of low frequency components below 1 Hz"
    mask_heart = (freqs >= hr_min_hz) & (freqs <= hr_max_hz) # Cardiac range
    mask_total = (freqs >= hr_min_hz) # Tutto sopra 1Hz
    
    energy_heart = np.sum(fft_vals[mask_heart] ** 2)
    energy_total = np.sum(fft_vals[mask_total] ** 2)
    
    if energy_total == 0:
        return 0.0
        
    return energy_heart / energy_total

def wicg_ssa_enhance_slow(signal: np.ndarray, fs: float, window_sec: float = 3.0) -> np.ndarray:
    """
    Implements Singular Spectrum Analysis (SSA) Enhancement (Section 5.2).
    Used to recover incomplete heartbeat patterns.
    """
    N = len(signal)
    L = int(fs * window_sec) # Lag window size [cite: 2154]
    
    # Safety check for window size
    if L >= N // 2:
        L = N // 3
    if L < 2:
        return signal

    K = N - L + 1
    
    # Step 1: Construct Trajectory Matrix (Hankel) [cite: 2019]
    # La matrice di traiettoria ha L righe e K colonne
    X = hankel(signal[:L], signal[L-1:])
    
    # Step 2: SVD [cite: 2038]
    U, S, Vt = svd(X, full_matrices=False)
    
    # Step 3: Component Reconstruction [cite: 2040]
    # "Select the singular vector corresponding to the biggest singular value"
    # Poiché abbiamo già filtrato passa-alto il segnale PRIMA di chiamare questa funzione,
    # la componente più forte è il battito cardiaco (la respirazione è rimossa).
    
    rank = 1 # Usiamo solo la componente principale come da paper [cite: 2071]
    X_rec = np.zeros_like(X)
    
    for i in range(rank):
        # Outer product of U_i and V_i scaled by S_i
        elem = S[i] * np.outer(U[:, i], Vt[i, :])
        X_rec += elem
        
    # Step 4: Anti-diagonal Averaging (Diagonal Averaging) [cite: 2072]
    # Ricostruzione della serie temporale dalla matrice Hankel ricostruita
    # Questa è una procedura standard SSA
    
    # Implementazione veloce dell'averaging diagonale
    # Per una matrice Hankel ricostruita da SVD, la media sulle anti-diagonali restituisce la serie.
    new_signal = np.zeros(N)
    count = np.zeros(N)
    
    # Iterazione semplice (può essere ottimizzata, ma per L~100 è ok)
    for r in range(X_rec.shape[0]):
        for c in range(X_rec.shape[1]):
            new_signal[r+c] += X_rec[r, c]
            count[r+c] += 1
            
    return new_signal / count
def wicg_ssa_enhance(signal: np.ndarray, fs: float, window_sec: float = 3.0) -> np.ndarray:
    """
    Implements Fast SSA using Partial SVD (k=1).
    Drastically faster than full SVD for single component reconstruction.
    """
    N = len(signal)
    L = int(fs * window_sec) 
    
    # Safety check
    if L >= N // 2: L = N // 3
    if L < 2: return signal

    K = N - L + 1
    
    # 1. Trajectory Matrix (Hankel)
    X = hankel(signal[:L], signal[L-1:])
    
    # 2. Fast Partial SVD (Compute only top 1 component)
    # k=1 restituisce la componente più forte (il battito cardiaco, dopo il filtro)
    try:
        # svds restituisce u, s, vt. 
        # Nota: per k piccolo, svds è molto più veloce di svd standard
        U, S, Vt = svds(X, k=1)
    except Exception as e:
        # Fallback in caso di convergenza fallita su segnali strani
        print_log_loc(f"SSA SVD Warning: {e}", LVL_DBG)
        return signal

    # 3. Component Reconstruction (Rank-1)
    # svds restituisce le componenti ordinate dalla più piccola, 
    # ma con k=1 ne abbiamo solo una.
    X_rec = S[0] * np.outer(U[:, 0], Vt[0, :])
    
    # 4. Diagonal Averaging (Vectorized approach roughly)
    # Ricostruzione rapida della serie temporale
    new_signal = np.zeros(N)
    count = np.zeros(N)
    
    # Ottimizzazione del loop diagonale
    # Possiamo sfruttare la struttura per sommare più velocemente, 
    # ma per finestre di queste dimensioni un loop compilato o semplice va bene.
    # Manteniamo il loop esplicito per chiarezza, è O(N*L) ma operazioni semplici.
    rows, cols = X_rec.shape
    for r in range(rows):
        for c in range(cols):
            new_signal[r+c] += X_rec[r, c]
            count[r+c] += 1
            
    return new_signal / count
#endregion

#region ======= Main Function =======

def estimate_hr_freq_WiCG(
        complex_signal_matrix: list[list[complex]], # N.B. Deve essere COMPLESSO
        fs: float = PAR__SAMP_FREQ_HZ
    ) -> float:
    
    """
    Implementation of the WiCG pipeline.
    
    Steps:
    1. PCA Denoising on Subcarrier Dimension (I/Q separate).
    2. Bandpass Filtering (High-pass > 0.8Hz).
    3. Candidate Selection via HER (Heartbeat Energy Ratio).
    4. SSA Enhancement on selected candidates.
    5. Spectral Aggregation and Peak Finding.
    """
    DEBUG_ON = True
    timestamp = int(time.time())

    DBG_TIME__START = get_microseconds_epoch()
    
    # Convert to numpy complex
    raw_matrix = np.array(complex_signal_matrix)
    if np.iscomplexobj(raw_matrix) == False:
        # Fallback se l'utente passa ampiezze per sbaglio, ma riduce l'efficacia
        print_log_loc("WARNING: WiCG requires Complex CSI. Using dummy phase.", LVL_ERR)
        raw_matrix = raw_matrix + 1j * 0 

    # --- 1. Ambient Noise Removal (PCA on CFR) ---
    # Section 4.1: "reconstructing the I and Q components... using PCA" 
    denoised_matrix = wicg_pca_denoise(raw_matrix)
    
    # Calcolo magnitudo per le fasi successive
    # "For CSI amplitude, its square is computed..." [cite: 1718]
    # Useremo l'ampiezza del segnale ricostruito
    mag_matrix = np.abs(denoised_matrix)
    
    DBG_TIME__PCA = get_microseconds_epoch()
    DBG_TIME__PCA_step = DBG_TIME__PCA - DBG_TIME__START

    # --- 2. High-Pass / Band-Pass Filtering ---
    # Section 5.2.3: "Before applying SSA, a high-pass filter is applied... above 0.8 Hz" [cite: 2016]

    filtered_channels = []
    
    for chan in mag_matrix.T:
        # Detrend base
        chan = detrend(chan)
        # Bandpass (Butterworth)
        # Using default parameters
        chan_bp = butter_bandpass_filter(chan)
        filtered_channels.append(chan_bp)
        
    filtered_channels = np.array(filtered_channels).T
    
    DBG_TIME__BANDPASS = get_microseconds_epoch()
    DBG_TIME__BANDPASS_step = DBG_TIME__BANDPASS - DBG_TIME__PCA
    
    # --- 3. Candidate Subcarrier Selection (HER) ---
    # Section 5.1 
    her_scores = []
    for i in range(filtered_channels.shape[1]):
        score = wicg_calculate_her(filtered_channels[:, i])
        her_scores.append(score)
    her_scores = np.array(her_scores)
    
    # Strategia ibrida: Soglia + Top-K Limit
    MAX_CANDIDATES = 15  # <-- LIMITATORE CRUCIALE PER LE PRESTAZIONI
    THRESH_PERC = 1.01
    
    if len(her_scores) > 0:
        # Ordina gli indici in base allo score decrescente
        sorted_indices = np.argsort(her_scores)[::-1]
        
        thresh = THRESH_PERC * np.max(her_scores)
        above = np.where(her_scores >= thresh)[0]

        candidate_indices = (
            above if len(above) >= MAX_CANDIDATES
            else sorted_indices[:MAX_CANDIDATES]
        )
    else:
        candidate_indices = []

    # Seleziona solo i segnali dei candidati scelti
    selected_signals = filtered_channels[:, candidate_indices]
    
    DBG_TIME__HER = get_microseconds_epoch()
    DBG_TIME__HER_step = DBG_TIME__HER - DBG_TIME__BANDPASS

    # --- 4. Heartbeat Components Enhancement (SSA) ---
    # Section 5.2 [cite: 1989]
    enhanced_signals = []
    
    for i in range(selected_signals.shape[1]):
        sig = selected_signals[:, i]
        # Apply SSA
        # "window size... L = Sampling Rate * 3" [cite: 2154]
        # Noi usiamo un parametro sicuro, es. 2-3 secondi
        sig_ssa = wicg_ssa_enhance(sig, fs, window_sec=3.0)
        enhanced_signals.append(sig_ssa)
        
    if not enhanced_signals:
        return 0.0
        
    enhanced_matrix = np.array(enhanced_signals).T
    
    DBG_TIME__SSA = get_microseconds_epoch()
    DBG_TIME__SSA_step = DBG_TIME__SSA - DBG_TIME__HER

    # --- 5. Heartbeat Rate Estimation (Spectral Aggregation) ---
    # Section 5.3: "aggregate the frequency spectra... Sum(|FFT(x)|)" [cite: 2164]
    
    n_padded = 4096
    avg_spectrum = np.zeros(n_padded // 2 + 1)
    
    for i in range(enhanced_matrix.shape[1]):
        fft_vals = np.abs(np.fft.rfft(enhanced_matrix[:, i], n=n_padded))
        # Normalizzazione opzionale per evitare che un canale domini
        if np.max(fft_vals) > 0:
            fft_vals /= np.max(fft_vals)
        avg_spectrum += fft_vals
        
    freqs = np.fft.rfftfreq(n_padded, d=1/fs)
    
    # Mask per range 1-2 Hz (60-120 BPM) come suggerito dal paper, o range utente
    # Il paper suggerisce 1-2Hz[cite: 2191], ma usiamo i parametri utente per flessibilità
    mask = (freqs >= PAR__HR_MIN_HZ) & (freqs <= PAR__HR_MAX_HZ)
    
    valid_freqs = freqs[mask]
    valid_spectrum = avg_spectrum[mask]
    
    if len(valid_spectrum) == 0:
        return 0.0
        
    # Peak finding
    best_idx = np.argmax(valid_spectrum)
    best_freq = valid_freqs[best_idx]

    DBG_TIME__EST = get_microseconds_epoch()
    DBG_TIME__EST_step = DBG_TIME__EST - DBG_TIME__SSA

    if DEBUG_ON:
        os.makedirs("./img/wicg_debug", exist_ok=True)
        
        # Creiamo una dashboard 2x2
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"WiCG Pipeline Debug - {timestamp}", fontsize=16)

        # PLOT 1: Denoising (Paper Section 4.1)
        # Confrontiamo il modulo del segnale grezzo vs PCA Denoised (primo subcarrier)
        raw_mag = np.abs(raw_matrix[:, 0])
        denoised_mag = np.abs(denoised_matrix[:, 0])
        # Normalizziamo per confronto visivo
        # axs[0, 0].plot(detrend(raw_mag)/np.std(raw_mag), label="Raw Amp (Detrended)", alpha=0.5, color='gray')
        axs[0, 0].plot(detrend(denoised_mag)/np.std(denoised_mag), label="WiCG PCA Denoised", color='blue')
        axs[0, 0].set_title("1. PCA Denoising (Comparison)")
        axs[0, 0].legend(loc='upper right', fontsize='small')
        axs[0, 0].grid(True, alpha=0.3)

        # PLOT 2: Candidate Selection HER (Paper Section 5.1)
        # Bar plot degli score HER
        axs[0, 1].bar(range(len(her_scores)), her_scores, color='skyblue', label='HER Score')
        #axs[0, 1].axhline(y=thresh, color='r', linestyle='--', label='Threshold (70%)')
        # Evidenziamo i selezionati
        axs[0, 1].bar(candidate_indices, her_scores[candidate_indices], color='orange', label='Selected')
        axs[0, 1].set_title(f"2. Subcarrier Selection (Candidates: {len(candidate_indices)})")
        axs[0, 1].legend(loc='upper right', fontsize='small')
        axs[0, 1].grid(True, axis='y', alpha=0.3)

        # PLOT 3: SSA Enhancement (Paper Section 5.2)
        # Mostriamo il primo candidato selezionato PRIMA e DOPO SSA
        if enhanced_matrix.shape[1] > 0:
            sig_before = selected_signals[:, 0]
            sig_after = enhanced_matrix[:, 0]
            axs[1, 0].plot(sig_before, label="Before SSA (Filtered)", alpha=0.6, color='orange')
            axs[1, 0].plot(sig_after, label="After SSA (Reconstructed)", color='green', linewidth=2)
            axs[1, 0].set_title("3. SSA Enhancement (Waveform Recovery)")
            axs[1, 0].legend(loc='upper right', fontsize='small')
            axs[1, 0].grid(True, alpha=0.3)

        # PLOT 4: Final Decision Spectrum (Paper Section 5.3)
        # Spettro finale aggregato
        norm_spectrum = valid_spectrum / np.max(valid_spectrum) if np.max(valid_spectrum) > 0 else valid_spectrum
        axs[1, 1].plot(valid_freqs * 60, norm_spectrum, label="Aggregated Spectrum", color='purple')
        axs[1, 1].axvline(best_freq * 60, color='red', linewidth=2, label=f"Est: {best_freq*60:.1f} BPM")
        
        axs[1, 1].set_title("4. Final HR Estimation")
        axs[1, 1].set_xlabel("BPM")
        axs[1, 1].legend(loc='upper right', fontsize='small')
        axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
        plt.savefig(f"./img/wicg_debug/{timestamp}_wicg.jpg")
        plt.close()
        
    DBG_TIME__PLOT = get_microseconds_epoch()
    DBG_TIME__PLOT_step = DBG_TIME__PLOT - DBG_TIME__EST

    print(f"STEP1: {DBG_TIME__PCA_step//1000:.3f} ms (PCA)")
    print(f"STEP2: {DBG_TIME__BANDPASS_step//1000:.3f} ms (BANDPASS)")
    print(f"STEP3: {DBG_TIME__HER_step//1000:.3f} ms (HER)")
    print(f"STEP4: {DBG_TIME__SSA_step//1000:.3f} ms (SSA)")
    print(f"STEP5: {DBG_TIME__EST_step//1000:.3f} ms (EST)")
    print(f"STEP6: {DBG_TIME__PLOT_step//1000:.3f} ms (PLOT)")
    print(f"TOTAL: {(DBG_TIME__PLOT-DBG_TIME__START)//1000:.3f} ms")

    return best_freq

#endregion

#region ======= DSP process =======

def PROC_DSP(mpdeque_dsp: MPDeque, mpdeque_hr: MPDeque, log_to_file: bool = False, log_filename: str = "log.log"):

    print_log_loc("Process alive", LVL_INF)

    while True:
        csi_data_window = mpdeque_dsp.popright(block=True)
        print_log_loc("Processing csi data window", LVL_DBG)

        new_hr_hz = estimate_hr_freq_WiCG(
            complex_signal_matrix=csi_data_window
        )
            
        # print(f"Est: {new_hr_hz*60:.1f} BPM \t/ Mean: {mean_hr*60:.1f} \t/Median: {mediam_hr*60:.1f}")
        print(f"Est: {new_hr_hz*60:.1f} BPM")
        
        mpdeque_hr.appendleft(new_hr_hz)

#endregion