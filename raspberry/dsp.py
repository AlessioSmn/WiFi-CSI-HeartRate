#region ======= Imports =======

import os
import time
import numpy as np
from mpque import MPDeque
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from log import print_log, LVL_DBG, LVL_INF, LVL_ERR, DebugLevel
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks, welch

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
PAR__BP_ALLOW_HZ = 0.01

# Savitzky-Golay polynomial order
PAR__SG_POLYORDER = 4

# Savitzky-Golay window length
PAR__SG_WINLEN = 15

# Number of carriers to use (by power in band)
PAR__TOP_CARRIERS = 15

# Minimum HR
PAR__HR_MIN_BPM = 45

# Maximum HR
PAR__HR_MAX_BPM = 200

# Minimum width for a peak to be selected (after FFT)
PAR__FT_BAND_WIDTH = 0.1 # Hz

# Percentage of peak power
PAR__FT_BAND_THRESH = 0.5

#endregion

#region ======= CSI Data functions =======

def resample_window(csi_data_window_counters):
    """
    TODO
    """

    counters = np.array([x[0] for x in csi_data_window_counters])
    signals  = np.array([x[1] for x in csi_data_window_counters])

    full_counters = np.arange(
        counters[0],
        counters[-1] + 1
    )

    n_samples = len(full_counters)
    n_subcarr = signals.shape[1]

    rebuilt = np.zeros((n_samples, n_subcarr))

    for ch in range(n_subcarr):
        rebuilt[:, ch] = np.interp(
            full_counters,
            counters,
            signals[:, ch]
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

def savitzky_golay_smooth(
        signal: np.ndarray,
        window_length: int = 15,
        polyorder: int = 3
    ) -> np.ndarray:
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

def dominant_frequency(
        signal: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: float) -> float:
    
    """
    Compute the dominant frequency of a real-valued signal within a specified frequency band.

    :param signal: Input time-domain signal (1D array of samples).
    :type signal: np.ndarray
    :param lowcut: Lower bound of the frequency band (Hz). Frequencies below this are ignored.
    :type lowcut: float
    :param highcut: Upper bound of the frequency band (Hz). Frequencies above this are ignored.
    :type highcut: float
    :param fs: Sampling frequency of the signal (Hz).
    :type fs: float
    :return: Dominant frequency (Hz) within the specified band.
    :rtype: float

    :note: The function uses the real FFT (rfft) and returns the frequency corresponding 
           to the maximum FFT amplitude within the [lowcut, highcut] range.
    """
    
    # === Step 0: Calculate number of samples

    # Number of samples for the signal
    samples = len(signal)
    
    # === Step 1: Calculate frequencies intervals

    # Calculate step frequencies recognized
    freqs = rfftfreq(samples, d=1/fs)

    # Mask out unwanted frequencies (array of True/False)
    mask = (freqs >= lowcut) & (freqs <= highcut)
    
    # Filter frequencies based on mask
    freq_band = freqs[mask]

    # === Step 2: Calculate dominant frequency

    # Real FFT of signal
    fft_vals = np.abs(rfft(signal))

    # Filter only in the [lowcut; highcut] band
    fft_band = fft_vals[mask]

    # Return dominant frequency
    return freq_band[np.argmax(fft_band)]

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
        thresh = 0.5 * peak_val   # -6 dB

        # Left bound
        l = i
        while l > 0 and fft_band[l] > thresh:
            l -= 1

        # Right bound
        r = i
        while r < len(fft_band) - 1 and fft_band[r] > thresh:
            r += 1

        bandwidth = (r - l) * df
        if bandwidth >= min_bw_hz:
            return freq_band[i]

    # Fallback: highest peak
    return freq_band[np.argmax(fft_band)]

#endregion

#region ======= Main function =======

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
                         Can be one of 'mean', 'median'. Default is 'mean'.
    :type aggr_method: str
    :param hr_min: Minimum expected heart rate in BPM. Default is 45 BPM.
    :type hr_min: float
    :param hr_max: Maximum expected heart rate in BPM. Default is 200 BPM.
    :type hr_max: float
    :param par_bp_order: order of Butterworth bandpass filter. Default is 3.
    :type par_bp_order: int
    :param par_bp_hr_allowance:  hertz of additional allowance of Butterworth bandpass filter. Default is 0.2 Hertz.
    :type par_bp_hr_allowance: float
    :param par_sg_order: polyorder of Saviztky-Golay filter. Default is 15.
    :type par_sg_order: int
    :param par_sg_winlen: window length of Saviztky-Golay filter. Default is 3.
    :type par_sg_winlen: int
    :param par_ft_band: Minimum bandwidth to select a peak. Default is 0.1 Hz
    :type par_ft_band: float
    :param par_ft_thresh: Percentage of the peak power to be considered in the same band. Default is 0.5 (50%)
    :type par_ft_thresh: float
    :return: Estimated heart rate in Hertz, averaged over selected channels.
    :rtype: float
    """
    DEBUG_ON = False

    hr_min_Hz = hr_min / 60.0
    hr_max_Hz = hr_max / 60.0

    # Convert to numpy
    signal_matrix = np.array(signal_matrix)
    samples, channels = signal_matrix.shape

    # Store processed channels and their powers
    processed_channels = []
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

        # Savitzky-Golay smoothing
        chan = savitzky_golay_smooth(
            signal=chan,
            window_length=par_sg_winlen,
            polyorder=par_sg_order
        )

        # Store filtered channel
        processed_channels.append(chan)

        # Power for selection
        channel_power.append(np.mean(chan**2))

    # Convert back to np array
    processed_channels = np.array(processed_channels).T

    # Select top_carriers channels by power
    top_carriers = min(top_carriers, len(channel_power))
    top_idx = np.argsort(channel_power)[-top_carriers:]
    strong_signal_matrix = processed_channels[:, top_idx]
        
    # DEBUG
    if DEBUG_ON:
        debug_spectrums(
            original_matrix=signal_matrix,
            processed_matrix=processed_channels,
            fs=fs,
            top_idx=top_idx
        )
    
    hr_hz_estimates = []

    # Compute HR per selected channel
    for chan in strong_signal_matrix.T:

        # Extract dominant frequency via FFT
        dominant_freq = dominant_wide_frequency(
            signal=chan,
            lowcut=hr_min_Hz,
            highcut=hr_max_Hz,
            fs=fs,
            min_bw_hz=par_ft_band,
            threshold=par_ft_thresh
        )

        hr_hz_estimates.append(dominant_freq)

    if aggr_method not in ['mean', 'median']:
        aggr_method = 'mean'

    res = -1

    # Return the aggregated heart rate
    if aggr_method == 'mean':
        res = np.mean(hr_hz_estimates)
    elif aggr_method == 'median':
        res = np.median(hr_hz_estimates)
        
    return res

#endregion

#region ======= DSP process =======

def PROC_DSP(mpdeque_dsp: MPDeque, mpdeque_hr: MPDeque, log_to_file: bool = False, log_filename: str = "log.log"):

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
            f.write(f"Savitzky-Golay - Polyomial order = [{PAR__SG_POLYORDER}]\n")
            f.write(f"Savitzky-Golay - Window length = [{PAR__SG_WINLEN}]\n")
            f.write(f"FFT - Peak band = [{PAR__FT_BAND_WIDTH}]\n")
            f.write(f"FFT - Band threshold = [{PAR__FT_BAND_WIDTH}]\n")
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

        # Enqueue
        mpdeque_hr.appendleft(hr_hz)

        print_log_loc("Estimate calculated and returned", LVL_DBG)

        if log_to_file:
            with open(log_filename, "a") as f:
                f.write(f"{Hz_to_BPM(hr_hz):.2f} BPM / {hr_hz:.3f} Hz\n")

#endregion