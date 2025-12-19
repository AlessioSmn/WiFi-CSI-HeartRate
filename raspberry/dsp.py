import ast
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from scipy.fft import rfft, rfftfreq

def parse_csi_amplitudes(csi_str):
    """
    Given an array of (re, im) values, ordered in a single string array, 
    returns an array with the relative amplitudes
    
    :param csi_str: Array of (re, im)

    :return: Array of amplitudes
    :rtype: Literal[-1] | None
    """
    # Convert the string into a Python list of ints
    values = ast.literal_eval(csi_str)

    # Reshape into (imag, real) pairs
    complex_pairs = np.array(values).reshape(-1, 2)

    # Convert to complex numbers: real + j*imag
    csi_complex = complex_pairs[:,1] + 1j * complex_pairs[:,0]

    # Compute amplitudes
    amplitudes = np.abs(csi_complex)
    return amplitudes

def butter_bandpass_filter(
        signal: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: float,
        order: int = 3) -> np.ndarray:
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

def estimate_hr_freq(
        signal_matrix: list[list[float]],
        fs: float,
        top_carriers: int = 10,
        aggr_method: str = 'mean',
        hr_min: float = 45,
        hr_max: float = 200,
        par_bp_order: int = 2,
        par_bp_hr_allowance: float = 0.5,
        par_sg_order: int = 2,
        par_sg_winlen: int = 11) -> float:
    
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
    :param par_bp_order: order of Butterworth bandpass filter. Default is 2.
    :type par_bp_order: int
    :param par_bp_hr_allowance:  hertz of additional allowance of Butterworth bandpass filter. Default is 0.5 Hertz.
    :type par_bp_hr_allowance: float
    :param par_sg_order: polyorder of Saviztky-Golay filter. Default is 11.
    :type par_sg_order: int
    :param par_sg_winlen: window length of Saviztky-Golay filter. Default is 2.
    :type par_sg_winlen: int
    :return: Estimated heart rate in Hertz, averaged over selected channels.
    :rtype: float
    """

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
    top_idx = np.argsort(channel_power)[-top_carriers:]
    strong_signal_matrix = processed_channels[:, top_idx]
    
    hr_hz_estimates = []

    # Compute HR per selected channel
    for chan in strong_signal_matrix.T:

        # Extract dominant frequency via FFT
        dominant_freq = dominant_frequency(
            signal=chan,
            lowcut=hr_min_Hz,
            highcut=hr_max_Hz,
            fs=fs
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


def estimate_hr(
        signal_matrix: list[list[float]],
        fs: float,
        top_carriers: int = 10,
        hr_min: float = 45,
        hr_max: float = 200,
        par_bp_order: int = 2,
        par_bp_hr_allowance: float = 0.5,
        par_sg_order: int = 2,
        par_sg_winlen: int = 11) -> float:
    
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
    :param hr_min: Minimum expected heart rate in BPM. Default is 45 BPM.
    :type hr_min: float
    :param hr_max: Maximum expected heart rate in BPM. Default is 200 BPM.
    :type hr_max: float
    :param par_bp_order: order of Butterworth bandpass filter. Default is 2.
    :type par_bp_order: int
    :param par_bp_hr_allowance:  hertz of additional allowance of Butterworth bandpass filter. Default is 0.5 Hertz.
    :type par_bp_hr_allowance: float
    :param par_sg_order: polyorder of Saviztky-Golay filter. Default is 11.
    :type par_sg_order: int
    :param par_sg_winlen: window length of Saviztky-Golay filter. Default is 2.
    :type par_sg_winlen: int
    :return: Estimated heart rate in beats per minute (BPM), averaged over selected channels.
    :rtype: float
    """

    hr_min_Hz = hr_min / 60.0
    hr_max_Hz = hr_max / 60.0
    hr_allowance = 0.5 # Hertz

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
    top_idx = np.argsort(channel_power)[-top_carriers:]
    strong_signal_matrix = processed_channels[:, top_idx]
    
    bpm_estimate = []

    # Compute HR per selected channel
    for chan in strong_signal_matrix.T:

        # Extract dominant frequency via FFT
        dominant_freq = dominant_frequency(
            signal=chan,
            lowcut=hr_min_Hz,
            highcut=hr_max_Hz,
            fs=fs
        )
        hr_bpm = dominant_freq * 60
        bpm_estimate.append(hr_bpm)

    # Return the mean HR
    mn = np.mean(bpm_estimate)
    md = np.median(bpm_estimate)
    return mn, md

