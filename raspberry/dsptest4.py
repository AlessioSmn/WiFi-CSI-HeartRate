import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.decomposition import PCA
from mpque import MPDeque

# --- TUA FUNZIONE DI INPUT (INVARIATA) ---
def resample_window_complex(csi_data_window_counters):
    """
    Resample a CSI window (COMPLEX data) aligned on a continuous counter axis.
    Handles Real and Imaginary parts separately for interpolation.
    """
    if not csi_data_window_counters:
        return np.array([])
        
    counters = np.array([x[0] for x in csi_data_window_counters])
    signals = np.array([x[1] for x in csi_data_window_counters]) 

    full_counters = np.arange(counters[0], counters[-1] + 1)

    # Detect missing counters (logging rimosso per brevità)
    n_samples = len(full_counters)
    n_subcarr = signals.shape[1]

    rebuilt = np.zeros((n_samples, n_subcarr), dtype=np.complex128)

    for ch in range(n_subcarr):
        y_vals = signals[:, ch]
        y_real = np.interp(full_counters, counters, np.real(y_vals))
        y_imag = np.interp(full_counters, counters, np.imag(y_vals))
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

def find_nearest_idx(array, value):
    """Trova l'indice del valore più vicino nell'array."""
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

# --- FUNZIONI CORE DI ELABORAZIONE ---

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Applica un filtro passa-banda Butterworth.
    Usa filtfilt per evitare sfasamenti temporali (zero-phase filtering).
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def sanitize_phase(csi_matrix):
    """
    Rimuove l'errore lineare di fase (CFO/SFO) da una matrice CSI Complessa.
    Input: (Time, Subcarriers) Complex
    Output: (Time, Subcarriers) Phase (Sanitized)
    """
    n_time, n_sub = csi_matrix.shape
    clean_phase = np.zeros((n_time, n_sub))
    
    # Indici delle sottoportanti (assumiamo indici lineari 0..N)
    # Se conosci gli indici reali IEEE (es. -28, -26...) usali qui per x_axis
    x_axis = np.arange(n_sub) 
    
    for t in range(n_time):
        # 1. Estrai fase e fai unwrapping (gestione salti 2pi)
        raw_phase = np.angle(csi_matrix[t, :])
        unwrapped = np.unwrap(raw_phase)
        
        # 2. Fit lineare (pendenza e intercetta)
        # Polyfit di grado 1 restituisce [slope, intercept]
        p = np.polyfit(x_axis, unwrapped, 1)
        slope, intercept = p[0], p[1]
        
        # 3. Sottrai la retta di errore
        est_error = slope * x_axis + intercept
        clean_phase[t, :] = unwrapped - est_error
        
    return clean_phase

def select_best_subcarriers(matrix, keep_n=10):
    """
    Seleziona le N sottoportanti con la varianza più alta.
    Input: Matrice (Time, Subcarriers) di Ampiezza o Fase
    Output: Matrice ridotta (Time, keep_n)
    """
    # Calcola varianza lungo l'asse temporale
    variances = np.var(matrix, axis=0)
    
    # Ottieni gli indici delle varianze maggiori
    # argsort ordina crescente, prendiamo gli ultimi keep_n
    best_indices = np.argsort(variances)[-keep_n:]
    
    return matrix[:, best_indices]

def check_harmonics(detected_bpm, spectrum, freqs):
    # Se il valore è basso, non ci sono armoniche inferiori fisiologiche da controllare
    if detected_bpm < 95:
        return detected_bpm
        
    peak_energy = np.max(spectrum) # Altezza del picco trovato (es. a 130)
    
    # 1. CONTROLLO MEZZA FREQUENZA (1/2)
    bpm_half = detected_bpm / 2
    
    # Cerchiamo energia attorno alla metà esatta (+/- 10 BPM di tolleranza)
    idx_start = find_nearest_idx(freqs, (bpm_half - 10)/60)
    idx_end = find_nearest_idx(freqs, (bpm_half + 10)/60)
    
    # Protezione indici array
    if idx_start >= idx_end: 
        idx_end = idx_start + 1
        
    # Calcola il massimo locale in quella zona
    energy_at_half = np.max(spectrum[idx_start:idx_end])
    
    # SOGLIA DI SICUREZZA: 0.5 (50%)
    # Se a 65 BPM c'è solo rumore (es. 10% del picco), allora 130 è REALE -> Ritorna 130.
    # Se a 65 BPM c'è un picco vero (es. 60% del picco a 130), allora 130 è FALSO -> Ritorna 65.
    if energy_at_half > 0.5 * peak_energy:
        return bpm_half

    # 2. CONTROLLO UN TERZO DI FREQUENZA (1/3) - Raro ma possibile (es. rilevi 180 invece di 60)
    # Lo facciamo solo se il BPM è molto alto (>130)
    if detected_bpm > 130:
        bpm_third = detected_bpm / 3
        idx_start_3 = find_nearest_idx(freqs, (bpm_third - 10)/60)
        idx_end_3 = find_nearest_idx(freqs, (bpm_third + 10)/60)
        
        if idx_start_3 < idx_end_3:
            energy_at_third = np.max(spectrum[idx_start_3:idx_end_3])
            # Qui siamo più severi, deve essere molto evidente per scendere di 3 volte
            if energy_at_third > 0.4 * peak_energy:
                return bpm_third

    return detected_bpm

def extract_heart_rate_fft_no_check(signal, fs, min_hz=0.7, max_hz=2.5):
    """
    Esegue FFT su un segnale 1D e trova il picco di frequenza dominante.
    """
    n = len(signal)
    
    # Finestra di Hanning per ridurre leakage spettrale
    windowed_signal = signal * np.hanning(n)
    
    # FFT Reale
    fft_spectrum = np.fft.rfft(windowed_signal)
    fft_freqs = np.fft.rfftfreq(n, d=1/fs)
    
    # Calcola magnitudo dello spettro
    magnitude = np.abs(fft_spectrum)
    
    # Maschera per cercare solo nel range fisiologico (es. 42 - 150 BPM)
    valid_mask = (fft_freqs >= min_hz) & (fft_freqs <= max_hz)
    
    if not np.any(valid_mask):
        return 0.0 # Nessun picco valido trovato
    
    # Trova il picco massimo nella regione valida
    masked_magnitude = magnitude[valid_mask]
    masked_freqs = fft_freqs[valid_mask]
    
    peak_idx = np.argmax(masked_magnitude)
    peak_freq = masked_freqs[peak_idx]
    
    return peak_freq * 60.0 # Converti Hz in BPM

def extract_heart_rate_fft(signal, fs, min_hz=0.7, max_hz=2.5):
    """
    Esegue FFT su un segnale 1D, rimuove transitori e gestisce armoniche.
    """
    # 1. Rimuovi transitori del filtro (es. primi e ultimi 2 secondi)
    # Non più fatto

    n = len(signal)
    
    # 2. Zero Padding per aumentare risoluzione FFT (interpolazione spettrale)
    n_padded = 2048 
    
    # Finestra di Hanning
    windowed_signal = signal * np.hanning(n)
    
    # FFT con padding
    fft_spectrum = np.fft.rfft(windowed_signal, n=n_padded)
    fft_freqs = np.fft.rfftfreq(n_padded, d=1/fs)
    magnitude = np.abs(fft_spectrum)
    
    # 3. Maschera per range fisiologico
    valid_mask = (fft_freqs >= min_hz) & (fft_freqs <= max_hz)
    
    if not np.any(valid_mask):
        return 0.0

    masked_magnitude = magnitude[valid_mask]
    masked_freqs = fft_freqs[valid_mask]
    
    # 4. Trova picco massimo (candidato iniziale)
    peak_idx = np.argmax(masked_magnitude)
    peak_bpm = masked_freqs[peak_idx] * 60.0
    
    # 5. Controllo Armoniche (Anti-raddoppio)
    # Passiamo l'intero spettro e frequenze per analizzare i picchi secondari
    final_bpm = check_harmonics(peak_bpm, magnitude, fft_freqs)
    
    return final_bpm

def evaluate_component_quality(signal, fs, min_hz=0.85, max_hz=2.5):
    """
    Calcola FFT e restituisce il BPM rilevato E un punteggio di qualità (SNR).
    """
    # 1. Pre-processing (Taglio e Finestra)
    trim = int(2 * fs)
    if len(signal) > (2 * trim + 10):
        signal = signal[trim:-trim]
    else:
        signal = signal[5:-5]
        
    n = len(signal)
    n_padded = 2048
    windowed = signal * np.hanning(n)
    fft_spec = np.abs(np.fft.rfft(windowed, n=n_padded))
    freqs = np.fft.rfftfreq(n_padded, d=1/fs)
    
    # 2. Focus sul range cardiaco
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(mask):
        return 0, 0.0 # Nessun segnale valido

    band_spectrum = fft_spec[mask]
    band_freqs = freqs[mask]
    
    # 3. Trova il picco massimo
    idx_max = np.argmax(band_spectrum)
    peak_val = band_spectrum[idx_max]
    detected_freq = band_freqs[idx_max]
    
    # 4. Calcola SNR (Signal-to-Noise Ratio)
    # Il "rumore" è la media dell'energia nella banda, escludendo il picco stesso
    avg_noise = (np.sum(band_spectrum) - peak_val) / (len(band_spectrum) - 1) if len(band_spectrum) > 1 else 1.0
    
    if avg_noise == 0: avg_noise = 1e-6 # Evita divisione per zero
    
    snr = peak_val / avg_noise
    
    return detected_freq * 60.0, snr

# --- PIPELINE COMPLETE ---

def process_pipeline_amplitude(rebuilt_csi, fs=20):
    """
    Pipeline A: Usa solo l'ampiezza (più robusta, meno sensibile).
    """

    # 1. Calcola Magnitudo
    amp_matrix = np.abs(rebuilt_csi)
    
    # 2. Selezione Sottoportanti (Top 10 per varianza)
    selected_matrix = select_best_subcarriers(amp_matrix, keep_n=10)
    
    # 3. PCA per estrarre componenti principali
    # Usiamo 3 componenti per essere sicuri di catturare respiro e cuore
    pca = PCA(n_components=3)
    components = pca.fit_transform(selected_matrix)
    
    # Di solito PC1 = Respiro/Moto, PC2 = Cuore. 
    # Proviamo a usare la PC2 (indice 1)
    target_signal = components[:, 1] 
    
    # 4. Filtro Passa-Banda (Isola il cuore)
    filtered_signal = butter_bandpass_filter(target_signal, 0.85, 2.75, fs)
    
    # 5. Stima BPM
    bpm = extract_heart_rate_fft(filtered_signal, fs)
    
    return bpm, filtered_signal

def process_pipeline_phase(rebuilt_csi, fs=20):
    """
    Pipeline B: Usa la fase sanitizzata (più sensibile, richiede pulizia).
    """
    # 1. Sanitizzazione Fase (Rimuove errore CFO/SFO)
    # Ritorna già la fase pulita, non complessi
    clean_phase_matrix = sanitize_phase(rebuilt_csi)

    # 2. Selezione Sottoportanti
    selected_matrix = select_best_subcarriers(clean_phase_matrix, keep_n=10)

    # 3. PCA
    pca = PCA(n_components=3)
    components = pca.fit_transform(selected_matrix)

    # Usiamo PC2 (indice 1)
    target_signal = components[:, 1]

    # 4. Filtro
    filtered_signal = butter_bandpass_filter(target_signal, 0.85, 2.75, fs, 5)

    # 5. Stima BPM
    bpm = extract_heart_rate_fft(filtered_signal, fs)

    return bpm, filtered_signal

def process_pipeline_adaptive_phase(rebuilt_csi, fs=20):
    """
    Pipeline C (Adattiva): Usa la Fase, ma cerca il cuore in PC1, PC2, PC3 o PC4
    scegliendo il segnale più 'pulito'.
    """
    # 1. Sanitizzazione
    clean_phase = sanitize_phase(rebuilt_csi)
    
    # 2. Selezione Sottoportanti
    selected_matrix = select_best_subcarriers(clean_phase, keep_n=10)
    
    # 3. PCA (Estraiamo 4 componenti per sicurezza)
    # Se il segnale è molto rumoroso, il cuore potrebbe finire nella PC3 o PC4
    pca = PCA(n_components=4)
    components = pca.fit_transform(selected_matrix)
    
    best_bpm = 0.0
    best_snr = -1.0
    best_comp_idx = -1
    
    # 4. Gara tra le componenti: chi ha il picco più bello?
    for i in range(1, 3):
        # Estrai componente i
        raw_sig = components[:, i]
        
        # Filtra
        filtered_sig = butter_bandpass_filter(raw_sig, 0.85, 2.75, fs, 5)
        
        # Valuta Qualità
        bpm, snr = evaluate_component_quality(filtered_sig, fs, min_hz=0.85, max_hz=2.5)
        
        # Debug (opzionale): vedi quale componente sta vincendo
        # print(f"  PC{i+1}: BPM={bpm:.1f}, SNR={snr:.2f}")
        
        if snr > best_snr:
            best_snr = snr
            best_bpm = bpm
            best_comp_idx = i
            
    winner_sig = butter_bandpass_filter(components[:, best_comp_idx], 0.85, 2.5, fs)
    final_bpm = extract_heart_rate_fft(winner_sig, fs)
    
    return final_bpm, winner_sig, best_comp_idx

#region ======= DSP process =======

def PROC_DSP(mpdeque_dsp: MPDeque, mpdeque_hr: MPDeque, log_to_file: bool = False, log_filename: str = "log.log"):


    while True:
        csi_complex_matrix = mpdeque_dsp.popright(block=True)

        bpm_amp, signal_amp = process_pipeline_amplitude(csi_complex_matrix, fs=20)
        bpm_pha, signal_pha = process_pipeline_phase(csi_complex_matrix, fs=20)
        bpm_pca, _, best_comp_idx = process_pipeline_adaptive_phase(csi_complex_matrix, fs=20)

        #print(f"Stima BPM (Ampiezza): {bpm_amp:.1f}")
        #print(f"Stima BPM (Fase): {bpm_pha:.1f}")
        #print(f"Stima BPM (PCA): {bpm_pca:.1f} (Component IDX {best_comp_idx})")

        print(f"{bpm_amp:3.1f} / {bpm_pha:3.1f} / {bpm_pca:3.1f} - (amp / phase / pca)")
                
        mpdeque_hr.appendleft(bpm_amp)
            
#endregion