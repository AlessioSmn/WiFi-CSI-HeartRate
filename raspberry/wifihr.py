#region ======= Imports =======
from collections import deque
from multiprocessing import Process, Manager, Condition, Queue

from mpque import MPDeque
from csisource import csi_data_source_init, get_csi_data, csi_data_source_close
from dsp import parse_csi_amplitudes, resample_window, PROC_DSP
from hrui import PROC_UI
from log import print_log, set_print_level, LVL_DBG, LVL_INF, LVL_ERR, DebugLevel
#endregion ======= Imports =======

#region ======= Utility functions, headers =======

def print_log_loc(s: str, log_level: DebugLevel = DebugLevel.INFO):
    s_loc = f"[MAIN] {s}"
    print_log(s_loc, log_level)

def Hz_to_BPM(hz: float) -> float:
    return hz * 60.0

DATA_COLUMNS_NAMES_C5C6 = ["type", "id", "mac", "rssi", "rate","noise_floor","fft_gain","agc_gain", "channel", "local_timestamp",  "sig_len", "rx_state", "len", "first_word", "data"]

# 24+1 fields (24 + 1 data field, it's array)
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]

#endregion

#region ======= MAIN APPLICATION FUNCTION (CONTROLLER) =======

def monitor_hr(
        port: str,
        from_serial: bool = True,
        print_level: str = 'info',
        start_ui_proc: bool = True,
        log_filename: str | None = None
    ):
    
    #region ======= Arguments check =======

    if print_level not in ['debug', 'info', 'error']:
        print("ERROR - print level not recognized. Should be one of 'debug', 'info', 'error'")
        return -1
    
    level_map = {
        'debug': DebugLevel.DEBUG,
        'info': DebugLevel.INFO,
        'error': DebugLevel.ERROR
    }

    set_print_level(level_map[print_level])
    
    #endregion

    #region ======= Parameters =======

    # Sampling frequency
    sampling_freq = 40 # Hertz

    # Csi data window lemgth
    window_len = 500 # TODO tuning

    # Samples taken between two estimates
    # If equal to sampling frequency it means an estimate per second
    iter_per_estimate = 40 
    
    #endregion
    
    #region ======= Processes =======

    child_processes = []

    manager = Manager()

    # === Process DSP (analyze signals)

    list_dsp = manager.list()
    cond_dsp = Condition()
    mpdeque_dsp = MPDeque(list_dsp, cond_dsp, maxlen=512)

    list_hr = manager.list()
    cond_hr = Condition()
    mpdeque_hr = MPDeque(list_hr, cond_hr, maxlen=512)

    if log_filename:
        log_to_file = True
    else:
        log_to_file = False

    PROC_dsp = Process(
        target=PROC_DSP,
        args=(mpdeque_dsp, mpdeque_hr, log_to_file, log_filename),
        name="DSP",
        daemon=True
    )
    PROC_dsp.start()
    child_processes.append(PROC_dsp)
    print_log_loc("Process DSP started", LVL_INF)

    # === Process UI (display results)

    list_ui = manager.list()
    cond_ui = Condition()
    mpdeque_ui = MPDeque(list_ui, cond_ui, maxlen=512)
    
    if start_ui_proc:

        refresh_ms = int(1000 * iter_per_estimate / sampling_freq)

        PROC_ui = Process(
            target=PROC_UI,
            args=(mpdeque_ui,refresh_ms),
            name="UI",
            daemon=True
        )
        PROC_ui.start()
        child_processes.append(PROC_ui)
        print_log_loc("Process UI started", LVL_INF)

    #endregion
    
    #region ======= Main loop =======
    
    # Number of iteration since last HR estimate
    iter = 0

    # Sliding window of csi data arrays (+ counters)
    csi_data_window_counters = deque(maxlen=window_len)

    # Id of last received sample
    _last_id = -1

    csi_data_source_init(port=port, from_ser=from_serial)

    print_log_loc("Application started, use CTRL+C to stop", LVL_INF)

    try:
        while True:
            
            #region ======= Estimates retrieval =======

            try:
                hr_hz = mpdeque_hr.popright(block=False)
                
                # Convert to BPM
                hr_bpm = Hz_to_BPM(hr_hz)
                
                print_log_loc(f"(main loop) - Heart rate estimated: {hr_bpm:.2f} BPM ({hr_hz:.3f} Hz)", LVL_INF)
                
                # Put in queue (in BPM)
                mpdeque_ui.appendleft(Hz_to_BPM(hr_hz))
            
            except IndexError:
                pass

            #endregion

            #region ======= CSI Data preprocessing =======
            try:
                csi_data = get_csi_data()

                # Track missimg samples (for debugging)
                if _last_id != -1 and _last_id != (int(csi_data[1]) - 1):
                    print_log_loc(f"(main loop) - csi sample missing (ID={_last_id+1})", LVL_DBG)
                _last_id = int(csi_data[1])

                # Ensure message length is recognized (among two standards)
                if len(csi_data) != len(DATA_COLUMNS_NAMES) and len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
                    print_log_loc(f"(main loop) - Message length is not recognized: Len is = {len(csi_data)}, can be {len(DATA_COLUMNS_NAMES) } or {len(DATA_COLUMNS_NAMES_C5C6)}", LVL_ERR)
                    continue
                print_log_loc("(main loop) - correct message length", LVL_DBG)

                # Ensure correct length
                if int(csi_data[-3]) != len(csi_data[-1]):
                    print_log_loc(f"(main loop) - Data length does not coincide: Len is = {len(csi_data[-1])}, advertised as {int(csi_data[-3])}", LVL_ERR)
                    continue
            
                print_log_loc("(main loop) - correct data length", LVL_DBG)

                # Calculate amplitudes
                amplitudes = parse_csi_amplitudes(csi_data[24])
                print_log_loc("(main loop) - amplitudes calculated", LVL_DBG)

                # Add to array
                msg_id = int(csi_data[1])
                csi_data_window_counters.append((msg_id, amplitudes))
                print_log_loc("(main loop) - amplitudes appended list", LVL_DBG)

                # If sufficient iteration reached, process the current matrix
                if iter >= iter_per_estimate:

                    if window_len > len(csi_data_window_counters):
                        continue

                    cdw_resampled = resample_window(csi_data_window_counters)
                    mpdeque_dsp.appendleft(cdw_resampled)
                    
                    # Reset iteration counter
                    iter = 0

                else:
                    iter += 1

            except ValueError as ve:
                if ve.args[0] == -3:
                    print_log_loc(f"Error in getting csi data: empty line", LVL_ERR)

                elif ve.args[0] == -2: # initial lines, system infp
                    pass

                else:
                    print_log_loc(f"Error in getting csi data: {ve}", LVL_ERR)
                continue
            except RuntimeError as e:
                print_log_loc(f"Runtime error in getting csi data: {e}", LVL_ERR)
                continue
            except IndexError:
                pass

            #endregion
    
    except KeyboardInterrupt:
        print_log_loc("Application shutdown...", LVL_INF)

    #endregion

    #region ======= Closing resources =======

    # Close the CSI source
    csi_data_source_close()
    print_log_loc("CSI Source closed", LVL_INF)

    # Send SIGTERM
    for p in child_processes:
        print_log_loc(f"Sending terminate signal to {p.name} process")
        p.terminate()

    # Wait for process end
    for p in child_processes:
        p.join(timeout=10)
        if p.is_alive():
            print_log_loc(f"{p.name} process did not terminate in time")
        else:
            print_log_loc(f"{p.name} process terminated")
    
    print_log_loc("Application closed successfully", LVL_INF)
    
    #endregion

    return

#endregion