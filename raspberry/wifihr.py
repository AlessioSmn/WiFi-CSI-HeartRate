
import json
import time
from collections import deque
from multiprocessing import Process, Queue
from queue import Empty


from csisource import csi_data_source_init, get_csi_data, csi_data_source_close, fecth_csi_data_proc
from dsp import parse_csi_amplitudes, estimate_hr_freq, dsp_process
from hrui import start_plotting, push_new_hr
from log import print_log, set_print_level, LVL_DBG, LVL_INF, LVL_ERR, DebugLevel

DATA_COLUMNS_NAMES_C5C6 = ["type", "id", "mac", "rssi", "rate","noise_floor","fft_gain","agc_gain", "channel", "local_timestamp",  "sig_len", "rx_state", "len", "first_word", "data"]

# 24+1 fields (24 + 1 data field, it's array)
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]


hr_estimates = None


def Hz_to_BPM(hz: float) -> float:
    return hz * 60.0

def monitor_hr(
        port: str,
        from_serial: bool = True,
        print_level: str = 'info'
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
    LOG = False

    sampling_freq = 40 # Hertz

    # BP_order = 3
    # BP_hr_all = 0.15
    # SG_polyorder = 3
    # SG_winlen = 15
    # TOP_carr = 35
    # HR_MIN = 45
    # HR_MAX = 200
    window_len = 400 # TODO tuning
    iter_per_estimate = 20 # If equal to sampling frequency it means an estimate for second
    
    if LOG:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("log.log", "a") as f:
            f.write(f"\n[{timestamp}] NEW SESSION\n")
            f.write(" --- Parameters:\n")
            f.write(f"Sample window length = [{window_len}]\n")
            f.write(f"Iterations per estimate = [{iter_per_estimate}]\n")
            f.write(f"Bandpass order = [{BP_order}]\n")
            f.write(f"Bandpass hr allowance = [{BP_hr_all}]\n")
            f.write(f"SG polyorder = [{SG_polyorder}]\n")
            f.write(f"SG window len = [{SG_winlen}]\n")
            f.write(f"Top carriers = [{TOP_carr}]\n")
            f.write(f"HR Min = [{HR_MIN}]\n")
            f.write(f"HR Min = [{HR_MAX}]\n")
            f.write(" --- Results: (mean / median)\n")

    #endregion
    
    #region ======= Processes =======

    # (1) === Connect to data source
    csi_data_source_init(port=port, from_ser=from_serial)
    print_log("Source initiated", LVL_INF)

    # (2) === CSI data source process
    # csi_data_queue = deque(maxlen=100)
    # csi_proc = Process(
    #     target=fecth_csi_data_proc,
    #     args=(port, from_serial, csi_data_queue),
    #     daemon=True
    # )
    # csi_proc.start()
    # print_log("CSI data source process started", LVL_INF)

    # (3) === Plotting process
    hr_queue = Queue()
    
    # iter_per_estimate / sampling_freq -> number of iterations per second
    refresh_time_ms = int(1000 * iter_per_estimate / sampling_freq)
    print_log(f"UI process refresh interval: {refresh_time_ms} ms", LVL_INF)
    ui_proc = Process(
        target=start_plotting,
        args=(hr_queue,refresh_time_ms),
        daemon=True
    )
    # ui_proc.start()
    print_log("UI process started", LVL_INF)

    csi_data_queue = Queue()
    hr_result_queue = Queue()
    dsp_proc = Process(
        target=dsp_process,
        args=(csi_data_queue,hr_result_queue),
        daemon=True
    )
    dsp_proc.start()


    #endregion
    
    #region ======= Main loop =======
    
    # Number of iteration since last HR estimate
    iter = 0

    # Sliding window of csi data arrays
    csi_data_window = deque(maxlen=window_len)

    # Overall iteration counter
    frame_num = 0
    _last_id = -1

    try:
        while True:
            frame_num += 1

            # handle any data, if ready
            try:
                hr_hz = hr_result_queue.get_nowait()
                
                # Convert to BPM
                hr_bpm = Hz_to_BPM(hr_hz)
                
                print_log(f"(main loop) - Heart rate estimated: {hr_bpm:.2f} ({hr_hz:.3f})", LVL_INF)
                
                # Append to array (in BPM)
                hr_queue.put(Hz_to_BPM(hr_hz))

            except Empty:
                # Data not ready
                pass

            # Get CSI data
            try:
                csi_data = get_csi_data()
                # csi_data = csi_data_queue.pop()
            except ValueError as ve:
                #if ve != ValueError(-2):
                    #print_log(f"(VE) Error in getting csi data: {ve}", LVL_ERR)
                continue
            except IndexError as e:
                print_log(f"(IE) Error in getting csi data: {e}", LVL_ERR)
                break
            except RuntimeError as e:
                print_log("(RE) Error in getting csi data", LVL_ERR)
                continue

            if _last_id != -1 and _last_id != (int(csi_data[1]) - 1):
                print_log(f"(main loop) - csi data MISSING  (ID={_last_id+1})", LVL_DBG)

            print_log(f"(main loop) - csi data received (ID={csi_data[1]})", LVL_DBG)
            _last_id = int(csi_data[1])

            # Ensure message length is recognized (among two standards)
            if len(csi_data) != len(DATA_COLUMNS_NAMES) and len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
                print_log(f"(main loop) - Message length is not recognized: len is = {len(csi_data)}, can be {len(DATA_COLUMNS_NAMES) } or {len(DATA_COLUMNS_NAMES_C5C6)}", LVL_ERR)
                continue

            print_log("(main loop) - correct message length", LVL_DBG)

            # Ensure correct length
            if int(csi_data[-3]) != len(csi_data[-1]):
                print_log(f"(main loop) - Data length does not coincide: len is = {len(csi_data[-1])}, advertised as {int(csi_data[-3])}", LVL_ERR)
                continue
            
            print_log("(main loop) - correct data length", LVL_DBG)

            # Calculate amplitudes
            amplitudes = parse_csi_amplitudes(csi_data[-1])
            print_log("(main loop) - amplitudes calculated", LVL_DBG)

            # Add to array
            csi_data_window.append(amplitudes)
            print_log("(main loop) - amplitudes appended list", LVL_DBG)

            # If sufficient iteration reached, process the current matrix
            if iter >= iter_per_estimate:

                if window_len > len(csi_data_window):
                    continue

                csi_data_queue.put(csi_data_window)
                
                # Estimate heart rate
                """
                hr_hz = estimate_hr_freq(
                    signal_matrix=csi_data_window,
                    fs=sampling_freq,
                    top_carriers=TOP_carr,
                    aggr_method='mean',
                    hr_min=HR_MIN,
                    hr_max=HR_MAX,
                    par_bp_order=BP_order,
                    par_bp_hr_allowance=BP_hr_all,
                    par_sg_order=SG_polyorder,
                    par_sg_winlen=SG_winlen
                )
                
                # Convert to BPM
                hr_bpm = Hz_to_BPM(hr_hz)
                
                print_log(f"(main loop) - Heart rate estimated: {hr_bpm:.2f} ({hr_hz:.3f})", LVL_INF)
                
                # Append to array (in BPM)
                hr_queue.put(Hz_to_BPM(hr_hz))

                # push_new_hr(Hz_to_BPM(hr_hz))

                if LOG:
                    # Estimate also median heart rate
                    hr_hz_median = estimate_hr_freq(
                        signal_matrix=csi_data_window,
                        fs=sampling_freq,
                        top_carriers=10,
                        aggr_method='median',
                        hr_min=HR_MIN,
                        hr_max=HR_MAX,
                        par_bp_order=BP_order,
                        par_bp_hr_allowance=BP_hr_all,
                        par_sg_order=SG_polyorder,
                        par_sg_winlen=SG_winlen)
                
                    with open("log.log", "a") as f:
                        f.write(f"{Hz_to_BPM(hr_hz):.2f} / {Hz_to_BPM(hr_hz_median):.2f}\n")
                
                    
                """
                iter = 0

            else:
                iter += 1
    except KeyboardInterrupt:
        print_log("Process interrupted", LVL_INF)

    #endregion

    #region ======= Closing resources =======
    
    # Close the source
    csi_data_source_close()
    print_log("Source closed", LVL_INF)
    
    #endregion

    return
