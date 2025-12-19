
import json
import time
from collections import deque
from multiprocessing import Process, Queue


from csisource import csi_data_source_init, get_csi_data, csi_data_source_close
from dsp import parse_csi_amplitudes, estimate_hr_freq
from hrui import start_plotting, push_new_hr

DATA_COLUMNS_NAMES_C5C6 = ["type", "id", "mac", "rssi", "rate","noise_floor","fft_gain","agc_gain", "channel", "local_timestamp",  "sig_len", "rx_state", "len", "first_word", "data"]

# 24+1 fields (24 + 1 data field, it's array)
DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]


hr_estimates = None


def Hz_to_BPM(hz: float) -> float:
    return hz * 60.0

def monitor_hr(port: str, csv_writer, log_file_fd):

    LOG = True

    sampling_freq = 20 # Hertz

    BP_order = 3
    BP_hr_all = 0.0
    SG_polyorder = 2
    SG_winlen = 11
    TOP_carr = 15
    HR_MIN = 45
    HR_MAX = 200
    window_len = 600 # TODO tuning
    iter_per_estimate = 10 # If equal to sampling frequency it means an estimate for second
    
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

    iter = 0

    # Sliding window of csi data arrays
    csi_data_window = deque(maxlen=window_len)
 
    csi_data_source_init(port=port, dbg_print=True, from_ser=False)

    # Start plottin HR data (separate process)
    hr_queue = Queue()
    ui_proc = Process(
        target=start_plotting,
        args=(hr_queue,500),
        daemon=True
    )
    ui_proc.start()

    frame_num = 0
    
    while True:
        frame_num += 1

        # Get CSI data
        try:
            csi_data = get_csi_data(log_file_fd)
        except IndexError as ie:
            break

        # Ensure message length is recognized (among two standards)
        if len(csi_data) != len(DATA_COLUMNS_NAMES) and len(csi_data) != len(DATA_COLUMNS_NAMES_C5C6):
            print(f"element number is not equal: {len(csi_data)} vs {len(DATA_COLUMNS_NAMES)}")
            # print(csi_data)
            log_file_fd.write("element number is not equal\n")
            # log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        # Load CSI Data array from string
        try:
            csi_raw_data = json.loads(csi_data[-1])
        except json.JSONDecodeError:
            # print("data is incomplete")
            log_file_fd.write("data is incomplete\n")
            # log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        # Ensure correct length
        csi_data_len = int (csi_data[-3])
        if csi_data_len != len(csi_raw_data):
            # print("csi_data_len is not equal",csi_data_len,len(csi_raw_data))
            log_file_fd.write("csi_data_len is not equal\n")
            print(f"csi_data_len is not equal: csi_data_len={csi_data_len} , len(csi_raw_data)={len(csi_raw_data)}")
            # log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            continue

        # Calculate amplitudes
        amplitudes = parse_csi_amplitudes(csi_data[24])

        # Add to array
        csi_data_window.append(amplitudes)

        # If sufficient iteration reached, process the current matrix
        if iter >= iter_per_estimate:

            if window_len > len(csi_data_window):
                continue

            # Estimate heart rate
            hr_hz = estimate_hr_freq(
                signal_matrix=csi_data_window,
                fs=sampling_freq,
                top_carriers=10,
                aggr_method='mean',
                hr_min=HR_MIN,
                hr_max=HR_MAX,
                par_bp_order=BP_order,
                par_bp_hr_allowance=BP_hr_all,
                par_sg_order=SG_polyorder,
                par_sg_winlen=SG_winlen)
            
            # Append to array (in BPM)
            hr_queue.put(Hz_to_BPM(hr_hz))

            push_new_hr(Hz_to_BPM(hr_hz))

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
            
            iter = 0

        else:
            iter += 1

    # Close the source
    csi_data_source_close()
    return
