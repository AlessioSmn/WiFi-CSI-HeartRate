import sys
import argparse
import numpy as np
import serial
from io import StringIO
import ast
from scipy.signal import butter, filtfilt, savgol_filter
from typing import List, Tuple, Optional, Dict, Any
import tensorflow as tf
from tensorflow import keras
import numpy as np
from collect_data import *
from raspberry.features_train import extract_features
import threading


DATA_COLUMNS_NAMES = ["type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
                      "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"]
HR_COLUMNS_NAMES = ["IR", "BPM", "AVG BPM"]
CSI_DATA_LENGTH = 384               # esp32 exposes only 192 subcarriers, each carrier has associated I/Q components, so 192 x 2 = 384
DC_REMOVAL_WINDOW_LENGTH = 100
SAMPLING_FREQUENCY = 20
SEGMENTATION_WINDOW_LENGTH = 100    # WINDOW_SIZE


stop_event = threading.Event()

new_data_event = threading.Event()
buffer_csi = []
buffer_csi_lock = threading.Lock()

new_input_event = threading.Event()
input_lstm = None
input_lstm_lock = threading.Lock()

model = keras.models.load_model(f"models/csi_hr_{SEGMENTATION_WINDOW_LENGTH}.keras", safe_mode=False)

def csi_read_thread(port):
    global new_data_event
    global buffer_csi
    global stop_event
    ser = serial.Serial(port=port, baudrate=115200,bytesize=8, parity='N', stopbits=1)
    if ser.isOpen():
        print("open success")
    else:
        print("open failed")
        return
    
    rows_count = 0
    print(f"Gathering data...")
    while not stop_event.is_set():
        outcome, strings, _ = iterate_data_rcv(ser, None, None, None, True)
        if outcome is None:
            break
        if outcome == False:
            continue

        with buffer_csi_lock:
            buffer_csi.append(strings)
            rows_count += 1

            if rows_count <= SEGMENTATION_WINDOW_LENGTH:
                print(f"gathering data... {rows_count}/{SEGMENTATION_WINDOW_LENGTH}")
                continue
            buffer_csi.pop(0)
        
        rows_count -= 1
        new_data_event.set()
    
    ser.close()

def csi_process_thread():
    global input_lstm
    global new_input_event
    global new_data_event
    global buffer_csi
    global buffer_csi_lock
    global stop_event
    settings = {}
    settings["training_phase"] = False
    settings["verbose"] = False
    settings["csi_data_length"] = CSI_DATA_LENGTH
    settings["sampling_frequency"] = SAMPLING_FREQUENCY
    settings["segmentation_window_length"] = SEGMENTATION_WINDOW_LENGTH

    while not stop_event.is_set():
        new_data_event.wait()
        new_data_event.clear()

        with buffer_csi_lock:
            buffer = buffer_csi.copy()
        df = from_buffer_to_df_detection(buffer, DATA_COLUMNS_NAMES)
        window = extract_features(df, settings)
        if len(window) != 1:
            continue

        with input_lstm_lock:
            input_lstm = window.copy()
        new_input_event.set()

def prediction_thread():
    global input_lstm
    global new_input_event
    global stop_event
    while not stop_event.is_set():
        new_input_event.wait()
        new_input_event.clear()

        with input_lstm_lock:
            window = input_lstm.copy()
        
        new_prediction = model.predict(window, verbose=0)
        print(new_prediction[0][0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port and display it graphically")
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of csv_recv device")
    parser.add_argument('-l', '--log', dest="log_file", action="store", default="./csi_data_log.txt",
                        help="Save other serial data the bad CSI data to a log file")

    args = parser.parse_args()
    serial_port = args.port
    
    t_read = threading.Thread(target=csi_read_thread, args=(serial_port,))
    t_process = threading.Thread(target=csi_process_thread)
    t_pred = threading.Thread(target=prediction_thread)

    t_read.start()
    t_process.start()
    t_pred.start()
    try:
        t_read.join()
        t_process.join()
        t_pred.join()
    except KeyboardInterrupt:
        print("Closing...")
        stop_event.set()