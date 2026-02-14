#region ======= Imports =======

import ast
import csv
import serial
import time
from io import StringIO
from collections import deque

from mpque import MPDeque
from log import print_log, LVL_DBG, LVL_INF, LVL_ERR, DebugLevel

#endregion

#region ======= Parameters =======

# Serial port number
PAR_SOURCE_NAME = None

# Indicates wheter to get data from serial or csv
PAR_SOURCE_SERIAL = True

# Serial connection
PAR_SER_CONN = None

# Csv list
PAR_CSV_ARRAY = []
PAR_CSV_LEN = 0
# Current csv index
PAR_CSV_INDEX = 0

#endregion

#region ======= Functions =======

def print_log_loc(s: str, log_level: DebugLevel = DebugLevel.INFO):
    s_loc = f"[CSI ] {s}"
    print_log(s_loc, log_level)

def csi_data_source_init(port, from_ser: bool = True):
    """
    Initializes the source of the data (serial or csv)
    
    :param port: Serial port or csv filename
    :param from_ser: Indicates wheter to get data from serial or csv
    :type from_ser: bool
    """
    global PAR_SER_CONN, PAR_SOURCE_SERIAL, PAR_CSV_ARRAY, PAR_CSV_LEN, PAR_CSV_INDEX

    PAR_SOURCE_SERIAL = from_ser

    if PAR_SOURCE_SERIAL:
        PAR_SER_CONN = serial.Serial(port=port, baudrate=921600, timeout=1, bytesize=8, parity='N', stopbits=1)
        if PAR_SER_CONN.isOpen():
            print_log_loc("Serial port opened successfully", LVL_INF)
            print_log_loc(f"Port     = {port}", LVL_DBG)
            print_log_loc(f"Baudrate = {921600}", LVL_DBG)
            print_log_loc(f"Bytesize = {8}", LVL_DBG)
            print_log_loc(f"Parity   = {'N'}", LVL_DBG)
            print_log_loc(f"stopbits = {1}", LVL_DBG)
        else:
            print("Serial port - Open failed", LVL_ERR)
            return -1
        
    else:
        with open(port, "r", newline="") as f:
            counter = 1
            for line in f:
                line = line.rstrip("\n")

                reader = csv.reader(StringIO(line))
                fields = next(reader)

                if fields[0] != "CSI_DATA":
                    csi_raw = fields[1]
                    headers = f'CSI_DATA,{counter},1a:00:00:00:00:00,-64,11,1,0,1,1,1,0,0,0,0,-96,0,11,2,30846,0,47,0,384,1,'
                    data = f'"{csi_raw}"'
                    line = f"{headers}{data}\n"
                    counter+=1
                PAR_CSV_ARRAY.append(line.rstrip("\n"))

            PAR_CSV_LEN = len(PAR_CSV_ARRAY)
            PAR_CSV_INDEX = 0
                
            print_log_loc(f"CSV lines read: {PAR_CSV_LEN}", LVL_DBG)
            # print(f"CSV lines read: {PAR_CSV_LEN}")
            #print(PAR_CSV_ARRAY[0])
            #print(PAR_CSV_ARRAY[1])
            #print(PAR_CSV_ARRAY[2])

def csi_data_source_close():
    """
    Closes the csi data source
    """

    global PAR_SOURCE_SERIAL, PAR_SER_CONN
    
    if PAR_SOURCE_SERIAL:
        PAR_SER_CONN.close()
        print_log_loc("Serial closed successfully", LVL_INF)

def get_csi_data():
    """
    Reads a single CSI data sample either from a serial port or a preloaded CSV array.

    The function handles two data sources:
    1. Serial: reads a line from the serial connection.
    2. CSV: reads the next line from a loaded CSV array, emulating the original 40 Hz sampling rate.

    Performs basic checks:
    - Raises ValueError(-3) if the line is empty (e.g., transmitter not alive).
    - Raises ValueError(-2) if the line does not contain "CSI_DATA" (e.g., initial ESP info).

    Parses the CSI amplitude array from the line if present.

    :return: List containing CSV fields plus the parsed CSI array (if available).
    :rtype: list
    :raises ValueError: -2 if line is not CSI_DATA, -3 if line is empty.
    """
    global PAR_SOURCE_SERIAL, PAR_SER_CONN, PAR_CSV_ARRAY, PAR_CSV_INDEX, PAR_CSV_LEN

    if PAR_SOURCE_SERIAL:
        # read line from serial
        #line = PAR_SER_CONN.readline().decode("utf-8", errors="ignore").strip()
        line = PAR_SER_CONN.readline().decode().strip()
    else:
        line = PAR_CSV_ARRAY[PAR_CSV_INDEX]
        PAR_CSV_INDEX += 1

        # samplig frequency: 40Hz -> 0.025 sec
        # to emulate real timings
        # time.sleep(1.025)
    
    # Empty line usually means the transmitter is not alive
    if not line:
        print(f"DEBUG PRINT LINE: {line}")
        raise ValueError(-3)
    
    # Line without "CSI_DATA" is usually initial information from ESP
    if "CSI_DATA" not in line:
        print_log(line, LVL_INF) # print initial information on ESP
        raise ValueError(-2)

    # Split line into Header fields and CSI data array
    headers, data_arr = line.rsplit(',"', 1)
    header_fields = headers.split(',')

    try:
        # Convert CSI data string into list
        data_arr = ast.literal_eval(data_arr[:-1])
    except SyntaxError:
        print(data_arr[:-1])
        return header_fields # without data
    
    # Append the parsed CSI array to the CSV fields
    header_fields.append(data_arr)

    return header_fields

#endregion
