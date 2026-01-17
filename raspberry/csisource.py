import csv
import ast
import serial
from io import StringIO
from collections import deque

from log import print_log, LVL_DBG, LVL_INF, LVL_ERR

# Serial port number
port = None

# Indicates wheter to get data from serial or csv
from_serial = True

# Serial connection
ser = None

# Csv list
csv_array = []
csv_len = 0

# Current csv index
csv_index = 0

def csi_data_source_init(port, from_ser: bool = True):
    """
    Initializes the source of the data (serial or csv)
    
    :param port: Serial port or csv filename
    :param from_ser: Indicates wheter to get data from serial or csv
    :type from_ser: bool
    """
    global ser, csv_array, from_serial, csv_len

    from_serial = from_ser

    if from_serial:
        ser = serial.Serial(port=port, baudrate=921600, timeout=1)
        if ser.isOpen():
            print_log("Serial port opened successfully", LVL_INF)
            print_log(f"Port     = {port}", LVL_DBG)
            print_log(f"Baudrate = {921600}", LVL_DBG)
            # print_log(f"Bytesize = {8}", LVL_DBG)
            # print_log(f"Parity   = {'N'}", LVL_DBG)
            # print_log(f"stopbits = {1}", LVL_DBG)
        else:
            print("Serial port - Open failed", LVL_ERR)
            return -1
        
    else:
        with open(port, newline='') as csvfile:
            full_data = list(csv.reader(csvfile, delimiter=','))
            for data_row in full_data:
                csv_array.append(data_row)

            csv_len = len(full_data)
                
            print_log(f"CSV lines read: {csv_len}", LVL_DBG)

def csi_data_source_close(dbg_print: bool = False):
    """
    Closes the csi data source
    
    :param dbg_print: Indicates wheter to print debug information
    :type dbg_print: bool
    """

    global from_serial, ser
    
    if from_serial:
        ser.close()
        if dbg_print:
            print_log("Serial closed successfully", LVL_INF)


def get_csi_data():
    global from_serial, ser, csv_array, csv_index, csv_len

    if from_serial:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        
        if "CSI_DATA" not in line:
            raise ValueError(-2)
        
        head, tail = line.rsplit(',"', 1)
        fields = head.split(',')

        try:
            data_arr = ast.literal_eval(tail[:-1])
        except SyntaxError as se:
            print(line)
            return fields
        
        fields.append(data_arr)

        return fields
    
    else:

        print_log(f"get_csi_data - Reading CSV line of index {csv_index}", LVL_DBG)
            
        # csi_cvs_row = csv_array[csv_index % csv_len]
        csi_cvs_row = csv_array[csv_index]

        # Suppose (hardcoded) 25 fields
        header_arr = csi_cvs_row[0:23]
        
        # Convert data fields to int
        data_arr = [int(x) for x in csi_cvs_row[23:]]

        print_log(f"get_csi_data - Header fields: [{','.join(map(str, header_arr))}]", LVL_DBG)
        
        csi_data = ["CSI_DATA"] + header_arr + [data_arr]

        print_log(f"get_csi_data - Final array len: {len(csi_data)}", LVL_DBG)
        
        csv_index += 1
        return csi_data
    

def fecth_csi_data_proc(port: str, from_ser: bool, hr_queue: deque):

    csi_data_source_init(port=port, from_ser=from_ser)

    while True:

        # get CSI data
        try:
            csi_data = get_csi_data()
        except ValueError as ve:
            if ve == 1:
                print_log("Error in get_csi_data", LVL_ERR)
            continue

        # Put into queue
        hr_queue.appendleft(csi_data)

        # Debug print
        print_log(f"Current CSI data queue length: {len(hr_queue)}", LVL_INF)
    