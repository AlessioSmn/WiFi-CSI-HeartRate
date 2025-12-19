import csv
import serial
from io import StringIO

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

def csi_data_source_init(port, dbg_print: bool = False, from_ser: bool = True):
    """
    Initializes the source of the data (serial or csv)
    
    :param port: Serial port or csv filename
    :param dbg_print: Indicates wheter to print debug information
    :type dbg_print: bool
    :param from_ser: Indicates wheter to get data from serial or csv
    :type from_ser: bool
    """
    global ser, csv_array, from_serial, csv_len

    from_serial = from_ser

    if from_serial:
        ser = serial.Serial(port=port, baudrate=921600,bytesize=8, parity='N', stopbits=1)
        if ser.isOpen():
            print("Serial port opened successfully")
            if dbg_print:
                print(f"Port     = {port}")
                print(f"Baudrate = {921600}")
                print(f"Bytesize = {8}")
                print(f"Parity   = {'N'}")
                print(f"stopbits = {1}")
        else:
            print("Open failed")
            return -1
        
    else:
        with open(port, newline='') as csvfile:
            full_data = list(csv.reader(csvfile, delimiter=','))
            for data_row in full_data:
                csv_array.append(data_row)

            csv_len = len(full_data)
                
            if dbg_print:
                print(f"Lines read: {csv_len}")

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
            print("Serial closed successfully")


def get_csi_data(log_file_fd, dbg_print: bool = False):
    global from_serial, ser, csv_array, csv_index, csv_len

    if from_serial:
        strings = str(ser.readline())
        if not strings:
            raise ValueError(-1)
        strings = strings.lstrip('b\'').rstrip('\\r\\n\'')
        index = strings.find('CSI_DATA')

        if index == -1:
            log_file_fd.write(strings + '\n')
            log_file_fd.flush()
            raise ValueError(-1)

        csv_reader = csv.reader(StringIO(strings))
        csi_data = next(csv_reader)
        return csi_data
    
    else:

        if dbg_print:
            print("get_csi_data - Reading line of index {0}".format(csv_index))
            
        # csi_cvs_row = csv_array[csv_index % csv_len]
        csi_cvs_row = csv_array[csv_index]

        if dbg_print:
            print(f"get_csi_data - Element count: {len(csi_cvs_row)}")

        # Suppose (hardcoded) 25 fields
        header_arr = csi_cvs_row[0:23]
        data_arr = csi_cvs_row[23:]

        if dbg_print:
            print(f"get_csi_data - Header fields: [{','.join(map(str, header_arr))}]")

        # Format data array as string
        data_str = "[" + ",".join(map(str, data_arr)) + "]"
        if dbg_print:
            print(f"get_csi_data - Data string: {data_str}")

        csi_data = ["CSI_DATA"] + header_arr + [data_str]

        if dbg_print:
            print(f"get_csi_data - Final array len: {len(csi_data)}")
            print(f"get_csi_data - Final array: {",".join(map(str, csi_data))}")
        
        csv_index += 1
        return csi_data
    