import csv
import argparse

from wifihr import monitor_hr

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port, extract heart rate and display it graphically")
    
    parser.add_argument('-p', '--port', dest='port', action='store', required=True,
                        help="Serial port number of csv_recv device")
    parser.add_argument('-s', '--store', dest='store_file', action='store', default='./csi_data.csv',
                        help="Save the data printed by the serial port to a file")
    parser.add_argument('-l', '--log', dest="log_file", action="store", default="./csi_data_log.txt",
                        help="Save other serial data the bad CSI data to a log file")

    args = parser.parse_args()
    serial_port = args.port
    file_name = args.store_file
    log_file_name = args.log_file

    monitor_hr(
        serial_port, 
        csv.writer(open(file_name, 'w')), 
        open(log_file_name, 'w'))
    
    print("Bye bye")
