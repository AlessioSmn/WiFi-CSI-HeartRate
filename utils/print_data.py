import os
import csv
import serial
from datetime import datetime
import time


# ===== CONFIG =====
# Linux:        /dev/ttyUSBx
# Windows:      COMx
# Raspberry:    /dev/serial0
BAUDRATE = 921600


SERIAL_PORT = input("Insert serial port name:")

# ===== SERIAL =====
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

LOOP_ON = True

while(LOOP_ON):
    try:
        line = ser.readline()          # legge fino a \n
        if line:
            print(line.decode().strip())

    except KeyboardInterrupt:
        LOOP_ON = False
    
