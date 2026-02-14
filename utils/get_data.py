import os
import csv
import serial
from datetime import datetime
import time


# ===== CONFIG =====
# Linux:        /dev/ttyUSBx
# Windows:      COMx
# Raspberry:    /dev/serial0
SERIAL_PORT = "COM7"
BAUDRATE = 921600
CSV_FILE = "A_x.csv"

# ===== SERIAL =====
ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

file_exists = os.path.isfile(CSV_FILE)
file_exists = True
start_ok = False

LOOP_ON = True

with open(CSV_FILE, "w", encoding="utf-8", buffering=1024*1024, newline="") as f:
    counter = 0

    read_line = ser.readline
    write_file = f.write

    print("Listening... CTRL+C to stop")

    try:
        while LOOP_ON:# Leggi i byte direttamente
            line_bytes = read_line()
            
            if not line_bytes:
                continue
                
            # Decodifica solo se strettamente necessario per il controllo "startswith"
            # Se sai che "CSI_DATA" è all'inizio, puoi controllare i primi byte
            if not line_bytes.startswith(b"CSI_DATA"):
                continue

            if not start_ok:
                start_ok = True
                start_time = time.perf_counter() # Più preciso per misurare intervalli

            # Scriviamo la stringa decodificata
            line_str = line_bytes.decode("utf-8", errors="ignore")
            write_file(line_str)

            counter += 1
            
            # Esegui il check di diagnostica raramente
            if counter & 255 == 0:  # Bitwise check (circa ogni 512 righe) è più veloce del modulo %
                curr_time = time.perf_counter()
                elapsed = curr_time - start_time
                print(f"{counter} rows - Est. HZ: {counter / elapsed:.2f}")
                
                if elapsed > 150:
                    break

            """
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            #line_raw = ser.readline()

            if not line or not line.startswith("CSI_DATA"):
                continue
            
            if not start_ok:
                start_ok = True
                start_time = datetime.now()

            f.write(line + '\n')

            counter += 1
            if counter % 250 == 0:
                end_time = datetime.now()
                time_difference_s = (end_time - start_time).total_seconds()
                time_difference_ms = time_difference_s * 10**3
                print(f"{counter} row saved (in {time_difference_s:.2f} s) - estimated HZ: {counter} / {time_difference_s} = {counter / time_difference_s:.2f} HZ")

                if time_difference_s > 150:
                    LOOP_ON = False
            """

    except KeyboardInterrupt:
        print("\nClosing...")
    finally:
        ser.close()
        f.flush()

