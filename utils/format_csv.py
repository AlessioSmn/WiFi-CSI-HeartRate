import csv
import os

input_csv = "data/raw_data.csv"
output_dir = "tests"
os.makedirs(output_dir, exist_ok=True)

CSI_RAW_IDX = 1  # colonna csi_raw

with open(input_csv, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)  # salta header

    test_idx = 1
    prev_ts = None

    out_file = open(os.path.join(output_dir, f"test_{test_idx}.csv"), "w", newline="", encoding="utf-8")
    writer = csv.writer(out_file)
    writer.writerow(header)

    for row in reader:
        if not row:
            continue

        ts = int(row[0])

        if prev_ts is not None and ts < prev_ts:
            out_file.close()
            test_idx += 1
            out_file = open(os.path.join(output_dir, f"test_{test_idx}.csv"), "w", newline="", encoding="utf-8")
            writer = csv.writer(out_file)
            writer.writerow(header)

        # rimuove TUTTI gli spazi dentro csi_raw
        row[CSI_RAW_IDX] = row[CSI_RAW_IDX].replace(" ", "")

        writer.writerow(row)
        prev_ts = ts

    out_file.close()
