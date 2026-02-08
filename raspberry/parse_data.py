import numpy as np
from io import StringIO
import numpy as np
import pandas as pd

def parse_csi_line(line, cols):
    """
    Parse a line sent via serial line
    
    :param line: received string in csv format
    :param cols: columns associated to each attribute
    """
    result = None
    try:
        result = pd.read_csv(
            StringIO(line),
            names=cols,
            quotechar='"'
        )
    except pd.errors.ParserError:
        result = None
    return result

def safe_parse_csi_data(data_str):
    """
    Parse the data field containing an array of csi data
    
    :param data_str: string containing the csi data array
    """
    if not isinstance(data_str, str) or not data_str.startswith("["):
        return None
    try:
        return np.fromstring(data_str[1:-1], sep=',', dtype=float)
    except Exception:
        return None


def from_buffer_to_df_detection(buffer_csi, cols_csi, csi_data_length=384):
    """
    Convert the buffer containing the received data to a Dataframe for processing
    
    :param buffer_csi: buffer with received data
    :param cols_csi: columns associated to the received data
    :param csi_data_length: length of the csi data array
    """
    parsed_rows = []

    #TODO probabilmente c'è un modo più efficiente
    for line_csi in buffer_csi:
        result_csi = parse_csi_line(line_csi, cols_csi)
        if result_csi is not None and not result_csi.empty:
            # Converti il risultato (DataFrame a riga singola) in dizionario e aggiungi alla lista
            parsed_rows.append(result_csi.iloc[0].to_dict())

    # create dataframe
    df_csi = pd.DataFrame(parsed_rows, columns=cols_csi)

    # only rows with type == "CSI_DATA" are valid
    df_csi = df_csi[df_csi["type"] == "CSI_DATA"].copy()

    # parse csi data array
    df_csi["csi_raw"] = df_csi["data"].map(safe_parse_csi_data)
    # drop invalid rows
    df_csi.dropna(subset=["csi_raw"], inplace=True)
    # compute csi data array length
    df_csi["csi_len"] = df_csi["csi_raw"].map(len)
    # remove arrays with wrong length
    df_csi = df_csi[df_csi["csi_len"] == csi_data_length].copy()

    return df_csi
