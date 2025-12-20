from enum import IntEnum
from datetime import datetime

class DebugLevel(IntEnum):
    DEBUG = 1
    INFO = 2
    ERROR = 3

LVL_DBG = DebugLevel.DEBUG
LVL_INF = DebugLevel.INFO
LVL_ERR = DebugLevel.ERROR

MIN_LEVEL = LVL_INF

def set_print_level(minimum_log_level: DebugLevel):
    global MIN_LEVEL
    MIN_LEVEL = minimum_log_level

def print_log(s: str, log_level: DebugLevel = DebugLevel.INFO):
    """
    Prints in format [timestamp] [Debug level] [string]
    
    :param s: String to print
    :type s: str
    :param log_level: Description
    :type log_level: DebugLevel
    """
    if log_level < MIN_LEVEL:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{log_level.name:<5}] {s}")
