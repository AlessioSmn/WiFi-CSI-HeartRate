import argparse

from wifihr import monitor_hr

if __name__ == '__main__':
    examples = """
    Examples:
     - from serial (COM3), debug level default (INFO)
        python main.py -p COM3
    
     - from serial (COM3), debug level DEBUG
        python main.py -p COM3 -d debug
    
     - from csv (file.csv), debug level DEBUG
        python main.py -p file.csv --csv -d debug
    
     - from csv (file.csv), debug level DEBUG, with log and with ui
        python main.py -p file.csv --csv -d debug
    """
    
    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port, extract heart rate and display it graphically",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-p', '--port',
        required=True,
        help="Serial port of CSI device (or .csv filename, if --csv specified)"
    )

    parser.add_argument(
        '--csv', 
        dest='from_serial', 
        action='store_false',
        help="If specifies, it inticates to use the parameter -p as a .csv filename"
    )
    parser.set_defaults(from_serial=True)

    parser.add_argument(
        '-d', '--debug',
        dest='print_level',
        choices=['debug', 'info', 'error'],
        default='info',
        help="Print level"
    )

    parser.add_argument(
        '-l', '--log',
        dest='log_filename',
        help="Log file name"
    )

    parser.add_argument(
        '-u', '--ui',
        dest='start_ui',
        action='store_true',
        help="Starts UI process if specified"
    )


    args = parser.parse_args()

    monitor_hr(
        port=args.port,
        from_serial=args.from_serial,
        print_level=args.print_level,
        start_ui_proc=args.start_ui,
        log_filename=args.log_filename
    )
