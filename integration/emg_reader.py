"""
Basic sanity-check tool for the EMG sensor wired into arm/src/main.cpp
(GPIO5, ADC1_CH4). Toggles the firmware's "emg" streaming mode on
connect and prints each raw ADC reading (0-4095, 0-3.3V) as a live
text meter so you can confirm the sensor and electrode placement work
before writing any gesture classification logic.

Usage:
    python integration/emg_reader.py --list-ports
    python integration/emg_reader.py --port COM4

Ctrl+C to stop.
"""

import argparse
import time

import serial
from serial.tools import list_ports

BAUD_RATE = 115200
ADC_MAX = 4095
METER_WIDTH = 40


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=str, default=None, help="serial port for the arm, e.g. COM4")
    parser.add_argument("--list-ports", action="store_true", help="list available serial ports and exit")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_ports:
        for p in list_ports.comports():
            print(f"{p.device}  {p.description}")
        return

    if not args.port:
        raise SystemExit("Specify --port (use --list-ports to see available ports).")

    ser = serial.Serial(args.port, BAUD_RATE, timeout=1)
    time.sleep(2)  # ESP32 resets when the serial port opens; wait for firmware to boot
    ser.reset_input_buffer()
    ser.write(b"emg\n")  # firmware boots with streaming off; this turns it on

    print("Reading EMG values, Ctrl+C to stop.")
    try:
        while True:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line.startswith("emg "):
                continue
            value = int(line.split()[1])
            bar_len = int(value / ADC_MAX * METER_WIDTH)
            bar = "#" * bar_len + "-" * (METER_WIDTH - bar_len)
            print(f"\r{value:5d}  [{bar}]", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        ser.write(b"emg\n")  # toggle streaming back off
        time.sleep(0.1)
        ser.close()


if __name__ == "__main__":
    main()
