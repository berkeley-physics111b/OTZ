"""
stage_control.py
===============
Serial hardware interface for picomotor stage controller.
Handles port open/close and all command send/receive logic.
"""

import time
import serial


# ---------------------------------------------------------------------------
# Low-level send helpers
# ---------------------------------------------------------------------------

def _send_no_log(command: str, port: serial.Serial) -> None:
    port.write((command + '\r\n').encode())


def _send_log(command: str, port: serial.Serial) -> None:
    msg = (command + '\r\n').encode()
    print(f'[Stage] sending {msg}')
    port.write(msg)
    port.flush()
    time.sleep(0.1)
    if port.in_waiting:
        response = port.read(port.in_waiting)
        print(f'[Stage] response: {response}')
    else:
        print('[Stage] no response')


def send_command(command: str, port: serial.Serial, log: bool = True) -> None:
    """Write a command to the stage.  
    Set log=False to suppress I/O printing. Currently needs to be True for it to work."""
    if log:
        _send_log(command, port)
    else:
        _send_no_log(command, port)


# ---------------------------------------------------------------------------
# Port management
# ---------------------------------------------------------------------------

DEFAULT_PORT     = 'COM3'
DEFAULT_BAUDRATE = 19200


def open_port(port_name: str = DEFAULT_PORT,
              baudrate: int  = DEFAULT_BAUDRATE) -> serial.Serial:
    """Open and return a configured Serial port, or raise serial.SerialException."""
    port = serial.Serial(
        port=port_name,
        baudrate=baudrate,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=2,
    )
    return port


def init_stage(port: serial.Serial) -> None:
    """Send startup commands: enable joystick, set coarse resolution."""
    send_command('JON',       port)
    send_command('RES COARSE', port)


def close_port(port: serial.Serial) -> None:
    """Restore safe defaults and close the port."""
    if port and port.is_open:
        send_command('RES COARSE', port)
        send_command('JON',        port)
        port.close()


# ---------------------------------------------------------------------------
# Motion commands
# ---------------------------------------------------------------------------

def move(port: serial.Serial,
         axis: int, distance: float, freq: float) -> None:
    """Move one axis by *distance* steps at *freq* Hz.  Negative = reverse."""
    send_command(f'VEL a1 {axis}={freq}', port)
    send_command(f'REL a1={distance} g',  port)


def joystick_on(port: serial.Serial) -> None:
    send_command('JON', port)


def joystick_off(port: serial.Serial) -> None:
    send_command('JOF', port)


def set_resolution(port: serial.Serial, fine: bool = False) -> None:
    send_command('RES FINE' if fine else 'RES COARSE', port)