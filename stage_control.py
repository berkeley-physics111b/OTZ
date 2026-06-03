import tkinter as tk
from tkinter import ttk
import serial
import time

def send_com_no_log(command, port):
    port.write((command + '\r\n').encode())

def send_com_log(command, port):
    msg = (command + '\r\n').encode()
    print(f'sending {msg}')
    port.write(msg)
    port.flush()
    time.sleep(0.1)
    if port.in_waiting:
        response = port.read(port.in_waiting)
        print(f'response: {response}')
    else:
        print('no response')

def send_com(command, port, log=True):
    if not log:
        #for some reason doesn't work in this case. still testing
        send_com_no_log(command, port)
    else:
        send_com_log(command, port)

class StageControl(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Stage Control")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.control_type = 'joystick'
        self.joystick_speed = 'coarse'
        self.port = None

        self._build_ui()
        self._connect_port()

    def _connect_port(self):
        try:
            self.port = serial.Serial(
                port='COM3',
                baudrate=19200,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=2
            )
            send_com('JON', self.port)
            send_com('RES COARSE', self.port)
            self._set_status('connected')
        except serial.SerialException as e:
            print(e)
            self._set_status('disconnected')

    def _set_status(self, state):
        # state: 'connected' | 'disconnected'
        if state == 'connected':
            self.status_dot.config(bg='#639922')
            self.status_label.config(text='Connected',
                                     fg='#3B6D11')
            self.reconnect_btn.grid_remove()
        else:
            self.status_dot.config(bg='#E24B4A')
            self.status_label.config(text='Disconnected',
                                     fg='#A32D2D')
            self.reconnect_btn.grid()

    def _build_ui(self):
        pad = {'padx': 8, 'pady': 4}

        # ── Status bar ───
        status_frame = ttk.Frame(self)
        status_frame.grid(row=0, column=0, columnspan=2,
                          sticky='ew', **pad)

        self.status_dot = tk.Label(status_frame, text=' ',
            width=2, relief='flat', bg='#E24B4A')
        self.status_dot.grid(row=0, column=0, padx=(0,6))

        self.status_label = tk.Label(status_frame,
            text='Disconnected', fg='#A32D2D', font=('TkDefaultFont', 10, 'bold'))
        self.status_label.grid(row=0, column=1)

        self.reconnect_btn = ttk.Button(status_frame, text='Reconnect',
            command=self._connect_port)
        self.reconnect_btn.grid(row=0, column=2, padx=(12,0))
        self.reconnect_btn.grid_remove()  # hidden until needed

        ctrl_frame = ttk.LabelFrame(self, text="Control Type")
        ctrl_frame.grid(row=1, column=0, **pad)
        self.ctrl_var = tk.StringVar(value='Joystick Control')
        for opt in ['Joystick Control', 'Software Control']:
            ttk.Radiobutton(ctrl_frame, text=opt, variable=self.ctrl_var,
                value=opt, command=self.on_ctrl_change).pack(anchor='w')

        spd_frame = ttk.LabelFrame(self, text="Joystick Speed")
        spd_frame.grid(row=1, column=1, **pad)
        self.spd_var = tk.StringVar(value='Coarse Control')
        for opt in ['Coarse Control', 'Fine Control']:
            ttk.Radiobutton(spd_frame, text=opt, variable=self.spd_var,
                value=opt, command=self.on_speed_change).pack(anchor='w')

        inp_frame = ttk.LabelFrame(self, text="Parameters")
        inp_frame.grid(row=2, column=0, columnspan=2, **pad)
        ttk.Label(inp_frame, text="Frequency:").grid(row=0, column=0)
        self.freq_var = tk.StringVar(value="250")
        ttk.Entry(inp_frame, textvariable=self.freq_var, width=10).grid(row=0, column=1)
        ttk.Label(inp_frame, text="Step size:").grid(row=1, column=0)
        self.step_var = tk.StringVar(value="100")
        ttk.Entry(inp_frame, textvariable=self.step_var, width=10).grid(row=1, column=1)

        btn_frame = ttk.LabelFrame(self, text="Move")
        btn_frame.grid(row=3, column=0, columnspan=2, **pad)
        ttk.Button(btn_frame, text="▲ Up",    command=self.move_up   ).grid(row=0, column=1)
        ttk.Button(btn_frame, text="◄ Left",  command=self.move_left ).grid(row=1, column=0)
        ttk.Button(btn_frame, text="► Right", command=self.move_right).grid(row=1, column=2)
        ttk.Button(btn_frame, text="▼ Down",  command=self.move_down ).grid(row=2, column=1)

    def _port_ok(self):
        # Guard used before every serial write
        return self.port is not None and self.port.is_open

    def get_params(self):
        return float(self.freq_var.get()), float(self.step_var.get())

    def move_left(self):
        if self.control_type == 'software' and self._port_ok():
            freq, step = self.get_params()
            send_com(f'VEL a1 1={freq}', self.port)
            send_com(f'REL a1={step} g', self.port)

    def move_right(self):
        if self.control_type == 'software' and self._port_ok():
            freq, step = self.get_params()
            send_com(f'VEL a1 1={freq}', self.port)
            send_com(f'REL a1={-step} g', self.port)

    def move_up(self):
        if self.control_type == 'software' and self._port_ok():
            freq, step = self.get_params()
            send_com(f'VEL a1 0={freq}', self.port)
            send_com(f'REL a1={step} g', self.port)

    def move_down(self):
        if self.control_type == 'software' and self._port_ok():
            freq, step = self.get_params()
            send_com(f'VEL a1 0={freq}', self.port)
            send_com(f'REL a1={-step} g', self.port)

    def on_ctrl_change(self):
        if not self._port_ok():
            print('ctrl change problem :(')
            return
        if self.ctrl_var.get() == 'Software Control':
            send_com('JOF', self.port)
            self.control_type = 'software'
        else:
            send_com('JON', self.port)
            self.control_type = 'joystick'
            res = 'RES FINE' if self.joystick_speed == 'fine' else 'RES COARSE'
            send_com(res, self.port)

    def on_speed_change(self):
        if self.spd_var.get() == 'Fine Control':
            self.joystick_speed = 'fine'
            if self._port_ok() and self.control_type == 'joystick':
                send_com('RES FINE', self.port)
        else:
            self.joystick_speed = 'coarse'
            if self._port_ok() and self.control_type == 'joystick':
                send_com('RES COARSE', self.port)

    def on_close(self):
        if self._port_ok():
            send_com('RES COARSE', self.port)
            send_com('JON', self.port)
            self.port.close()
        self.destroy()

if __name__ == '__main__':
    StageControl().mainloop()