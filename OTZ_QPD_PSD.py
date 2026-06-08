"""
OTZ_QPD_PSD.py
==============
Live dual-channel oscilloscope + PI stage control — dark-theme Tkinter GUI.

Layout
------
  ┌──────────────────────────────────────────────────────────────────┐
  │  [● ADS status]  [Connect] [Disconnect]  Device idx: [__]       │
  ├──────────────────────────────────────┬───────────────────────────┤
  │  Stage Control                       │  CH1 & CH2  V vs t        │
  │    [● Stage status] [Connect][Disc.] │                           │
  │    ○ Joystick  ○ Software            │                           │
  │    ○ Coarse    ○ Fine                │                           │
  │    Freq: [___]  Step: [___]          │                           │
  │    [▲]  [◄][►]  [▼]                 │                           │
  ├──────────────────────────────────────┼───────────────────────────┤
  │  XY — CH2 vs CH1                    │  PSD  CH1 & CH2           │
  └──────────────────────────────────────┴───────────────────────────┘
  [ Export PSD ]  [ Export V(t) ]   FPS: —

Requirements
------------
  pip install numpy matplotlib pyserial
  stage_control.py in the same directory
  waveforms_ads.py importable; Digilent WaveForms + Adept Runtime installed

Usage
-----
  python OTZ_QPD_PSD.py
  python OTZ_QPD_PSD.py --rate 200000 --samples 4096 --range 5.0
"""

import csv
import os
import sys
import time
import threading
import queue
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import stage_control

# ---------------------------------------------------------------------------
# WaveForms driver
# ---------------------------------------------------------------------------
try:
    from waveforms_ads import WaveFormsADS, DWFError
    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Dark-theme palette
# ---------------------------------------------------------------------------
BG          = "#1a1a2e"   # window / figure background
PANEL_BG    = "#16213e"   # axes / widget panel background
BORDER      = "#0f3460"   # frame borders, grid lines
FG          = "#e0e0e0"   # primary text
FG_DIM      = "#8888aa"   # secondary text / tick labels

CH1_COL     = "#4fc3f7"   # light blue
CH2_COL     = "#ef5350"   # soft red
XY_COL      = "#69f0ae"   # mint green
GRID_COL    = "#2a2a4a"
MINOR_COL   = "#222240"

STATUS_CONNECTED    = "#43d17a"   # green
STATUS_CONNECTING   = "#90caf9"   # light blue (replaces amber/orange)
STATUS_DISCONNECTED = "#ef5350"   # red

CONFIRM_COL = "#43d17a"   # green for save confirmations

# ---------------------------------------------------------------------------
# Matplotlib dark style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     PANEL_BG,
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    FG,
    "axes.grid":          True,
    "grid.color":         GRID_COL,
    "grid.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "text.color":         FG,
    "xtick.color":        FG_DIM,
    "ytick.color":        FG_DIM,
    "font.size":          8,
    "axes.titlesize":     9,
    "axes.labelsize":     8,
    "xtick.labelsize":    7,
    "ytick.labelsize":    7,
    "lines.antialiased":  True,
    "figure.autolayout":  False,
    "legend.facecolor":   PANEL_BG,
    "legend.edgecolor":   BORDER,
    "legend.labelcolor":  FG,
})

# ---------------------------------------------------------------------------
# Tk dark style (ttk)
# ---------------------------------------------------------------------------
_STYLE_INITED = False

def _init_ttk_style(root: tk.Tk) -> None:
    global _STYLE_INITED
    if _STYLE_INITED:
        return
    _STYLE_INITED = True

    root.configure(bg=BG)
    s = ttk.Style(root)
    s.theme_use("clam")

    common = {"background": BG, "foreground": FG,
              "fieldbackground": PANEL_BG, "bordercolor": BORDER,
              "darkcolor": PANEL_BG, "lightcolor": BORDER,
              "troughcolor": PANEL_BG, "selectbackground": BORDER,
              "selectforeground": FG}

    for widget in ("TFrame", "TLabel", "TLabelframe", "TLabelframe.Label",
                   "TCheckbutton", "TRadiobutton"):
        s.configure(widget, **{k: v for k, v in common.items()
                                if k in ("background", "foreground")})

    s.configure("TButton",
                background=BORDER, foreground=FG,
                bordercolor=BORDER, relief="flat", padding=4)
    s.map("TButton",
          background=[("active", "#1a4080"), ("disabled", PANEL_BG)],
          foreground=[("disabled", FG_DIM)])

    s.configure("TEntry",
                fieldbackground=PANEL_BG, foreground=FG,
                insertcolor=FG, bordercolor=BORDER)

    s.configure("TSpinbox",
                fieldbackground=PANEL_BG, foreground=FG,
                insertcolor=FG, bordercolor=BORDER,
                arrowcolor=FG)

    s.configure("TSeparator", background=BORDER)

    s.configure("TRadiobutton",
                background=BG, foreground=FG,
                indicatorcolor=BORDER, focuscolor=BG)
    s.map("TRadiobutton",
          indicatorcolor=[("selected", STATUS_CONNECTED)])

    s.configure("TLabelframe",
                background=BG, bordercolor=BORDER)
    s.configure("TLabelframe.Label",
                background=BG, foreground=FG_DIM)


# ===========================================================================
# PSD helper
# ===========================================================================

def compute_psd(signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import welch
        return welch(signal, fs=sample_rate, nperseg=min(len(signal), 512), scaling="density")
    except ImportError:
        n = len(signal)
        w = np.hanning(n)
        fft_v = np.fft.rfft(signal * w)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        psd = (np.abs(fft_v) ** 2) / (sample_rate * (w ** 2).sum())
        psd[1:-1] *= 2
        return freqs, psd


# ===========================================================================
# ADS acquisition thread
# ===========================================================================

class AcquisitionThread(threading.Thread):
    def __init__(self, device, sample_rate: float, n_samples: int,
                 out_queue: queue.Queue):
        super().__init__(daemon=True)
        self._dev  = device
        self._rate = sample_rate
        self._n    = n_samples
        self._q    = out_queue
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            from waveforms_ads import acqmodeSingle, trigsrcNone, DwfStateDone
        except ImportError:
            return

        dev = self._dev
        dev.analog_in_reset()
        dev.analog_in_set_sample_rate(self._rate)
        dev.analog_in_set_buffer_size(self._n)
        dev.analog_in_set_acquisition_mode(acqmodeSingle)
        dev.analog_in_channel_enable(0)
        dev.analog_in_channel_enable(1)
        dev.analog_in_set_trigger_source(trigsrcNone)

        while not self._stop.is_set():
            try:
                dev.analog_in_configure(reconfigure=True, start=True)
                deadline = time.time() + 5.0
                while not self._stop.is_set():
                    state = dev.analog_in_status(read_data=True)
                    if state == DwfStateDone or time.time() > deadline:
                        break
                    time.sleep(0.001)
                ch1 = dev.analog_in_get_data(0, self._n)
                ch2 = dev.analog_in_get_data(1, self._n)
                try:
                    self._q.put_nowait((ch1, ch2))
                except queue.Full:
                    pass
            except Exception as exc:
                print(f"[AcqThread] {exc}", file=sys.stderr)
                time.sleep(0.1)


# ===========================================================================
# Main Application
# ===========================================================================

class OscilloscopeApp:

    _POLL_MS      = 50    # ms between Tk .after() polls
    _MIN_DRAW_SEC = 0.08  # ~12 fps cap

    def __init__(self, root: tk.Tk, sample_rate=100_000, n_samples=4096,
                 input_range=5.0, device_index=-1):
        self._root      = root
        self._rate      = float(sample_rate)
        self._n         = int(n_samples)
        self._range     = float(input_range)
        self._dev_index = device_index

        # ADS state
        self._device: Optional[object] = None
        self._acq:    Optional[AcquisitionThread] = None
        self._q       = queue.Queue(maxsize=2)

        # Stage state
        self._stage_port:     Optional[object] = None
        self._stage_ctrl_type = "joystick"   # "joystick" | "software"
        self._stage_speed     = "coarse"     # "coarse"   | "fine"

        # Waveform buffers
        self._ch1     = np.zeros(self._n)
        self._ch2     = np.zeros(self._n)
        self._freqs1  = np.array([1.0, 2.0])
        self._psd1    = np.array([1e-9, 1e-9])
        self._freqs2  = np.array([1.0, 2.0])
        self._psd2    = np.array([1e-9, 1e-9])
        self._time_ms = np.linspace(0, self._n / self._rate * 1e3, self._n)

        # FPS tracking
        self._last_draw   = 0.0
        self._frame_count = 0
        self._psd_skip    = 0
        self._fps_t0      = time.time()
        self._fps         = 0.0

        root.title("OTZ QPD Scope")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        _init_ttk_style(root)
        self._build_ui()
        self._set_ads_status("disconnected")
        self._set_stage_status("disconnected")
        self._root.after(self._POLL_MS, self._poll)

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self):
        root = self._root

        # ── ADS control bar ───────────────────────────────────────────────
        ads_bar = ttk.Frame(root, padding=(6, 4))
        ads_bar.pack(side=tk.TOP, fill=tk.X)

        self._ads_status_canvas = tk.Canvas(
            ads_bar, width=14, height=14, highlightthickness=0, bg=BG)
        self._ads_status_canvas.pack(side=tk.LEFT, padx=(0, 4))
        self._ads_dot = self._ads_status_canvas.create_oval(
            2, 2, 12, 12, fill=STATUS_DISCONNECTED, outline="")

        self._ads_status_lbl = tk.Label(
            ads_bar, text="ADS: Disconnected",
            fg=STATUS_DISCONNECTED, bg=BG,
            font=("TkDefaultFont", 9, "bold"))
        self._ads_status_lbl.pack(side=tk.LEFT, padx=(0, 16))

        self._ads_device_lbl = tk.Label(
            ads_bar, text="Device: —", fg=FG_DIM, bg=BG)
        self._ads_device_lbl.pack(side=tk.LEFT, padx=(0, 20))

        tk.Label(ads_bar, text="Device index:", fg=FG, bg=BG).pack(side=tk.LEFT)
        self._ads_idx_var = tk.IntVar(value=self._dev_index)
        ttk.Spinbox(ads_bar, from_=-1, to=15, width=4,
                    textvariable=self._ads_idx_var).pack(side=tk.LEFT, padx=(2, 10))

        self._ads_btn_connect = ttk.Button(
            ads_bar, text="Connect", command=self._ads_on_connect)
        self._ads_btn_connect.pack(side=tk.LEFT, padx=2)

        self._ads_btn_disconnect = ttk.Button(
            ads_bar, text="Disconnect", command=self._ads_on_disconnect,
            state=tk.DISABLED)
        self._ads_btn_disconnect.pack(side=tk.LEFT, padx=2)

        self._fps_var = tk.StringVar(value="FPS: —")
        tk.Label(ads_bar, textvariable=self._fps_var,
                 fg=FG_DIM, bg=BG).pack(side=tk.RIGHT, padx=6)

        ttk.Separator(root, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # ── Main content area (left panel + right plots) ──────────────────
        content = ttk.Frame(root)
        content.pack(fill=tk.BOTH, expand=True)

        # LEFT: stage control panel (fixed width)
        self._build_stage_panel(content)

        # RIGHT: matplotlib canvas (expands to fill)
        self._build_plots(content)

        ttk.Separator(root, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # ── Bottom export bar ─────────────────────────────────────────────
        export_bar = ttk.Frame(root, padding=(6, 4))
        export_bar.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(export_bar, text="Export PSD (CH1+CH2)",
                   command=self._export_psd).pack(side=tk.LEFT, padx=4)
        ttk.Button(export_bar, text="Export V(t) (CH1+CH2)",
                   command=self._export_vt).pack(side=tk.LEFT, padx=4)

        self._export_msg_var = tk.StringVar(value="")
        tk.Label(export_bar, textvariable=self._export_msg_var,
                 fg=CONFIRM_COL, bg=BG).pack(side=tk.LEFT, padx=8)

    # ── Stage control panel ───────────────────────────────────────────────

    def _build_stage_panel(self, parent):
        panel = tk.Frame(parent, bg=BG, width=210)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        panel.pack_propagate(False)

        # Stage status row
        status_row = tk.Frame(panel, bg=BG)
        status_row.pack(fill=tk.X, pady=(0, 6))

        self._stage_status_canvas = tk.Canvas(
            status_row, width=14, height=14, highlightthickness=0, bg=BG)
        self._stage_status_canvas.pack(side=tk.LEFT, padx=(0, 4))
        self._stage_dot = self._stage_status_canvas.create_oval(
            2, 2, 12, 12, fill=STATUS_DISCONNECTED, outline="")

        self._stage_status_lbl = tk.Label(
            status_row, text="Stage: Disconnected",
            fg=STATUS_DISCONNECTED, bg=BG,
            font=("TkDefaultFont", 9, "bold"))
        self._stage_status_lbl.pack(side=tk.LEFT)

        # Stage connect / disconnect buttons
        btn_row = tk.Frame(panel, bg=BG)
        btn_row.pack(fill=tk.X, pady=(0, 8))

        self._stage_btn_connect = ttk.Button(
            btn_row, text="Connect", command=self._stage_on_connect)
        self._stage_btn_connect.pack(side=tk.LEFT, padx=(0, 4))

        self._stage_btn_disconnect = ttk.Button(
            btn_row, text="Disconnect", command=self._stage_on_disconnect,
            state=tk.DISABLED)
        self._stage_btn_disconnect.pack(side=tk.LEFT)

        # Control type
        ctrl_frame = ttk.LabelFrame(panel, text="Control Type")
        ctrl_frame.pack(fill=tk.X, pady=4)
        self._ctrl_var = tk.StringVar(value="Joystick Control")
        for opt in ("Joystick Control", "Software Control"):
            ttk.Radiobutton(ctrl_frame, text=opt, variable=self._ctrl_var,
                            value=opt,
                            command=self._stage_on_ctrl_change).pack(anchor="w")

        # Joystick speed
        spd_frame = ttk.LabelFrame(panel, text="Joystick Speed")
        spd_frame.pack(fill=tk.X, pady=4)
        self._spd_var = tk.StringVar(value="Coarse Control")
        for opt in ("Coarse Control", "Fine Control"):
            ttk.Radiobutton(spd_frame, text=opt, variable=self._spd_var,
                            value=opt,
                            command=self._stage_on_speed_change).pack(anchor="w")

        # Parameters
        param_frame = ttk.LabelFrame(panel, text="Parameters")
        param_frame.pack(fill=tk.X, pady=4)

        tk.Label(param_frame, text="Frequency:", fg=FG, bg=BG).grid(
            row=0, column=0, sticky="w", padx=4, pady=2)
        self._freq_var = tk.StringVar(value="250")
        ttk.Entry(param_frame, textvariable=self._freq_var,
                  width=10).grid(row=0, column=1, padx=4, pady=2)

        tk.Label(param_frame, text="Step size:", fg=FG, bg=BG).grid(
            row=1, column=0, sticky="w", padx=4, pady=2)
        self._step_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self._step_var,
                  width=10).grid(row=1, column=1, padx=4, pady=2)

        # Directional buttons
        move_frame = ttk.LabelFrame(panel, text="Move")
        move_frame.pack(fill=tk.X, pady=4)

        btn_kw = {"width": 7}
        ttk.Button(move_frame, text="▲ Up",
                   command=self._stage_move_up,    **btn_kw).grid(row=0, column=1, pady=2)
        ttk.Button(move_frame, text="◄ Left",
                   command=self._stage_move_left,  **btn_kw).grid(row=1, column=0, padx=2)
        ttk.Button(move_frame, text="► Right",
                   command=self._stage_move_right, **btn_kw).grid(row=1, column=2, padx=2)
        ttk.Button(move_frame, text="▼ Down",
                   command=self._stage_move_down,  **btn_kw).grid(row=2, column=1, pady=2)

        for col in (0, 1, 2):
            move_frame.columnconfigure(col, weight=1)

    # ── Matplotlib panels ─────────────────────────────────────────────────

    def _build_plots(self, parent):
        self._fig = Figure(figsize=(11, 7), dpi=100)

        # 2×2 grid: top-right = combined V(t), bottom-left = XY, bottom-right = PSD
        # top-left cell is occupied by the Tk stage panel, so we only add 3 axes
        gs = gridspec.GridSpec(
            2, 2, figure=self._fig,
            hspace=0.38, wspace=0.30,
            left=0.07, right=0.97,
            top=0.93, bottom=0.07,
        )
        self._ax_vt  = self._fig.add_subplot(gs[0, 1])   # top-right: CH1+CH2 V(t)
        self._ax_xy  = self._fig.add_subplot(gs[1, 0])   # bottom-left: XY
        self._ax_psd = self._fig.add_subplot(gs[1, 1])   # bottom-right: PSD

        self._setup_vt_ax()
        self._setup_xy_ax()
        self._setup_psd_ax()
        self._create_artists()

        self._fig.text(
            0.5, 0.975,
            f"{self._rate/1e3:.0f} kHz  ·  {self._n} samples",
            ha="center", va="top", fontsize=7, color=FG_DIM,
        )

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _setup_vt_ax(self):
        ax = self._ax_vt
        ax.set_title("CH1 & CH2 — Voltage vs Time", pad=3)
        ax.set_xlabel("Time (ms)", labelpad=1)
        ax.set_ylabel("Voltage (V)", labelpad=1)
        ax.set_xlim(self._time_ms[0], self._time_ms[-1])
        ax.set_ylim(-self._range / 2 * 1.15, self._range / 2 * 1.15)
        ax.minorticks_on()
        ax.grid(True, which="minor", linewidth=0.25, color=MINOR_COL)

    def _setup_xy_ax(self):
        ax = self._ax_xy
        ax.set_title("XY — CH2 vs CH1", pad=3)
        ax.set_xlabel("CH1 (V)", labelpad=1)
        ax.set_ylabel("CH2 (V)", labelpad=1)
        lim = self._range / 2 * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="datalim")

    def _setup_psd_ax(self):
        ax = self._ax_psd
        ax.set_title("PSD — CH1 & CH2", pad=3)
        ax.set_xlabel("Frequency (Hz)", labelpad=1)
        ax.set_ylabel("PSD (V²/Hz)", labelpad=1)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(1.0, self._rate / 2)
        ax.xaxis.set_minor_locator(
            plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.yaxis.set_minor_locator(
            plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.tick_params(axis="both", which="minor", labelbottom=False, labelleft=False)
        ax.grid(True, which="minor", linewidth=0.25, color=MINOR_COL)

    def _create_artists(self):
        tw = self._time_ms

        # Combined V(t) — both channels on one axis
        self._line_ch1, = self._ax_vt.plot(tw, self._ch1, color=CH1_COL,
                                            lw=0.9, label="CH1")
        self._line_ch2, = self._ax_vt.plot(tw, self._ch2, color=CH2_COL,
                                            lw=0.9, label="CH2")
        self._ax_vt.legend(fontsize=7, loc="upper right", framealpha=0.7)

        # Stats box (two-line text in top-right corner of V(t) panel)
        self._stat_vt = self._ax_vt.text(
            0.01, 0.98, "",
            fontsize=6.5, va="top", ha="left",
            transform=self._ax_vt.transAxes,
            color=FG,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PANEL_BG,
                      edgecolor=BORDER, alpha=0.9),
        )

        # XY — larger, solid scatter-style points for visibility
        self._line_xy, = self._ax_xy.plot(
            self._ch1, self._ch2,
            color=XY_COL, lw=0,
            marker="o", markersize=2.5, markeredgewidth=0,
            alpha=0.75,
        )

        # PSD
        self._line_psd1, = self._ax_psd.plot([], [], color=CH1_COL, lw=1.1, label="CH1")
        self._line_psd2, = self._ax_psd.plot([], [], color=CH2_COL, lw=1.1, label="CH2")
        self._ax_psd.legend(fontsize=7, loc="upper right", framealpha=0.7)

    # =========================================================================
    # Status helpers
    # =========================================================================

    def _set_ads_status(self, state: str, device_name: str = "—"):
        colours = {
            "connected":    (STATUS_CONNECTED,    "ADS: Connected"),
            "connecting":   (STATUS_CONNECTING,   "ADS: Connecting…"),
            "disconnected": (STATUS_DISCONNECTED, "ADS: Disconnected"),
        }
        col, text = colours.get(state, colours["disconnected"])
        self._ads_status_canvas.itemconfig(self._ads_dot, fill=col)
        self._ads_status_lbl.config(text=text, fg=col)
        self._ads_device_lbl.config(text=f"Device: {device_name}")

        connected = (state == "connected")
        self._ads_btn_connect.config(
            state=tk.DISABLED if connected else tk.NORMAL)
        self._ads_btn_disconnect.config(
            state=tk.NORMAL if connected else tk.DISABLED)

    def _set_stage_status(self, state: str):
        colours = {
            "connected":    (STATUS_CONNECTED,    "Stage: Connected"),
            "connecting":   (STATUS_CONNECTING,   "Stage: Connecting…"),
            "disconnected": (STATUS_DISCONNECTED, "Stage: Disconnected"),
        }
        col, text = colours.get(state, colours["disconnected"])
        self._stage_status_canvas.itemconfig(self._stage_dot, fill=col)
        self._stage_status_lbl.config(text=text, fg=col)

        connected = (state == "connected")
        self._stage_btn_connect.config(
            state=tk.DISABLED if connected else tk.NORMAL)
        self._stage_btn_disconnect.config(
            state=tk.NORMAL if connected else tk.DISABLED)

    # =========================================================================
    # ADS connect / disconnect
    # =========================================================================

    def _ads_on_connect(self):
        if not HW_AVAILABLE:
            messagebox.showerror(
                "Driver Missing",
                "waveforms_ads is not installed.\n"
                "Install it and ensure Digilent Adept Runtime is present.")
            return
        self._set_ads_status("connecting")
        idx = self._ads_idx_var.get()
        threading.Thread(target=self._ads_connect_worker,
                         args=(idx,), daemon=True).start()

    def _ads_connect_worker(self, index: int):
        try:
            dev = WaveFormsADS(device_index=index, auto_configure=0)
            dev.analog_in_set_range(0, self._range)
            dev.analog_in_set_range(1, self._range)
            self._device = dev
            name = str(dev)
        except Exception as exc:
            self._root.after(0, lambda: self._set_ads_status("disconnected"))
            self._root.after(0, lambda: messagebox.showerror(
                "ADS Connection Failed", str(exc)))
            return

        self._acq = AcquisitionThread(self._device, self._rate, self._n, self._q)
        self._acq.start()
        self._root.after(0, lambda: self._set_ads_status("connected", name))

    def _ads_on_disconnect(self):
        self._stop_acquisition()
        self._set_ads_status("disconnected")

    def _stop_acquisition(self):
        if self._acq is not None:
            self._acq.stop()
            self._acq.join(timeout=2.0)
            self._acq = None
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    # =========================================================================
    # Stage connect / disconnect
    # =========================================================================

    def _stage_on_connect(self):
        self._set_stage_status("connecting")
        threading.Thread(target=self._stage_connect_worker,
                         daemon=True).start()

    def _stage_connect_worker(self):
        try:
            port = stage_control.open_port()
            stage_control.init_stage(port)
            self._stage_port = port
        except Exception as exc:
            self._root.after(0, lambda: self._set_stage_status("disconnected"))
            self._root.after(0, lambda: messagebox.showerror(
                "Stage Connection Failed", str(exc)))
            return
        self._root.after(0, lambda: self._set_stage_status("connected"))

    def _stage_on_disconnect(self):
        if self._stage_port is not None:
            threading.Thread(
                target=stage_control.close_port,
                args=(self._stage_port,),
                daemon=True,
            ).start()
            self._stage_port = None
        self._set_stage_status("disconnected")

    def _stage_port_ok(self) -> bool:
        return (self._stage_port is not None and
                self._stage_port.is_open)

    # =========================================================================
    # Stage motion / mode callbacks
    # =========================================================================

    def _stage_get_params(self) -> Tuple[float, float]:
        return float(self._freq_var.get()), float(self._step_var.get())

    def _stage_move_left(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            freq, step = self._stage_get_params()
            stage_control.move(self._stage_port, axis=1,  distance= step, freq=freq)

    def _stage_move_right(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            freq, step = self._stage_get_params()
            stage_control.move(self._stage_port, axis=1,  distance=-step, freq=freq)

    def _stage_move_up(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            freq, step = self._stage_get_params()
            stage_control.move(self._stage_port, axis=0,  distance= step, freq=freq)

    def _stage_move_down(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            freq, step = self._stage_get_params()
            stage_control.move(self._stage_port, axis=0,  distance=-step, freq=freq)

    def _stage_on_ctrl_change(self):
        if not self._stage_port_ok():
            return
        if self._ctrl_var.get() == "Software Control":
            stage_control.joystick_off(self._stage_port)
            self._stage_ctrl_type = "software"
        else:
            stage_control.joystick_on(self._stage_port)
            self._stage_ctrl_type = "joystick"
            stage_control.set_resolution(
                self._stage_port, fine=(self._stage_speed == "fine"))

    def _stage_on_speed_change(self):
        if self._spd_var.get() == "Fine Control":
            self._stage_speed = "fine"
            if self._stage_port_ok() and self._stage_ctrl_type == "joystick":
                stage_control.set_resolution(self._stage_port, fine=True)
        else:
            self._stage_speed = "coarse"
            if self._stage_port_ok() and self._stage_ctrl_type == "joystick":
                stage_control.set_resolution(self._stage_port, fine=False)

    # =========================================================================
    # Poll / draw loop
    # =========================================================================

    def _poll(self):
        now = time.time()
        if now - self._last_draw >= self._MIN_DRAW_SEC:
            ch1 = ch2 = None
            try:
                while True:
                    ch1, ch2 = self._q.get_nowait()
            except queue.Empty:
                pass
            if ch1 is not None:
                self._redraw(ch1, ch2, now)
        self._root.after(self._POLL_MS, self._poll)

    def _redraw(self, ch1, ch2, now):
        self._ch1, self._ch2 = ch1, ch2
        self._last_draw = now
        self._frame_count += 1

        # V(t) — update both lines on the shared axis
        self._line_ch1.set_ydata(ch1)
        self._line_ch2.set_ydata(ch2)

        pk = max(abs(ch1.max()), abs(ch1.min()),
                 abs(ch2.max()), abs(ch2.min()), 1e-9)
        self._ax_vt.set_ylim(-pk * 1.15, pk * 1.15)

        rms1 = np.sqrt(np.mean(ch1 ** 2))
        rms2 = np.sqrt(np.mean(ch2 ** 2))
        self._stat_vt.set_text(
            f"CH1  pk {ch1.max():+.3f}  min {ch1.min():+.3f}  "
            f"rms {rms1:.3f}  μ {ch1.mean():+.4f}\n"
            f"CH2  pk {ch2.max():+.3f}  min {ch2.min():+.3f}  "
            f"rms {rms2:.3f}  μ {ch2.mean():+.4f}"
        )

        # XY
        self._line_xy.set_xdata(ch1)
        self._line_xy.set_ydata(ch2)
        lim = max(abs(ch1).max(), abs(ch2).max(), 1e-9) * 1.15
        self._ax_xy.set_xlim(-lim, lim)
        self._ax_xy.set_ylim(-lim, lim)

        # PSD — every other frame
        self._psd_skip += 1
        if self._psd_skip >= 2:
            self._psd_skip = 0
            self._freqs1, self._psd1 = compute_psd(ch1, self._rate)
            self._freqs2, self._psd2 = compute_psd(ch2, self._rate)
            f1, p1 = self._freqs1[1:], self._psd1[1:]
            f2, p2 = self._freqs2[1:], self._psd2[1:]
            self._line_psd1.set_data(f1, p1)
            self._line_psd2.set_data(f2, p2)
            valid = np.concatenate([p1[p1 > 0], p2[p2 > 0]])
            if len(valid):
                self._ax_psd.set_ylim(valid.min() * 0.1, valid.max() * 10)

        # FPS
        elapsed = now - self._fps_t0
        if elapsed >= 1.5:
            self._fps         = self._frame_count / elapsed
            self._fps_t0      = now
            self._frame_count = 0
            self._fps_var.set(f"FPS: {self._fps:.1f}")

        self._canvas.draw_idle()

    # =========================================================================
    # Export
    # =========================================================================

    def _flash_export(self, msg: str, duration: float = 3.0):
        self._export_msg_var.set(msg)
        self._root.after(int(duration * 1000),
                         lambda: self._export_msg_var.set(""))

    def _meta_rows(self, stamp):
        return [
            [f"# OTZ QPD Export  {stamp}"],
            [f"# Sample rate: {self._rate:.0f} Hz", f"N: {self._n}"],
        ]

    def _write_csv(self, fname, header_rows, col_names, rows):
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            for row in header_rows:
                w.writerow(row)
            w.writerow(col_names)
            for row in rows:
                w.writerow([f"{v:.6g}" for v in row])
        print(f"[Export] {os.path.abspath(fname)}")

    def _export_psd(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"psd_ch1_ch2_{stamp}.csv"
        f1, p1 = self._freqs1, self._psd1
        f2, p2 = self._freqs2, self._psd2
        if len(f1) != len(f2) or not np.allclose(f1, f2):
            p2 = np.interp(f1, f2, p2)
        try:
            self._write_csv(
                fname, self._meta_rows(stamp),
                ["frequency_hz", "psd_ch1_v2_per_hz", "psd_ch2_v2_per_hz"],
                zip(f1, p1, p2),
            )
            self._flash_export(f"✓ Saved {fname}")
        except OSError as exc:
            messagebox.showerror("Export Error", str(exc))

    def _export_vt(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"vt_ch1_ch2_{stamp}.csv"
        try:
            self._write_csv(
                fname, self._meta_rows(stamp),
                ["time_ms", "ch1_v", "ch2_v"],
                zip(self._time_ms, self._ch1, self._ch2),
            )
            self._flash_export(f"✓ Saved {fname}")
        except OSError as exc:
            messagebox.showerror("Export Error", str(exc))

    # =========================================================================
    # Close
    # =========================================================================

    def _on_close(self):
        self._stop_acquisition()
        self._stage_on_disconnect()
        self._root.destroy()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = OscilloscopeApp(root)
    root.mainloop()