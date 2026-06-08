"""
OTZ_QPD_PSD.py
==============
Live dual-channel oscilloscope + PI stage control — dark-theme Tkinter GUI.

Layout
------
  ┌──────────────────────────────────────────────────────────────────┐
  │  [● ADS status]  [Connect] [Disconnect]  Device idx: [__]  FPS  │
  ├──────────────────────┬───────────────────────────────────────────┤
  │  Stage Control       │  Oscilloscope Settings                    │
  │    [● status]        │    Sample freq (Hz): [______]             │
  │    [Connect][Disc.]  │    History length (s): [______]  [Apply]  │
  │    ○ Joystick/Soft   │    Active: … kHz · … s · … samples       │
  │    ○ Coarse / Fine   ├───────────────────────────────────────────┤
  │    Freq / Step       │  CH1 & CH2 — Voltage vs Time             │
  │    [▲][◄][►][▼]     ├───────────────────────────────────────────┤
  ├──────────────────────┤  PSD  CH1 & CH2                          │
  │  XY — CH2 vs CH1    │                                           │
  └──────────────────────┴───────────────────────────────────────────┘
  [ Export PSD ]  [ Export V(t) ]

Requirements
------------
  pip install numpy matplotlib pyserial
  stage_control.py in the same directory
  waveforms_ads.py importable; Digilent WaveForms + Adept Runtime installed
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
from tkinter import ttk, messagebox, filedialog
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
BG       = "#1a1a2e"
PANEL_BG = "#16213e"
BORDER   = "#0f3460"
FG       = "#e0e0e0"
FG_DIM   = "#8888aa"

CH1_COL    = "#4fc3f7"
CH2_COL    = "#ef5350"
XY_COL     = "#69f0ae"
GRID_COL   = "#2a2a4a"
MINOR_COL  = "#222240"

STATUS_CONNECTED    = "#43d17a"
STATUS_CONNECTING   = "#90caf9"
STATUS_DISCONNECTED = "#ef5350"
CONFIRM_COL         = "#43d17a"

# ---------------------------------------------------------------------------
# Matplotlib dark style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   FG,
    "axes.grid":         True,
    "grid.color":        GRID_COL,
    "grid.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "text.color":        FG,
    "xtick.color":       FG_DIM,
    "ytick.color":       FG_DIM,
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "lines.antialiased": True,
    "figure.autolayout": False,
    "legend.facecolor":  PANEL_BG,
    "legend.edgecolor":  BORDER,
    "legend.labelcolor": FG,
})

# ---------------------------------------------------------------------------
# Tk dark style
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
    for w in ("TFrame", "TLabel", "TLabelframe", "TLabelframe.Label",
              "TCheckbutton", "TRadiobutton"):
        s.configure(w, background=BG, foreground=FG)
    s.configure("TButton", background=BORDER, foreground=FG,
                bordercolor=BORDER, relief="flat", padding=4)
    s.map("TButton",
          background=[("active", "#1a4080"), ("disabled", PANEL_BG)],
          foreground=[("disabled", FG_DIM)])
    s.configure("TEntry", fieldbackground=PANEL_BG, foreground=FG,
                insertcolor=FG, bordercolor=BORDER)
    s.configure("TSpinbox", fieldbackground=PANEL_BG, foreground=FG,
                insertcolor=FG, bordercolor=BORDER, arrowcolor=FG)
    s.configure("TSeparator", background=BORDER)
    s.configure("TRadiobutton", background=BG, foreground=FG,
                indicatorcolor=BORDER, focuscolor=BG)
    s.map("TRadiobutton", indicatorcolor=[("selected", STATUS_CONNECTED)])
    s.configure("TLabelframe", background=BG, bordercolor=BORDER)
    s.configure("TLabelframe.Label", background=BG, foreground=FG_DIM)


# ===========================================================================
# PSD helper  (called from a background thread)
# ===========================================================================

def compute_psd(signal: np.ndarray,
                sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import welch
        nperseg = min(len(signal), 4096)
        return welch(signal, fs=sample_rate, nperseg=nperseg, scaling="density")
    except ImportError:
        n = len(signal)
        w = np.hanning(n)
        fft_v = np.fft.rfft(signal * w)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        psd   = (np.abs(fft_v) ** 2) / (sample_rate * (w ** 2).sum())
        psd[1:-1] *= 2
        return freqs, psd


# ===========================================================================
# Acquisition thread
# ===========================================================================

class AcquisitionThread(threading.Thread):
    """
    Acquires small chunks (~_CHUNK_MS ms) continuously and pushes them into
    a queue.  The main thread drains the queue into its own ring buffer.
    maxsize=0 means unbounded — the ring buffer on the consumer side handles
    the fixed memory footprint; we never want to drop hardware data silently.
    """
    _CHUNK_MS = 40   # target chunk duration in ms

    def __init__(self, device, sample_rate: float, out_queue: queue.Queue):
        super().__init__(daemon=True)
        self._dev  = device
        self._rate = sample_rate
        self._n    = max(64, int(round(sample_rate * self._CHUNK_MS / 1000)))
        self._q    = out_queue
        self._stop = threading.Event()

    @property
    def chunk_size(self) -> int:
        return self._n

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
                self._q.put((ch1, ch2))   # blocks if consumer is slow — no silent drops
            except Exception as exc:
                print(f"[AcqThread] {exc}", file=sys.stderr)
                time.sleep(0.1)


# ===========================================================================
# Ring buffer
# ===========================================================================

class RingBuffer:
    """
    Pre-allocated circular buffer for two channels.
    Write pointer advances on each push; a contiguous view is materialised
    only when view() is called (one np.empty + two np.copyto, O(N)).
    """
    def __init__(self, capacity: int):
        self._cap  = capacity
        self._buf1 = np.zeros(capacity * 2)   # double-length for zero-copy wrap
        self._buf2 = np.zeros(capacity * 2)
        self._ptr  = 0   # points to the oldest sample slot

    def resize(self, new_capacity: int) -> None:
        self._cap  = new_capacity
        self._buf1 = np.zeros(new_capacity * 2)
        self._buf2 = np.zeros(new_capacity * 2)
        self._ptr  = 0

    def push(self, ch1: np.ndarray, ch2: np.ndarray) -> None:
        n = len(ch1)
        # Write at ptr and ptr+cap (mirror) so any window is contiguous
        for start in (self._ptr, self._ptr + self._cap):
            end = start + n
            if end <= len(self._buf1):
                self._buf1[start:end] = ch1
                self._buf2[start:end] = ch2
            else:
                # Wrap around within the double buffer
                split = len(self._buf1) - start
                self._buf1[start:] = ch1[:split]
                self._buf1[:end - len(self._buf1)] = ch1[split:]
                self._buf2[start:] = ch2[:split]
                self._buf2[:end - len(self._buf2)] = ch2[split:]
        self._ptr = (self._ptr + n) % self._cap

    def view(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return contiguous arrays of the last `capacity` samples."""
        start = self._ptr
        return (self._buf1[start:start + self._cap].copy(),
                self._buf2[start:start + self._cap].copy())


# ===========================================================================
# Main Application
# ===========================================================================

_DEFAULT_RATE_HZ  = 8_000
_DEFAULT_HISTORY_S = 8.0
_DEFAULT_RANGE_V   = 5.0
_DEFAULT_DEV_IDX   = -1

class OscilloscopeApp:

    _POLL_MS      = 50     # how often to drain the queue and check for a redraw
    _DRAW_HZ      = 10     # target display update rate
    _PSD_EVERY    = 3      # redraw frames between PSD recomputes

    def __init__(self, root: tk.Tk):
        self._root = root

        self._rate      = float(_DEFAULT_RATE_HZ)
        self._history_s = float(_DEFAULT_HISTORY_S)
        self._range     = float(_DEFAULT_RANGE_V)

        # ADS
        self._device: Optional[object] = None
        self._acq:    Optional[AcquisitionThread] = None
        self._q       = queue.Queue()   # unbounded — ring buffer is the limit

        # Stage
        self._stage_port:      Optional[object] = None
        self._stage_ctrl_type  = "joystick"
        self._stage_speed      = "coarse"

        # Ring buffer
        self._ring     = RingBuffer(self._calc_history_n())
        self._time_s   = np.linspace(0.0, self._history_s,
                                     self._calc_history_n())

        # PSD — computed on a background thread to avoid blocking Tk
        self._freqs1   = np.array([1.0, 2.0])
        self._psd1     = np.array([1e-9, 1e-9])
        self._freqs2   = np.array([1.0, 2.0])
        self._psd2     = np.array([1e-9, 1e-9])
        self._psd_lock = threading.Lock()
        self._psd_busy = False   # True while background PSD thread is running

        # Draw bookkeeping
        self._draw_interval = 1.0 / self._DRAW_HZ
        self._last_draw     = 0.0
        self._frame_count   = 0
        self._psd_countdown = 0
        self._fps_t0        = time.time()

        root.title("OTZ QPD Scope")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        _init_ttk_style(root)
        self._build_ui()
        self._set_ads_status("disconnected")
        self._set_stage_status("disconnected")
        self._root.after(self._POLL_MS, self._poll)

    # -------------------------------------------------------------------------
    # Sizing helpers
    # -------------------------------------------------------------------------

    def _calc_history_n(self) -> int:
        return max(64, int(round(self._rate * self._history_s)))

    # =========================================================================
    # UI
    # =========================================================================

    def _build_ui(self):
        root = self._root

        # ── ADS top bar ───────────────────────────────────────────────────
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
        self._ads_status_lbl.pack(side=tk.LEFT, padx=(0, 12))

        self._ads_device_lbl = tk.Label(ads_bar, text="Device: —",
                                        fg=FG_DIM, bg=BG)
        self._ads_device_lbl.pack(side=tk.LEFT, padx=(0, 18))

        tk.Label(ads_bar, text="Device index:", fg=FG, bg=BG).pack(side=tk.LEFT)
        self._ads_idx_var = tk.IntVar(value=_DEFAULT_DEV_IDX)
        ttk.Spinbox(ads_bar, from_=-1, to=15, width=4,
                    textvariable=self._ads_idx_var).pack(side=tk.LEFT, padx=(2, 8))

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

        # ── Main content ──────────────────────────────────────────────────
        content = ttk.Frame(root)
        content.pack(fill=tk.BOTH, expand=True)

        self._build_left_panel(content)
        self._build_right_column(content)

        ttk.Separator(root, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # ── Export bar ────────────────────────────────────────────────────
        export_bar = ttk.Frame(root, padding=(6, 4))
        export_bar.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(export_bar, text="Export PSD (CH1+CH2)",
                   command=self._export_psd).pack(side=tk.LEFT, padx=4)
        ttk.Button(export_bar, text="Export V(t) (CH1+CH2)",
                   command=self._export_vt).pack(side=tk.LEFT, padx=4)

        self._export_msg_var = tk.StringVar(value="")
        tk.Label(export_bar, textvariable=self._export_msg_var,
                 fg=CONFIRM_COL, bg=BG).pack(side=tk.LEFT, padx=8)

    # ── Left column: stage controls + XY ─────────────────────────────────

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=BG, width=230)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        left.pack_propagate(False)
        self._build_stage_panel(left)

    def _build_stage_panel(self, parent):
        status_row = tk.Frame(parent, bg=BG)
        status_row.pack(fill=tk.X, pady=(0, 4))

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

        btn_row = tk.Frame(parent, bg=BG)
        btn_row.pack(fill=tk.X, pady=(0, 6))
        self._stage_btn_connect = ttk.Button(
            btn_row, text="Connect", command=self._stage_on_connect)
        self._stage_btn_connect.pack(side=tk.LEFT, padx=(0, 4))
        self._stage_btn_disconnect = ttk.Button(
            btn_row, text="Disconnect", command=self._stage_on_disconnect,
            state=tk.DISABLED)
        self._stage_btn_disconnect.pack(side=tk.LEFT)

        ctrl_frame = ttk.LabelFrame(parent, text="Control Type")
        ctrl_frame.pack(fill=tk.X, pady=2)
        self._ctrl_var = tk.StringVar(value="Joystick Control")
        for opt in ("Joystick Control", "Software Control"):
            ttk.Radiobutton(ctrl_frame, text=opt, variable=self._ctrl_var,
                            value=opt,
                            command=self._stage_on_ctrl_change).pack(anchor="w")

        spd_frame = ttk.LabelFrame(parent, text="Joystick Speed")
        spd_frame.pack(fill=tk.X, pady=2)
        self._spd_var = tk.StringVar(value="Coarse Control")
        for opt in ("Coarse Control", "Fine Control"):
            ttk.Radiobutton(spd_frame, text=opt, variable=self._spd_var,
                            value=opt,
                            command=self._stage_on_speed_change).pack(anchor="w")

        param_frame = ttk.LabelFrame(parent, text="Parameters")
        param_frame.pack(fill=tk.X, pady=2)
        tk.Label(param_frame, text="Frequency:", fg=FG, bg=BG).grid(
            row=0, column=0, sticky="w", padx=4, pady=1)
        self._freq_var = tk.StringVar(value="250")
        ttk.Entry(param_frame, textvariable=self._freq_var,
                  width=9).grid(row=0, column=1, padx=4, pady=1)
        tk.Label(param_frame, text="Step size:", fg=FG, bg=BG).grid(
            row=1, column=0, sticky="w", padx=4, pady=1)
        self._step_var = tk.StringVar(value="100")
        ttk.Entry(param_frame, textvariable=self._step_var,
                  width=9).grid(row=1, column=1, padx=4, pady=1)

        move_frame = ttk.LabelFrame(parent, text="Move")
        move_frame.pack(fill=tk.X, pady=2)
        bk = {"width": 7}
        ttk.Button(move_frame, text="▲ Up",
                   command=self._stage_move_up,    **bk).grid(row=0, column=1, pady=2)
        ttk.Button(move_frame, text="◄ Left",
                   command=self._stage_move_left,  **bk).grid(row=1, column=0, padx=2)
        ttk.Button(move_frame, text="► Right",
                   command=self._stage_move_right, **bk).grid(row=1, column=2, padx=2)
        ttk.Button(move_frame, text="▼ Down",
                   command=self._stage_move_down,  **bk).grid(row=2, column=1, pady=2)
        for c in (0, 1, 2):
            move_frame.columnconfigure(c, weight=1)

    # ── Right column: settings + single figure (XY | V(t) | PSD) ─────────

    def _build_right_column(self, parent):
        right = ttk.Frame(parent)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._build_settings_panel(right)
        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(4, 0))
        self._build_plots(right)

    def _build_settings_panel(self, parent):
        sf = ttk.LabelFrame(parent, text="Oscilloscope Settings")
        sf.pack(fill=tk.X)

        inner = ttk.Frame(sf, padding=(6, 4))
        inner.pack(fill=tk.X)

        ttk.Label(inner, text="Sample frequency (Hz):").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self._setting_rate_var = tk.StringVar(value=str(int(_DEFAULT_RATE_HZ)))
        ttk.Entry(inner, textvariable=self._setting_rate_var,
                  width=10).grid(row=0, column=1, sticky="w", pady=3)

        ttk.Label(inner, text="History length (s):").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self._setting_history_var = tk.StringVar(value=str(_DEFAULT_HISTORY_S))
        ttk.Entry(inner, textvariable=self._setting_history_var,
                  width=10).grid(row=1, column=1, sticky="w", pady=3)

        # Button cluster: Apply | Start | Stop | Clear Graphs
        btn_frame = ttk.Frame(inner)
        btn_frame.grid(row=0, column=2, rowspan=2, padx=(14, 0), sticky="ns")

        ttk.Button(btn_frame, text="Apply",
                   command=self._settings_apply).pack(side=tk.LEFT, padx=(0, 2))

        self._acq_btn_start = ttk.Button(btn_frame, text="Start",
                                         command=self._acq_start,
                                         state=tk.DISABLED)
        self._acq_btn_start.pack(side=tk.LEFT, padx=2)

        self._acq_btn_stop = ttk.Button(btn_frame, text="Stop",
                                        command=self._acq_stop,
                                        state=tk.DISABLED)
        self._acq_btn_stop.pack(side=tk.LEFT, padx=2)

        ttk.Button(btn_frame, text="Clear Graphs",
                   command=self._clear_graphs).pack(side=tk.LEFT, padx=(2, 0))

        self._settings_msg_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self._settings_msg_var,
                 fg=CONFIRM_COL, bg=BG, font=("TkDefaultFont", 8)).grid(
            row=0, column=3, rowspan=2, padx=(10, 0), sticky="w")

        self._settings_info_var = tk.StringVar(value=self._settings_info_str())
        tk.Label(inner, textvariable=self._settings_info_var,
                 fg=FG_DIM, bg=BG, font=("TkDefaultFont", 7)).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(2, 0))

    def _settings_info_str(self) -> str:
        n = self._calc_history_n()
        return (f"Active:  {self._rate:,.0f} Hz  ·  {self._history_s:.1f} s  "
                f"·  {n:,} samples  ·  Nyquist {self._rate/2:,.0f} Hz")

    # ── Single matplotlib figure: 2-col GridSpec ──────────────────────────
    # Left col (narrow): XY plot
    # Right col (wide):  V(t) on top, PSD on bottom

    def _build_plots(self, parent):
        self._fig = Figure(figsize=(10, 5.5), dpi=100)

        # Outer: 1 row × 2 cols  (XY left, [Vt+PSD] right)
        outer = gridspec.GridSpec(
            1, 2, figure=self._fig,
            width_ratios=[1, 2.6],
            left=0.06, right=0.97,
            top=0.94, bottom=0.09,
            wspace=0.32,
        )
        # Right sub-grid: 2 rows
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[1],
            hspace=0.48,
        )

        self._ax_xy  = self._fig.add_subplot(outer[0])
        self._ax_vt  = self._fig.add_subplot(inner[0])
        self._ax_psd = self._fig.add_subplot(inner[1])

        self._setup_xy_ax()
        self._setup_vt_ax()
        self._setup_psd_ax()
        self._create_plot_artists()

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_xy_ax(self):
        ax = self._ax_xy
        ax.set_title("XY — CH2 vs CH1", pad=3)
        ax.set_xlabel("CH1 (V)", labelpad=1)
        ax.set_ylabel("CH2 (V)", labelpad=1)
        lim = self._range / 2 * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")

    def _setup_vt_ax(self):
        ax = self._ax_vt
        ax.set_title("CH1 & CH2 — Voltage vs Time", pad=3)
        ax.set_xlabel("Time (s)", labelpad=1)
        ax.set_ylabel("Voltage (V)", labelpad=1)
        ax.set_xlim(0.0, self._history_s)
        ax.set_ylim(-self._range / 2 * 1.15, self._range / 2 * 1.15)
        ax.minorticks_on()
        ax.grid(True, which="minor", linewidth=0.25, color=MINOR_COL)

    def _setup_psd_ax(self):
        ax = self._ax_psd
        ax.set_title("PSD — CH1 & CH2", pad=3)
        ax.set_xlabel("Frequency (Hz)", labelpad=1)
        ax.set_ylabel("PSD (V²/Hz)", labelpad=1)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(1.0, max(self._rate / 2, 2.0))
        ax.xaxis.set_minor_locator(
            plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.yaxis.set_minor_locator(
            plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20))
        ax.tick_params(axis="both", which="minor",
                       labelbottom=False, labelleft=False)
        ax.grid(True, which="minor", linewidth=0.25, color=MINOR_COL)

    def _create_plot_artists(self):
        n  = self._calc_history_n()
        tw = self._time_s

        # XY
        self._line_xy, = self._ax_xy.plot(
            [], [], color=XY_COL, lw=0,
            marker="o", markersize=2.5, markeredgewidth=0, alpha=0.75)

        # V(t)
        self._line_ch1, = self._ax_vt.plot(
            tw, np.zeros(n), color=CH1_COL, lw=0.9, label="CH1")
        self._line_ch2, = self._ax_vt.plot(
            tw, np.zeros(n), color=CH2_COL, lw=0.9, label="CH2")
        self._ax_vt.legend(fontsize=7, loc="upper right", framealpha=0.7)

        self._stat_vt = self._ax_vt.text(
            0.01, 0.98, "",
            fontsize=6.5, va="top", ha="left",
            transform=self._ax_vt.transAxes, color=FG,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PANEL_BG,
                      edgecolor=BORDER, alpha=0.9))

        # PSD
        self._line_psd1, = self._ax_psd.plot(
            [], [], color=CH1_COL, lw=1.1, label="CH1")
        self._line_psd2, = self._ax_psd.plot(
            [], [], color=CH2_COL, lw=1.1, label="CH2")
        self._ax_psd.legend(fontsize=7, loc="upper right", framealpha=0.7)

    # =========================================================================
    # Settings — Apply
    # =========================================================================

    def _settings_apply(self):
        try:
            new_rate = float(self._setting_rate_var.get())
            if new_rate <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Setting",
                                 "Sample frequency must be a positive number.")
            return

        try:
            new_hist = float(self._setting_history_var.get())
            if new_hist <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid Setting",
                                 "History length must be a positive number.")
            return

        rate_changed = (new_rate != self._rate)
        hist_changed = (new_hist != self._history_s)

        if not rate_changed and not hist_changed:
            self._flash_settings("Already applied.")
            return

        self._rate      = new_rate
        self._history_s = new_hist

        new_n = self._calc_history_n()
        self._ring.resize(new_n)
        self._time_s = np.linspace(0.0, self._history_s, new_n)

        # Reset PSD
        with self._psd_lock:
            self._freqs1 = np.array([1.0, 2.0])
            self._psd1   = np.array([1e-9, 1e-9])
            self._freqs2 = np.array([1.0, 2.0])
            self._psd2   = np.array([1e-9, 1e-9])

        # Restart acquisition if rate changed and device is live
        if rate_changed and self._device is not None:
            was_acquiring = self._acq is not None
            self._stop_acquisition(close_device=False)
            if was_acquiring:
                self._acq = AcquisitionThread(self._device, self._rate, self._q)
                self._acq.start()
                self._set_acquiring()

        # Update plot axes
        self._ax_vt.set_xlim(0.0, self._history_s)
        self._ax_psd.set_xlim(1.0, max(self._rate / 2, 2.0))

        # Resize V(t) line x-data
        self._line_ch1.set_xdata(self._time_s)
        self._line_ch1.set_ydata(np.zeros(new_n))
        self._line_ch2.set_xdata(self._time_s)
        self._line_ch2.set_ydata(np.zeros(new_n))

        self._line_psd1.set_data([], [])
        self._line_psd2.set_data([], [])
        self._line_xy.set_data([], [])

        self._canvas.draw_idle()
        self._settings_info_var.set(self._settings_info_str())
        self._flash_settings("✓ Applied")

    def _flash_settings(self, msg: str, duration: float = 2.5):
        self._settings_msg_var.set(msg)
        self._root.after(int(duration * 1000),
                         lambda: self._settings_msg_var.set(""))

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
        # Start enabled when connected but not acquiring; Stop when acquiring
        self._acq_btn_start.config(
            state=tk.NORMAL if connected else tk.DISABLED)
        self._acq_btn_stop.config(state=tk.DISABLED)

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
        self._acq = AcquisitionThread(self._device, self._rate, self._q)
        self._acq.start()
        self._root.after(0, lambda: self._set_ads_status("connected", name))
        self._root.after(0, self._set_acquiring)

    def _ads_on_disconnect(self):
        self._stop_acquisition(close_device=True)
        self._set_ads_status("disconnected")

    def _stop_acquisition(self, close_device: bool = True):
        if self._acq is not None:
            self._acq.stop()
            self._acq.join(timeout=2.0)
            self._acq = None
        if close_device and self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        # Drain the queue
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def _set_acquiring(self):
        """Flip Start/Stop buttons to reflect that acquisition is running."""
        self._acq_btn_start.config(state=tk.DISABLED)
        self._acq_btn_stop.config(state=tk.NORMAL)

    def _acq_start(self):
        """Start (or restart) acquisition while keeping the device open."""
        if self._device is None:
            return
        if self._acq is not None:
            return  # already running
        self._acq = AcquisitionThread(self._device, self._rate, self._q)
        self._acq.start()
        self._set_acquiring()

    def _acq_stop(self):
        """Stop acquisition without disconnecting the device."""
        self._stop_acquisition(close_device=False)
        # Re-enable Start so user can restart
        self._acq_btn_start.config(state=tk.NORMAL)
        self._acq_btn_stop.config(state=tk.DISABLED)

    def _clear_graphs(self):
        """Zero the ring buffer and blank all plot artists."""
        self._ring.resize(self._calc_history_n())
        with self._psd_lock:
            self._freqs1 = np.array([1.0, 2.0])
            self._psd1   = np.array([1e-9, 1e-9])
            self._freqs2 = np.array([1.0, 2.0])
            self._psd2   = np.array([1e-9, 1e-9])
        n = self._calc_history_n()
        self._line_ch1.set_ydata(np.zeros(n))
        self._line_ch2.set_ydata(np.zeros(n))
        self._line_xy.set_data([], [])
        self._line_psd1.set_data([], [])
        self._line_psd2.set_data([], [])
        self._stat_vt.set_text("")
        self._canvas.draw_idle()

    # =========================================================================
    # Stage connect / disconnect
    # =========================================================================

    def _stage_on_connect(self):
        self._set_stage_status("connecting")
        threading.Thread(target=self._stage_connect_worker, daemon=True).start()

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
            threading.Thread(target=stage_control.close_port,
                             args=(self._stage_port,), daemon=True).start()
            self._stage_port = None
        self._set_stage_status("disconnected")

    def _stage_port_ok(self) -> bool:
        return self._stage_port is not None and self._stage_port.is_open

    # =========================================================================
    # Stage motion
    # =========================================================================

    def _stage_get_params(self) -> Tuple[float, float]:
        return float(self._freq_var.get()), float(self._step_var.get())

    def _stage_move_left(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            f, s = self._stage_get_params()
            stage_control.move(self._stage_port, axis=1, distance= s, freq=f)

    def _stage_move_right(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            f, s = self._stage_get_params()
            stage_control.move(self._stage_port, axis=1, distance=-s, freq=f)

    def _stage_move_up(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            f, s = self._stage_get_params()
            stage_control.move(self._stage_port, axis=0, distance= s, freq=f)

    def _stage_move_down(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            f, s = self._stage_get_params()
            stage_control.move(self._stage_port, axis=0, distance=-s, freq=f)

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
    # Poll / draw
    # =========================================================================

    def _poll(self):
        # Drain all pending chunks into the ring buffer
        got_data = False
        try:
            while True:
                ch1_chunk, ch2_chunk = self._q.get_nowait()
                self._ring.push(ch1_chunk, ch2_chunk)
                got_data = True
        except queue.Empty:
            pass

        now = time.time()
        if got_data and (now - self._last_draw) >= self._draw_interval:
            self._redraw(now)

        self._root.after(self._POLL_MS, self._poll)

    def _redraw(self, now: float):
        self._last_draw = now
        self._frame_count += 1

        ch1, ch2 = self._ring.view()

        # V(t)
        self._line_ch1.set_ydata(ch1)
        self._line_ch2.set_ydata(ch2)
        pk = max(abs(ch1).max(), abs(ch2).max(), 1e-9)
        self._ax_vt.set_ylim(-pk * 1.15, pk * 1.15)
        rms1 = float(np.sqrt(np.mean(ch1 ** 2)))
        rms2 = float(np.sqrt(np.mean(ch2 ** 2)))
        self._stat_vt.set_text(
            f"CH1  pk {ch1.max():+.3f}  min {ch1.min():+.3f}  "
            f"rms {rms1:.3f}  μ {ch1.mean():+.4f}\n"
            f"CH2  pk {ch2.max():+.3f}  min {ch2.min():+.3f}  "
            f"rms {rms2:.3f}  μ {ch2.mean():+.4f}"
        )

        # XY
        self._line_xy.set_data(ch1, ch2)
        lim = max(abs(ch1).max(), abs(ch2).max(), 1e-9) * 1.15
        self._ax_xy.set_xlim(-lim, lim)
        self._ax_xy.set_ylim(-lim, lim)

        # PSD — kick off a background thread every _PSD_EVERY frames
        self._psd_countdown -= 1
        if self._psd_countdown <= 0:
            self._psd_countdown = self._PSD_EVERY
            if not self._psd_busy:
                self._psd_busy = True
                threading.Thread(
                    target=self._compute_psd_bg,
                    args=(ch1.copy(), ch2.copy()),
                    daemon=True,
                ).start()

        # Apply last-computed PSD results
        with self._psd_lock:
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
            fps           = self._frame_count / elapsed
            self._fps_t0  = now
            self._frame_count = 0
            self._fps_var.set(f"FPS: {fps:.1f}")

        self._canvas.draw_idle()

    def _compute_psd_bg(self, ch1: np.ndarray, ch2: np.ndarray):
        """Runs in a daemon thread; writes results under the lock."""
        f1, p1 = compute_psd(ch1, self._rate)
        f2, p2 = compute_psd(ch2, self._rate)
        with self._psd_lock:
            self._freqs1 = f1
            self._psd1   = p1
            self._freqs2 = f2
            self._psd2   = p2
        self._psd_busy = False

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
            [f"# Sample rate: {self._rate:.0f} Hz",
             f"History: {self._history_s:.1f} s",
             f"N: {self._calc_history_n()}"],
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
        self._acq_stop()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = filedialog.asksaveasfilename(
            title="Export PSD",
            initialfile=f"psd_ch1_ch2_{stamp}.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not fname:
            return
        with self._psd_lock:
            f1, p1 = self._freqs1.copy(), self._psd1.copy()
            f2, p2 = self._freqs2.copy(), self._psd2.copy()
        if len(f1) != len(f2) or not np.allclose(f1, f2):
            p2 = np.interp(f1, f2, p2)
        try:
            self._write_csv(
                fname, self._meta_rows(stamp),
                ["frequency_hz", "psd_ch1_v2_per_hz", "psd_ch2_v2_per_hz"],
                zip(f1, p1, p2),
            )
            self._flash_export(f"✓ Saved {os.path.basename(fname)}")
        except OSError as exc:
            messagebox.showerror("Export Error", str(exc))

    def _export_vt(self):
        self._acq_stop()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = filedialog.asksaveasfilename(
            title="Export V(t)",
            initialfile=f"vt_ch1_ch2_{stamp}.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not fname:
            return
        ch1, ch2 = self._ring.view()
        try:
            self._write_csv(
                fname, self._meta_rows(stamp),
                ["time_s", "ch1_v", "ch2_v"],
                zip(self._time_s, ch1, ch2),
            )
            self._flash_export(f"✓ Saved {os.path.basename(fname)}")
        except OSError as exc:
            messagebox.showerror("Export Error", str(exc))

    # =========================================================================
    # Close
    # =========================================================================

    def _on_close(self):
        self._stop_acquisition(close_device=True)
        self._stage_on_disconnect()
        self._root.destroy()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    root = tk.Tk()
    OscilloscopeApp(root)
    root.mainloop()