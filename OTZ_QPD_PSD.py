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
  │    [● status]        │    Sample freq (Hz): [______] [Apply]     │
  │    [Connect][Disc.]  │    History length (s): [______]           │
  │    ○ Joystick        │                                           │
  │    ○ Software        ├───────────────────────────────────────────┤
  │    ○ Coarse / Fine   │  CH1 & CH2 — Voltage vs Time             │
  │    Freq / Step       │                                           │
  │    [▲][◄][►][▼]     ├───────────────────────────────────────────┤
  ├──────────────────────┤  PSD  CH1 & CH2                          │
  │  XY — CH2 vs CH1    │                                           │
  └──────────────────────┴───────────────────────────────────────────┘
  [ Export PSD ]  [ Export V(t) ]

Requirements
------------
  pip install numpy matplotlib pyserial
  stage_serial.py in the same directory
  waveforms_ads.py importable; Digilent WaveForms + Adept Runtime installed

Usage
-----
  python OTZ_QPD_PSD.py
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

import stage_serial

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
BG          = "#1a1a2e"
PANEL_BG    = "#16213e"
BORDER      = "#0f3460"
FG          = "#e0e0e0"
FG_DIM      = "#8888aa"

CH1_COL     = "#4fc3f7"
CH2_COL     = "#ef5350"
XY_COL      = "#69f0ae"
GRID_COL    = "#2a2a4a"
MINOR_COL   = "#222240"

STATUS_CONNECTED    = "#43d17a"
STATUS_CONNECTING   = "#90caf9"
STATUS_DISCONNECTED = "#ef5350"
CONFIRM_COL         = "#43d17a"

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
    for widget in ("TFrame", "TLabel", "TLabelframe", "TLabelframe.Label",
                   "TCheckbutton", "TRadiobutton"):
        s.configure(widget, background=BG, foreground=FG)
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
                insertcolor=FG, bordercolor=BORDER, arrowcolor=FG)
    s.configure("TSeparator", background=BORDER)
    s.configure("TRadiobutton", background=BG, foreground=FG,
                indicatorcolor=BORDER, focuscolor=BG)
    s.map("TRadiobutton",
          indicatorcolor=[("selected", STATUS_CONNECTED)])
    s.configure("TLabelframe", background=BG, bordercolor=BORDER)
    s.configure("TLabelframe.Label", background=BG, foreground=FG_DIM)


# ===========================================================================
# PSD helper
# ===========================================================================

def compute_psd(signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.signal import welch
        nperseg = min(len(signal), 4096)
        return welch(signal, fs=sample_rate, nperseg=nperseg, scaling="density")
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
    """
    Continuously acquires fixed-size chunks from the ADS and pushes them to
    a queue.  The chunk size (_n) is chosen to be ~50 ms worth of samples so
    the queue stays responsive regardless of history length.
    """
    def __init__(self, device, sample_rate: float, chunk_size: int,
                 out_queue: queue.Queue):
        super().__init__(daemon=True)
        self._dev   = device
        self._rate  = sample_rate
        self._n     = chunk_size
        self._q     = out_queue
        self._stop  = threading.Event()

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

# Defaults
_DEFAULT_SAMPLE_RATE_HZ = 8_000
_DEFAULT_HISTORY_S      = 8.0
_DEFAULT_INPUT_RANGE_V  = 5.0
_DEFAULT_DEVICE_INDEX   = -1
# Acquisition chunk: aim for ~50 ms per chunk (minimum 64 samples)
_CHUNK_MS = 50

class OscilloscopeApp:

    _POLL_MS      = 50
    _MIN_DRAW_SEC = 0.08

    def __init__(self, root: tk.Tk):
        self._root = root

        # Live acquisition settings (updated on Apply)
        self._rate      = float(_DEFAULT_SAMPLE_RATE_HZ)
        self._history_s = float(_DEFAULT_HISTORY_S)
        self._range     = float(_DEFAULT_INPUT_RANGE_V)

        # ADS state
        self._device: Optional[object] = None
        self._acq:    Optional[AcquisitionThread] = None
        self._q       = queue.Queue(maxsize=4)

        # Stage state
        self._stage_port:     Optional[object] = None
        self._stage_ctrl_type = "joystick"
        self._stage_speed     = "coarse"

        # Rolling history buffers — sized to history_s * rate
        self._history_n = self._calc_history_n()
        self._hist_ch1  = np.zeros(self._history_n)
        self._hist_ch2  = np.zeros(self._history_n)
        self._time_s    = np.linspace(0.0, self._history_s, self._history_n)

        # PSD buffers (last computed)
        self._freqs1 = np.array([1.0, 2.0])
        self._psd1   = np.array([1e-9, 1e-9])
        self._freqs2 = np.array([1.0, 2.0])
        self._psd2   = np.array([1e-9, 1e-9])

        # Draw bookkeeping
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

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _calc_history_n(self) -> int:
        """Total samples in the rolling display buffer."""
        return max(64, int(round(self._rate * self._history_s)))

    def _calc_chunk_n(self) -> int:
        """Acquisition chunk size: ~_CHUNK_MS ms, rounded to power-of-2 minimum."""
        n = int(round(self._rate * _CHUNK_MS / 1000))
        return max(64, n)

    # =========================================================================
    # UI construction
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

        self._ads_device_lbl = tk.Label(
            ads_bar, text="Device: —", fg=FG_DIM, bg=BG)
        self._ads_device_lbl.pack(side=tk.LEFT, padx=(0, 18))

        tk.Label(ads_bar, text="Device index:", fg=FG, bg=BG).pack(side=tk.LEFT)
        self._ads_idx_var = tk.IntVar(value=_DEFAULT_DEVICE_INDEX)
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

        # ── Main content: left panel + right column ───────────────────────
        content = ttk.Frame(root)
        content.pack(fill=tk.BOTH, expand=True)

        self._build_left_panel(content)
        self._build_right_column(content)

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

    # ── Left panel: stage control + XY plot ──────────────────────────────

    def _build_left_panel(self, parent):
        """
        Left column: stage control widget on top, XY matplotlib axes below.
        The XY axes is embedded in its own small Figure so it can sit naturally
        alongside the Tk widgets without fighting the right-column GridSpec.
        """
        left = tk.Frame(parent, bg=BG, width=230)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        left.pack_propagate(False)

        self._build_stage_panel(left)

        # Thin separator between stage controls and XY plot
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # XY figure embedded in left column
        self._fig_xy = Figure(figsize=(2.3, 2.3), dpi=100)
        self._ax_xy  = self._fig_xy.add_subplot(111)
        self._fig_xy.subplots_adjust(left=0.18, right=0.97, top=0.90, bottom=0.16)
        self._setup_xy_ax()

        canvas_xy = FigureCanvasTkAgg(self._fig_xy, master=left)
        canvas_xy.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._canvas_xy = canvas_xy

        # XY artist
        self._line_xy, = self._ax_xy.plot(
            [], [],
            color=XY_COL, lw=0,
            marker="o", markersize=2.5, markeredgewidth=0,
            alpha=0.75,
        )

    def _build_stage_panel(self, parent):
        # Stage status row
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

    # ── Right column: settings panel + V(t) + PSD ────────────────────────

    def _build_right_column(self, parent):
        right = ttk.Frame(parent)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_settings_panel(right)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(4, 0))

        self._build_plots(right)

    def _build_settings_panel(self, parent):
        settings = ttk.LabelFrame(parent, text="Oscilloscope Settings")
        settings.pack(fill=tk.X, pady=(0, 2))

        inner = ttk.Frame(settings, padding=(6, 4))
        inner.pack(fill=tk.X)

        # Sample frequency
        ttk.Label(inner, text="Sample frequency (Hz):").grid(
            row=0, column=0, sticky="w", padx=(0, 6), pady=3)
        self._setting_rate_var = tk.StringVar(
            value=str(int(_DEFAULT_SAMPLE_RATE_HZ)))
        ttk.Entry(inner, textvariable=self._setting_rate_var,
                  width=10).grid(row=0, column=1, sticky="w", pady=3)

        # History length
        ttk.Label(inner, text="History length (s):").grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=3)
        self._setting_history_var = tk.StringVar(
            value=str(_DEFAULT_HISTORY_S))
        ttk.Entry(inner, textvariable=self._setting_history_var,
                  width=10).grid(row=1, column=1, sticky="w", pady=3)

        # Apply button + feedback label on same row as the button
        btn_row = ttk.Frame(inner)
        btn_row.grid(row=0, column=2, rowspan=2, padx=(14, 0), sticky="ns")

        ttk.Button(btn_row, text="Apply",
                   command=self._settings_apply).pack(anchor="center", expand=True)

        self._settings_msg_var = tk.StringVar(value="")
        tk.Label(inner, textvariable=self._settings_msg_var,
                 fg=CONFIRM_COL, bg=BG, font=("TkDefaultFont", 8)).grid(
            row=0, column=3, rowspan=2, padx=(10, 0), sticky="w")

        # Read-only info label showing active settings
        self._settings_info_var = tk.StringVar(value=self._settings_info_str())
        tk.Label(inner, textvariable=self._settings_info_var,
                 fg=FG_DIM, bg=BG, font=("TkDefaultFont", 7)).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(2, 0))

    def _settings_info_str(self) -> str:
        history_n = self._calc_history_n()
        nyquist   = self._rate / 2
        return (f"Active:  {self._rate:,.0f} Hz  ·  {self._history_s:.1f} s  "
                f"·  {history_n:,} samples  ·  Nyquist {nyquist:,.0f} Hz")

    # ── Matplotlib plots (V(t) + PSD) ─────────────────────────────────────

    def _build_plots(self, parent):
        self._fig = Figure(figsize=(9, 5.5), dpi=100)
        gs = gridspec.GridSpec(
            2, 1, figure=self._fig,
            hspace=0.42,
            left=0.08, right=0.97,
            top=0.94, bottom=0.09,
        )
        self._ax_vt  = self._fig.add_subplot(gs[0])
        self._ax_psd = self._fig.add_subplot(gs[1])

        self._setup_vt_ax()
        self._setup_psd_ax()
        self._create_plot_artists()

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_vt_ax(self):
        ax = self._ax_vt
        ax.set_title("CH1 & CH2 — Voltage vs Time", pad=3)
        ax.set_xlabel("Time (s)", labelpad=1)
        ax.set_ylabel("Voltage (V)", labelpad=1)
        ax.set_xlim(0.0, self._history_s)
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
        ax.set_aspect("equal", adjustable="box")

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

    def _create_plot_artists(self):
        tw = self._time_s

        self._line_ch1, = self._ax_vt.plot(
            tw, self._hist_ch1, color=CH1_COL, lw=0.9, label="CH1")
        self._line_ch2, = self._ax_vt.plot(
            tw, self._hist_ch2, color=CH2_COL, lw=0.9, label="CH2")
        self._ax_vt.legend(fontsize=7, loc="upper right", framealpha=0.7)

        self._stat_vt = self._ax_vt.text(
            0.01, 0.98, "",
            fontsize=6.5, va="top", ha="left",
            transform=self._ax_vt.transAxes,
            color=FG,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PANEL_BG,
                      edgecolor=BORDER, alpha=0.9),
        )

        self._line_psd1, = self._ax_psd.plot(
            [], [], color=CH1_COL, lw=1.1, label="CH1")
        self._line_psd2, = self._ax_psd.plot(
            [], [], color=CH2_COL, lw=1.1, label="CH2")
        self._ax_psd.legend(fontsize=7, loc="upper right", framealpha=0.7)

    # =========================================================================
    # Settings — Apply
    # =========================================================================

    def _settings_apply(self):
        # Parse and validate
        try:
            new_rate = float(self._setting_rate_var.get())
            if new_rate <= 0:
                raise ValueError("must be > 0")
        except ValueError:
            messagebox.showerror("Invalid Setting",
                                 "Sample frequency must be a positive number.")
            return

        try:
            new_history = float(self._setting_history_var.get())
            if new_history <= 0:
                raise ValueError("must be > 0")
        except ValueError:
            messagebox.showerror("Invalid Setting",
                                 "History length must be a positive number.")
            return

        rate_changed    = (new_rate    != self._rate)
        history_changed = (new_history != self._history_s)

        if not rate_changed and not history_changed:
            self._flash_settings("Already applied.")
            return

        self._rate      = new_rate
        self._history_s = new_history

        # Rebuild history buffers
        self._history_n = self._calc_history_n()
        self._hist_ch1  = np.zeros(self._history_n)
        self._hist_ch2  = np.zeros(self._history_n)
        self._time_s    = np.linspace(0.0, self._history_s, self._history_n)

        # Reset PSD buffers so stale data is not displayed
        self._freqs1 = np.array([1.0, 2.0])
        self._psd1   = np.array([1e-9, 1e-9])
        self._freqs2 = np.array([1.0, 2.0])
        self._psd2   = np.array([1e-9, 1e-9])

        # If the sample rate changed and ADS is running, restart acquisition
        if rate_changed and self._device is not None:
            self._stop_acquisition()
            chunk_n = self._calc_chunk_n()
            self._acq = AcquisitionThread(
                self._device, self._rate, chunk_n, self._q)
            self._acq.start()

        # Update plot axes and artists to match new settings
        self._ax_vt.set_xlim(0.0, self._history_s)
        self._ax_psd.set_xlim(1.0, self._rate / 2)

        # Rebuild V(t) line x-data (length changed)
        self._line_ch1.set_xdata(self._time_s)
        self._line_ch1.set_ydata(self._hist_ch1)
        self._line_ch2.set_xdata(self._time_s)
        self._line_ch2.set_ydata(self._hist_ch2)

        # Clear PSD lines
        self._line_psd1.set_data([], [])
        self._line_psd2.set_data([], [])

        self._canvas.draw_idle()
        self._canvas_xy.draw_idle()

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

        chunk_n = self._calc_chunk_n()
        self._acq = AcquisitionThread(self._device, self._rate, chunk_n, self._q)
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
        threading.Thread(target=self._stage_connect_worker, daemon=True).start()

    def _stage_connect_worker(self):
        try:
            port = stage_serial.open_port()
            stage_serial.init_stage(port)
            self._stage_port = port
        except Exception as exc:
            self._root.after(0, lambda: self._set_stage_status("disconnected"))
            self._root.after(0, lambda: messagebox.showerror(
                "Stage Connection Failed", str(exc)))
            return
        self._root.after(0, lambda: self._set_stage_status("connected"))

    def _stage_on_disconnect(self):
        if self._stage_port is not None:
            threading.Thread(target=stage_serial.close_port,
                             args=(self._stage_port,), daemon=True).start()
            self._stage_port = None
        self._set_stage_status("disconnected")

    def _stage_port_ok(self) -> bool:
        return self._stage_port is not None and self._stage_port.is_open

    # =========================================================================
    # Stage motion / mode callbacks
    # =========================================================================

    def _stage_get_params(self) -> Tuple[float, float]:
        return float(self._freq_var.get()), float(self._step_var.get())

    def _stage_move_left(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            freq, step = self._stage_get_params()
            stage_serial.move(self._stage_port, axis=1, distance= step, freq=freq)

    def _stage_move_right(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            freq, step = self._stage_get_params()
            stage_serial.move(self._stage_port, axis=1, distance=-step, freq=freq)

    def _stage_move_up(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            freq, step = self._stage_get_params()
            stage_serial.move(self._stage_port, axis=0, distance= step, freq=freq)

    def _stage_move_down(self):
        if self._stage_ctrl_type == "software" and self._stage_port_ok():
            freq, step = self._stage_get_params()
            stage_serial.move(self._stage_port, axis=0, distance=-step, freq=freq)

    def _stage_on_ctrl_change(self):
        if not self._stage_port_ok():
            return
        if self._ctrl_var.get() == "Software Control":
            stage_serial.joystick_off(self._stage_port)
            self._stage_ctrl_type = "software"
        else:
            stage_serial.joystick_on(self._stage_port)
            self._stage_ctrl_type = "joystick"
            stage_serial.set_resolution(
                self._stage_port, fine=(self._stage_speed == "fine"))

    def _stage_on_speed_change(self):
        if self._spd_var.get() == "Fine Control":
            self._stage_speed = "fine"
            if self._stage_port_ok() and self._stage_ctrl_type == "joystick":
                stage_serial.set_resolution(self._stage_port, fine=True)
        else:
            self._stage_speed = "coarse"
            if self._stage_port_ok() and self._stage_ctrl_type == "joystick":
                stage_serial.set_resolution(self._stage_port, fine=False)

    # =========================================================================
    # Poll / draw loop
    # =========================================================================

    def _poll(self):
        now = time.time()
        if now - self._last_draw >= self._MIN_DRAW_SEC:
            # Drain all queued chunks, appending each to the rolling buffers
            got_data = False
            try:
                while True:
                    ch1_chunk, ch2_chunk = self._q.get_nowait()
                    n = len(ch1_chunk)
                    self._hist_ch1 = np.roll(self._hist_ch1, -n)
                    self._hist_ch2 = np.roll(self._hist_ch2, -n)
                    self._hist_ch1[-n:] = ch1_chunk
                    self._hist_ch2[-n:] = ch2_chunk
                    got_data = True
            except queue.Empty:
                pass

            if got_data:
                self._redraw(now)

        self._root.after(self._POLL_MS, self._poll)

    def _redraw(self, now):
        self._last_draw = now
        self._frame_count += 1

        ch1 = self._hist_ch1
        ch2 = self._hist_ch2

        # V(t)
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

        # XY (separate canvas)
        self._line_xy.set_xdata(ch1)
        self._line_xy.set_ydata(ch2)
        lim = max(abs(ch1).max(), abs(ch2).max(), 1e-9) * 1.15
        self._ax_xy.set_xlim(-lim, lim)
        self._ax_xy.set_ylim(-lim, lim)

        # PSD — every other draw frame
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
        self._canvas_xy.draw_idle()

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
             f"N: {self._history_n}"],
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
                ["time_s", "ch1_v", "ch2_v"],
                zip(self._time_s, self._hist_ch1, self._hist_ch2),
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