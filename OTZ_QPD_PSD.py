"""
live_scope_tk.py
================
Live dual-channel oscilloscope dashboard — Tkinter GUI edition.
Uses Matplotlib embedded in a Tk window; hardware driven by Digilent WaveForms.

Layout
------
  ┌──────────────────────────────────────────────────────────┐
  │  [●  CONNECTED / ○  DISCONNECTED]  Device: <name>       │  ← status bar
  │  [Connect]  [Disconnect]   Device index: [__]           │  ← controls
  ├─────────────────────┬────────────────────────────────────┤
  │  CH1  V vs t        │  CH2  V vs t                      │
  ├─────────────────────┼────────────────────────────────────┤
  │  CH2 vs CH1 (XY)   │  PSD  CH1 & CH2                   │
  └─────────────────────┴────────────────────────────────────┘
  [ Export PSD (CH1+CH2) ]   [ Export V(t) (CH1+CH2) ]

Requirements
------------
  pip install numpy matplotlib
  waveforms_ads.py importable (same directory or PYTHONPATH)
  Digilent WaveForms + Adept Runtime installed

Usage
-----
  python live_scope_tk.py
  python live_scope_tk.py --rate 200000 --samples 4096 --range 5.0
"""

import argparse
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

# ---------------------------------------------------------------------------
# Hardware driver
# ---------------------------------------------------------------------------
try:
    from waveforms_ads import WaveFormsADS, DWFError
    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
CH1_COL     = "#1f77b4"
CH2_COL     = "#d62728"
XY_COL      = "#2ca02c"
CONFIRM_COL = "#e67e22"

STATUS_CONNECTED    = "#27ae60"   # green dot
STATUS_CONNECTING   = "#f39c12"   # amber dot
STATUS_DISCONNECTED = "#c0392b"   # red dot

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.6,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "lines.antialiased": True,
    "figure.autolayout": False,
})


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
# Acquisition thread
# ===========================================================================

class AcquisitionThread(threading.Thread):
    def __init__(self, device, sample_rate: float, n_samples: int, out_queue: queue.Queue):
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

    _POLL_MS        = 50     # ms between Tk .after() polls
    _MIN_DRAW_SEC   = 0.08   # ~12 fps cap

    def __init__(self, root: tk.Tk, sample_rate=100_000, n_samples=4096,
                 input_range=5.0, device_index=-1):
        self._root        = root
        self._rate        = float(sample_rate)
        self._n           = int(n_samples)
        self._range       = float(input_range)
        self._dev_index   = device_index

        self._device: Optional[object] = None
        self._acq:    Optional[AcquisitionThread] = None
        self._q       = queue.Queue(maxsize=2)

        self._ch1    = np.zeros(self._n)
        self._ch2    = np.zeros(self._n)
        self._freqs1 = np.array([1.0, 2.0])
        self._psd1   = np.array([1e-9, 1e-9])
        self._freqs2 = np.array([1.0, 2.0])
        self._psd2   = np.array([1e-9, 1e-9])
        self._time_ms = np.linspace(0, self._n / self._rate * 1e3, self._n)

        self._last_draw   = 0.0
        self._frame_count = 0
        self._psd_skip    = 0
        self._fps_t0      = time.time()
        self._fps         = 0.0

        root.title("WaveForms Live Scope")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._set_status("disconnected")
        self._root.after(self._POLL_MS, self._poll)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = self._root

        # ── Top control bar ──────────────────────────────────────────
        ctrl_frame = ttk.Frame(root, padding=(6, 4))
        ctrl_frame.pack(side=tk.TOP, fill=tk.X)

        # Status indicator (canvas circle + label)
        self._status_canvas = tk.Canvas(ctrl_frame, width=14, height=14,
                                        highlightthickness=0, bg=root.cget("bg"))
        self._status_canvas.pack(side=tk.LEFT, padx=(0, 4))
        self._status_dot = self._status_canvas.create_oval(2, 2, 12, 12,
                                                           fill=STATUS_DISCONNECTED,
                                                           outline="")

        self._status_label = ttk.Label(ctrl_frame, text="Disconnected",
                                       font=("TkDefaultFont", 9, "bold"))
        self._status_label.pack(side=tk.LEFT, padx=(0, 16))

        self._device_label = ttk.Label(ctrl_frame, text="Device: —",
                                       foreground="#555555")
        self._device_label.pack(side=tk.LEFT, padx=(0, 24))

        # Device index spinbox
        ttk.Label(ctrl_frame, text="Device index:").pack(side=tk.LEFT)
        self._dev_idx_var = tk.IntVar(value=self._dev_index)
        ttk.Spinbox(ctrl_frame, from_=-1, to=15, width=4,
                    textvariable=self._dev_idx_var).pack(side=tk.LEFT, padx=(2, 10))

        # Connect / Disconnect buttons
        self._btn_connect = ttk.Button(ctrl_frame, text="Connect",
                                       command=self._on_connect)
        self._btn_connect.pack(side=tk.LEFT, padx=2)

        self._btn_disconnect = ttk.Button(ctrl_frame, text="Disconnect",
                                          command=self._on_disconnect,
                                          state=tk.DISABLED)
        self._btn_disconnect.pack(side=tk.LEFT, padx=2)

        # FPS label (right-aligned)
        self._fps_var = tk.StringVar(value="FPS: —")
        ttk.Label(ctrl_frame, textvariable=self._fps_var,
                  foreground="#888888").pack(side=tk.RIGHT, padx=6)

        ttk.Separator(root, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # ── Matplotlib figure ─────────────────────────────────────────
        self._fig = Figure(figsize=(12, 7), dpi=100)
        self._build_plots()

        self._canvas = FigureCanvasTkAgg(self._fig, master=root)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Separator(root, orient=tk.HORIZONTAL).pack(fill=tk.X)

        # ── Bottom export bar ─────────────────────────────────────────
        export_frame = ttk.Frame(root, padding=(6, 4))
        export_frame.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(export_frame, text="Export PSD (CH1+CH2)",
                   command=self._export_psd).pack(side=tk.LEFT, padx=4)
        ttk.Button(export_frame, text="Export V(t) (CH1+CH2)",
                   command=self._export_vt).pack(side=tk.LEFT, padx=4)

        self._export_status_var = tk.StringVar(value="")
        self._export_status_lbl = ttk.Label(export_frame,
                                            textvariable=self._export_status_var,
                                            foreground=CONFIRM_COL)
        self._export_status_lbl.pack(side=tk.LEFT, padx=8)

    def _build_plots(self):
        gs = gridspec.GridSpec(
            2, 2, figure=self._fig,
            hspace=0.38, wspace=0.28,
            left=0.07, right=0.97,
            top=0.94, bottom=0.05,
        )
        self._ax_ch1 = self._fig.add_subplot(gs[0, 0])
        self._ax_ch2 = self._fig.add_subplot(gs[0, 1])
        self._ax_xy  = self._fig.add_subplot(gs[1, 0])
        self._ax_psd = self._fig.add_subplot(gs[1, 1])

        self._setup_vt_ax(self._ax_ch1, "CH1 — Voltage vs Time")
        self._setup_vt_ax(self._ax_ch2, "CH2 — Voltage vs Time")
        self._setup_xy_ax()
        self._setup_psd_ax()
        self._create_artists()

        self._fig.text(
            0.01, 0.975,
            f"WaveForms Scope  ·  {self._rate/1e3:.0f} kHz  ·  {self._n} samp",
            ha="left", va="top", fontsize=7, color="#888888",
        )

    def _setup_vt_ax(self, ax, title):
        ax.set_title(title, pad=3)
        ax.set_xlabel("Time (ms)", labelpad=1)
        ax.set_ylabel("Voltage (V)", labelpad=1)
        ax.set_xlim(self._time_ms[0], self._time_ms[-1])
        ax.set_ylim(-self._range / 2 * 1.15, self._range / 2 * 1.15)
        ax.minorticks_on()
        ax.grid(True, which="minor", linewidth=0.25, color="#eeeeee")

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
        ax.grid(True, which="minor", linewidth=0.25, color="#eeeeee")

    def _create_artists(self):
        tw = self._time_ms
        self._line_ch1, = self._ax_ch1.plot(tw, self._ch1, color=CH1_COL, lw=0.9)
        self._line_ch2, = self._ax_ch2.plot(tw, self._ch2, color=CH2_COL, lw=0.9)
        self._line_xy,  = self._ax_xy.plot(
            self._ch1, self._ch2,
            color=XY_COL, lw=0, marker=",", markersize=1, alpha=0.5)
        self._line_psd1, = self._ax_psd.plot([], [], color=CH1_COL, lw=1.0, label="CH1")
        self._line_psd2, = self._ax_psd.plot([], [], color=CH2_COL, lw=1.0, label="CH2")
        self._ax_psd.legend(fontsize=7, loc="upper right", framealpha=0.8)

        stat_kw = dict(
            fontsize=7, va="top", ha="right",
            transform=self._ax_ch1.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85),
        )
        self._stat_ch1 = self._ax_ch1.text(0.99, 0.98, "", **stat_kw)
        stat_kw["transform"] = self._ax_ch2.transAxes
        self._stat_ch2 = self._ax_ch2.text(0.99, 0.98, "", **stat_kw)

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, state: str, device_name: str = "—"):
        """state: 'connected' | 'connecting' | 'disconnected'"""
        colours = {
            "connected":    (STATUS_CONNECTED,    "Connected",    "bold"),
            "connecting":   (STATUS_CONNECTING,   "Connecting…",  "bold"),
            "disconnected": (STATUS_DISCONNECTED, "Disconnected", "normal"),
        }
        col, text, weight = colours.get(state, colours["disconnected"])
        self._status_canvas.itemconfig(self._status_dot, fill=col)
        self._status_label.config(text=text,
                                  font=("TkDefaultFont", 9, weight),
                                  foreground=col)
        self._device_label.config(text=f"Device: {device_name}")

        connected = (state == "connected")
        self._btn_connect.config(   state=tk.DISABLED if connected else tk.NORMAL)
        self._btn_disconnect.config(state=tk.NORMAL   if connected else tk.DISABLED)

    # ------------------------------------------------------------------
    # Connect / Disconnect
    # ------------------------------------------------------------------

    def _on_connect(self):
        if not HW_AVAILABLE:
            messagebox.showerror(
                "Driver Missing",
                "waveforms_ads is not installed or not importable.\n"
                "Install it and ensure Digilent Adept Runtime is present.")
            return
        self._set_status("connecting")
        idx = self._dev_idx_var.get()
        threading.Thread(target=self._connect_worker, args=(idx,),
                         daemon=True).start()

    def _connect_worker(self, index: int):
        try:
            dev = WaveFormsADS(device_index=index, auto_configure=0)
            dev.analog_in_set_range(0, self._range)
            dev.analog_in_set_range(1, self._range)
            self._device = dev
            name = str(dev)
        except Exception as exc:
            self._root.after(0, lambda: self._set_status("disconnected"))
            self._root.after(0, lambda: messagebox.showerror(
                "Connection Failed", str(exc)))
            return

        self._acq = AcquisitionThread(self._device, self._rate, self._n, self._q)
        self._acq.start()
        self._root.after(0, lambda: self._set_status("connected", name))

    def _on_disconnect(self):
        self._stop_acquisition()
        self._set_status("disconnected")

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
        # Drain queue
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Poll / draw  (replaces FuncAnimation — driven by Tk event loop)
    # ------------------------------------------------------------------

    def _poll(self):
        now = time.time()
        if now - self._last_draw >= self._MIN_DRAW_SEC:
            # Drain to latest frame
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

        self._line_ch1.set_ydata(ch1)
        self._line_ch2.set_ydata(ch2)

        for ax, sig, stat in (
            (self._ax_ch1, ch1, self._stat_ch1),
            (self._ax_ch2, ch2, self._stat_ch2),
        ):
            pk = max(abs(sig.max()), abs(sig.min()), 1e-9)
            ax.set_ylim(-pk * 1.15, pk * 1.15)
            rms = np.sqrt(np.mean(sig ** 2))
            stat.set_text(
                f"pk {sig.max():+.3f}  min {sig.min():+.3f}\n"
                f"rms {rms:.3f}  μ {sig.mean():+.4f}"
            )

        self._line_xy.set_xdata(ch1)
        self._line_xy.set_ydata(ch2)
        lim = max(abs(ch1).max(), abs(ch2).max(), 1e-9) * 1.15
        self._ax_xy.set_xlim(-lim, lim)
        self._ax_xy.set_ylim(-lim, lim)

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

        elapsed = now - self._fps_t0
        if elapsed >= 1.5:
            self._fps         = self._frame_count / elapsed
            self._fps_t0      = now
            self._frame_count = 0
            self._fps_var.set(f"FPS: {self._fps:.1f}")

        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _flash_export(self, msg: str, duration: float = 3.0):
        self._export_status_var.set(msg)
        self._root.after(int(duration * 1000),
                         lambda: self._export_status_var.set(""))

    def _meta_rows(self, stamp):
        return [
            [f"# WaveForms Export  {stamp}"],
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

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def _on_close(self):
        self._stop_acquisition()
        self._root.destroy()


# ===========================================================================
# Entry point
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="WaveForms live dual-channel scope (Tkinter).")
    p.add_argument("--device",  type=int,   default=-1,      metavar="IDX")
    p.add_argument("--rate",    type=float, default=100_000, metavar="HZ")
    p.add_argument("--samples", type=int,   default=4096,    metavar="N")
    p.add_argument("--range",   type=float, default=5.0,     metavar="V",
                   dest="voltage_range")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(
        f"WaveForms Live Scope (Tkinter)\n"
        f"  Rate:    {args.rate/1e3:.1f} kHz\n"
        f"  Buffer:  {args.samples} samples  ({args.samples/args.rate*1e3:.1f} ms)\n"
        f"  Range:   ±{args.voltage_range/2:.2f} V\n"
    )
    root = tk.Tk()
    app = OscilloscopeApp(
        root,
        sample_rate=args.rate,
        n_samples=args.samples,
        input_range=args.voltage_range,
        device_index=args.device,
    )
    root.mainloop()