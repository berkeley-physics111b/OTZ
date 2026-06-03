"""
live_scope.py
=============
Live dual-channel oscilloscope dashboard for Digilent WaveForms hardware.

Layout
------
  ┌─────────────────────┬─────────────────────┐
  │  CH1  V vs t        │  CH2  V vs t        │
  ├─────────────────────┼─────────────────────┤
  │  CH2 vs CH1 (XY)   │  PSD  CH1 & CH2     │
  └─────────────────────┴─────────────────────┘
  [ EXPORT PSD (CH1+CH2) ]  [ EXPORT V(t) (CH1+CH2) ]

Requirements
------------
  pip install numpy matplotlib
  waveforms_ads.py must be importable (same directory or on PYTHONPATH)
  Digilent WaveForms + Adept Runtime installed

Usage
-----
  python live_scope.py                    # auto-open first device
  python live_scope.py --device 1         # open device index 1
  python live_scope.py --rate 200000      # 200 kHz sample rate
  python live_scope.py --samples 4096     # buffer size
  python live_scope.py --range 5.0        # ±2.5 V input range
"""

import argparse
import csv
import os
import sys
import time
import threading
import queue
from datetime import datetime
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

# ---------------------------------------------------------------------------
# Hardware driver
# ---------------------------------------------------------------------------
try:
    from waveforms_ads import WaveFormsADS, DWFError
    HW_AVAILABLE = True
except ImportError:
    print("Waveforms packages not installed. Close and try again")

# ---------------------------------------------------------------------------
# Colours — distinct per channel, otherwise plain
# ---------------------------------------------------------------------------
CH1_COL     = "#1f77b4"   # blue
CH2_COL     = "#d62728"   # red
XY_COL      = "#2ca02c"   # green
CONFIRM_COL = "#e67e22"   # orange for save confirmations

# ---------------------------------------------------------------------------
# Style — standard matplotlib
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
# Acquisition thread
# ===========================================================================

class AcquisitionThread(threading.Thread):
    """Captures frames in the background and pushes them to a queue."""

    def __init__(self, device, sample_rate, n_samples, out_queue):
        super().__init__(daemon=True)
        self._dev   = device
        self._rate  = sample_rate
        self._n     = n_samples
        self._q     = out_queue
        self._stop  = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):

        # Hardware path
        from waveforms_ads import acqmodeSingle, trigsrcNone, DwfStateDone
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
                while True:
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
# PSD helper
# ===========================================================================

def compute_psd(signal: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """Single-sided PSD via Welch (scipy) or Hanning periodogram (numpy fallback)."""
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
# Dashboard
# ===========================================================================

class OscilloscopeDashboard:
    """4-panel live scope: CH1 V(t), CH2 V(t), XY, PSD."""

    _MIN_DRAW_INTERVAL = 0.08   # throttle to ~12 fps max

    def __init__(self, sample_rate=100_000, n_samples=4096,
                 input_range=5.0, device_index=-1):
        self._rate  = float(sample_rate)
        self._n     = int(n_samples)
        self._range = float(input_range)

        self._ch1    = np.zeros(self._n)
        self._ch2    = np.zeros(self._n)
        self._freqs1 = np.array([1.0, 2.0])
        self._psd1   = np.array([1e-9, 1e-9])
        self._freqs2 = np.array([1.0, 2.0])
        self._psd2   = np.array([1e-9, 1e-9])

        self._last_draw   = 0.0
        self._frame_count = 0
        self._psd_skip    = 0      # compute PSD every other draw frame
        self._fps_t0      = time.time()
        self._fps         = 0.0

        self._q      = queue.Queue(maxsize=2)
        self._device = self._open_device(device_index)
        self._build_figure()
        self._acq = AcquisitionThread(
            self._device, self._rate, self._n, self._q
        )
        self._acq.start()

        self._anim = FuncAnimation(
            self._fig, self._update,
            interval=50,
            blit=False,
            cache_frame_data=False,
        )

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    def _open_device(self, index):
        if HW_AVAILABLE:
            try:
                dev = WaveFormsADS(device_index=index, auto_configure=0)
                dev.analog_in_set_range(0, self._range)
                dev.analog_in_set_range(1, self._range)
                print(f"Opened: {dev}")
                return dev
            except Exception as exc:
                print(f"Hardware unavailable ({exc}) — waiting and trying again.")
                time.sleep(5)
                _open_device(self, index)
        else:
            print("Package not available for connecting with ADS.")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def _build_figure(self):
        self._fig = plt.figure(figsize=(12, 7))
        try:
            self._fig.canvas.manager.set_window_title("WaveForms Live Scope")
        except Exception:
            pass

        # Main 2×2 grid, leaving room at bottom for buttons
        gs = gridspec.GridSpec(
            2, 2,
            figure=self._fig,
            hspace=0.38, wspace=0.28,
            left=0.07, right=0.97,
            top=0.94, bottom=0.11,
        )
        self._ax_ch1 = self._fig.add_subplot(gs[0, 0])
        self._ax_ch2 = self._fig.add_subplot(gs[0, 1])
        self._ax_xy  = self._fig.add_subplot(gs[1, 0])
        self._ax_psd = self._fig.add_subplot(gs[1, 1])

        self._time_ms = np.linspace(0, self._n / self._rate * 1e3, self._n)

        self._setup_vt_ax(self._ax_ch1, "CH1 — Voltage vs Time")
        self._setup_vt_ax(self._ax_ch2, "CH2 — Voltage vs Time")
        self._setup_xy_ax()
        self._setup_psd_ax()
        self._create_artists()
        self._add_buttons()

        # Info bar
        self._fig.text(
            0.01, 0.975,
            f"WaveForms Scope  ·  {self._rate/1e3:.0f} kHz  ·  {self._n} samp  ·  {mode_str}",
            ha="left", va="top", fontsize=7, color="#888888",
        )
        self._fps_text = self._fig.text(
            0.99, 0.975, "FPS: —",
            ha="right", va="top", fontsize=7, color="#888888",
        )

    def _setup_vt_ax(self, ax, title):
        ax.set_title(title, pad=3)
        ax.set_xlabel("Time (ms)", labelpad=1)
        ax.set_ylabel("Voltage (V)", labelpad=1)
        ax.set_xlim(self._time_ms[0], self._time_ms[-1])
        ax.set_ylim(-self._range / 2 * 1.15, self._range / 2 * 1.15)
        # Minor ticks on linear axes are fine with AutoMinorLocator
        ax.xaxis.set_minor_locator(plt.AutoLocator())
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
        # Log axes require LogLocator for minor ticks — NOT AutoMinorLocator
        ax.xaxis.set_minor_locator(
            plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20)
        )
        ax.yaxis.set_minor_locator(
            plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=20)
        )
        ax.tick_params(axis="both", which="minor", labelbottom=False, labelleft=False)
        ax.grid(True, which="minor", linewidth=0.25, color="#eeeeee")

    def _create_artists(self):
        tw = self._time_ms
        self._line_ch1, = self._ax_ch1.plot(tw, self._ch1, color=CH1_COL, lw=0.9)
        self._line_ch2, = self._ax_ch2.plot(tw, self._ch2, color=CH2_COL, lw=0.9)
        self._line_xy,  = self._ax_xy.plot(
            self._ch1, self._ch2,
            color=XY_COL, lw=0, marker=",", markersize=1, alpha=0.5,
        )
        self._line_psd1, = self._ax_psd.plot([], [], color=CH1_COL, lw=1.0, label="CH1")
        self._line_psd2, = self._ax_psd.plot([], [], color=CH2_COL, lw=1.0, label="CH2")
        self._ax_psd.legend(fontsize=7, loc="upper right", framealpha=0.8)

        # Compact stats label inside each V(t) panel
        stat_kw = dict(
            fontsize=7, va="top", ha="right",
            transform=self._ax_ch1.transAxes,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#cccccc", alpha=0.85),
        )
        self._stat_ch1 = self._ax_ch1.text(0.99, 0.98, "", **stat_kw)
        stat_kw["transform"] = self._ax_ch2.transAxes
        self._stat_ch2 = self._ax_ch2.text(0.99, 0.98, "", **stat_kw)

    def _add_buttons(self):
        ax_psd = self._fig.add_axes([0.07, 0.025, 0.20, 0.038])
        self._btn_psd = Button(ax_psd, "Export PSD (CH1+CH2)")
        self._btn_psd.label.set_fontsize(8)
        self._btn_psd.on_clicked(lambda _: self._export_psd())

        ax_vt = self._fig.add_axes([0.31, 0.025, 0.20, 0.038])
        self._btn_vt = Button(ax_vt, "Export V(t) (CH1+CH2)")
        self._btn_vt.label.set_fontsize(8)
        self._btn_vt.on_clicked(lambda _: self._export_vt())

    # ------------------------------------------------------------------
    # Animation update
    # ------------------------------------------------------------------

    def _update(self, _frame):
        now = time.time()
        if now - self._last_draw < self._MIN_DRAW_INTERVAL:
            return

        # Drain queue — always render the latest available frame
        ch1 = ch2 = None
        try:
            while True:
                ch1, ch2 = self._q.get_nowait()
        except queue.Empty:
            pass

        if ch1 is None:
            return

        self._ch1, self._ch2 = ch1, ch2
        self._last_draw = now
        self._frame_count += 1

        # V(t) — just update ydata, no artist recreation
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

        # XY
        self._line_xy.set_xdata(ch1)
        self._line_xy.set_ydata(ch2)
        lim = max(abs(ch1).max(), abs(ch2).max(), 1e-9) * 1.15
        self._ax_xy.set_xlim(-lim, lim)
        self._ax_xy.set_ylim(-lim, lim)

        # PSD — recompute every other draw frame to cut CPU load
        self._psd_skip += 1
        if self._psd_skip >= 2:
            self._psd_skip = 0
            self._freqs1, self._psd1 = compute_psd(ch1, self._rate)
            self._freqs2, self._psd2 = compute_psd(ch2, self._rate)
            # Drop DC bin (index 0) — can't display on log-x axis
            f1, p1 = self._freqs1[1:], self._psd1[1:]
            f2, p2 = self._freqs2[1:], self._psd2[1:]
            self._line_psd1.set_data(f1, p1)
            self._line_psd2.set_data(f2, p2)
            valid = np.concatenate([p1[p1 > 0], p2[p2 > 0]])
            if len(valid):
                self._ax_psd.set_ylim(valid.min() * 0.1, valid.max() * 10)

        # FPS display — update every 1.5 s
        elapsed = now - self._fps_t0
        if elapsed >= 1.5:
            self._fps    = self._frame_count / elapsed
            self._fps_t0 = now
            self._frame_count = 0
            self._fps_text.set_text(f"FPS: {self._fps:.1f}")

        self._fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _flash_title(self, ax, msg, duration=2.5):
        """Briefly replace an axis title with a save-confirmation message."""
        orig = ax.get_title()
        ax.set_title(msg, color=CONFIRM_COL)
        self._fig.canvas.draw_idle()
        def _restore():
            time.sleep(duration)
            ax.set_title(orig)
            self._fig.canvas.draw_idle()
        threading.Thread(target=_restore, daemon=True).start()

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
            self._flash_title(self._ax_psd, f"Saved: {fname}")
        except OSError as exc:
            print(f"[Export PSD] ERROR: {exc}", file=sys.stderr)

    def _export_vt(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"vt_ch1_ch2_{stamp}.csv"
        try:
            self._write_csv(
                fname, self._meta_rows(stamp),
                ["time_ms", "ch1_v", "ch2_v"],
                zip(self._time_ms, self._ch1, self._ch2),
            )
            self._flash_title(self._ax_ch1, f"Saved: {fname}")
            self._flash_title(self._ax_ch2, f"Saved: {fname}")
        except OSError as exc:
            print(f"[Export V(t)] ERROR: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def show(self):
        try:
            plt.show()
        finally:
            self._acq.stop()
            if HW_AVAILABLE:
                try:
                    self._device.close()
                except Exception:
                    pass
            print("Scope closed.")


# ===========================================================================
# Entry point
# ===========================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="WaveForms live dual-channel scope.")
    p.add_argument("--device",  type=int,   default=-1,      metavar="IDX")
    p.add_argument("--rate",    type=float, default=100_000, metavar="HZ")
    p.add_argument("--samples", type=int,   default=4096,    metavar="N")
    p.add_argument("--range",   type=float, default=5.0,     metavar="V",
                   dest="voltage_range")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(
        f"WaveForms Live Scope\n"
        f"  Rate:    {args.rate/1e3:.1f} kHz\n"
        f"  Buffer:  {args.samples} samples  ({args.samples/args.rate*1e3:.1f} ms)\n"
        f"  Range:   ±{args.voltage_range/2:.2f} V\n"
    )
    OscilloscopeDashboard(
        sample_rate=args.rate,
        n_samples=args.samples,
        input_range=args.voltage_range,
        device_index=args.device
    ).show()
