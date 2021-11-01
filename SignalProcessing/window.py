import numpy as np
from scipy import signal
import sys
from . import signal_tools


class Window:
    def __init__(self, time, sig, M, window_type="Hanning", FS=False):
        """
        Processes a signal with a moving window.
        Only windows with 1/2 overlap are allowed.

        @param time: Time
        @param sig: Signal to be processed
        @param M: Window length
        @param window_type: window type (default: Hanning)
        @param FS: Acquisition frequency (default False - It is computed based on time)
        """
        self.time = time
        self.signal = sig
        self.M = M
        self.nb_cycles = int(np.ceil(len(self.signal) / M))

        # ToDo: extend with other windows: check overlap
        if window_type == "Hanning":
            self.window = signal.hann(M, False)
        else:
            sys.exit(f"Window {window_type} not defined")

        # round off for end of the file. otherwise window goes over the end
        if self.nb_cycles % 2 == 0:
            self.round_off = 1
        else:
            self.round_off = 2

        # number of windows
        self.nb_windows = int((self.nb_cycles * 2 - 1) - self.round_off)

        # results
        self.frequency = []
        self.signal_fft = np.zeros((M, self.nb_windows))
        self.signal_int = np.zeros(len(self.signal))

        # acquisition frequency
        if not FS:
            FS = int(np.ceil(1 / np.mean(np.diff(time))))
        self.Fs = FS

        return

    def fft(self):
        """
        Produces a moving FFT
        """
        # considering half overlap
        for i in range(self.nb_windows):
            idx_i = i * self.M / 2
            # window signal
            signal_w = self.window * self.signal[int(idx_i): int(idx_i + self.M)]

            # fft window signal
            sig = signal_tools.Signal(self.time[int(idx_i): int(idx_i + self.M)], signal_w)
            sig.fft(window=self.window)

            # add to result
            self.signal_fft[:, i] = sig.amplitude

        # add to results
        self.frequency = sig.frequency
        return

    def integration(self):
        # considering half overlap
        for i in range(self.nb_windows):
            idx_i = i * self.M / 2
            # window signal
            signal_w = self.window * self.signal[int(idx_i): int(idx_i + self.M)]

            # fft window signal
            sig = signal_tools.Signal(self.time[int(idx_i): int(idx_i + self.M)], signal_w, FS=self.Fs)
            sig.integrate()

            # add to result
            self.signal_int[int(idx_i):int(idx_i + self.M)] += sig.signal
        return

    def filter(self):
        return
