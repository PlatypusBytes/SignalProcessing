import os
import numpy as np
import matplotlib.pylab as plt
from scipy import signal
import sys
from SignalProcessing import signal_tools

WINDOWS_TYPES = ["Hann", "Barthann", "Bartlett", "Triang"]


class Window(signal_tools.Signal):
    def __init__(self, time, sig, M, window_type="Hann"):
        """
        Processes a signal with a moving window.
        Only windows with 1/2 overlap are allowed.

        @param time: Time
        @param sig: Signal to be processed
        @param M: Window length
        @param window_type: window type (default: Hanning)
        @param FS: Acquisition frequency (default False - It is computed based on time)
        """
        super(Window, self).__init__(time, sig)

        self.M = M
        if M % 2 != 0:
            sys.exit("Window length must be even")
        self.nb_cycles = int(np.ceil(len(self.signal) / M))

        # ToDo: extend with other windows: check overlap
        if window_type == "Hann":
            self.window = signal.hann(M, False)
        elif window_type == "Barthann":
            self.window = signal.barthann(M, False)
        elif window_type == "Bartlett":
            self.window = signal.bartlett(M, False)
        elif window_type == "Triang":
            self.window = signal.triang(M, False)
        else:
            sys.exit(f"Window {window_type} not defined\nWindow must be: {', '.join(WINDOWS_TYPES)}")

        # round off for end of the file. otherwise window goes over the end
        if self.nb_cycles % 2 == 0:
            self.round_off = 1
        else:
            self.round_off = 2

        # number of windows
        self.nb_windows = int((self.nb_cycles * 2) - self.round_off)

        # settings of window
        self.log.update({"Window": {"Type": window_type,
                                    "Length": M}})

        # extend signal
        if len(sig) < (self.nb_windows + self.round_off) / 2 * self.M:
            self.time = np.zeros(int((self.nb_windows + self.round_off) / 2 * self.M))
            self.signal = np.zeros(int((self.nb_windows + self.round_off) / 2 * self.M))

            self.time[:len(time)] = time
            self.time[len(time):] = np.cumsum(np.ones(int((self.nb_windows + self.round_off) / 2 * self.M) - len(time)) * np.mean(np.diff(time))) + time[-1]
            self.signal[:len(sig)] = sig

        # additional variables
        self.int_order = 0  # integration order
        self.spectrogram = []  # spectrogram
        self.spectrogram_time = []  # time for spectrogram
        return

    def fft(self, length=False):
        """
        Produces a moving FFT
        """

        # length of FFT. if not available use window length
        if not length:
            length = self.M

        # signal to be integrated
        self.spectrogram = np.zeros((length, self.nb_windows), dtype="complex128")
        self.spectrogram_time = np.zeros(self.nb_windows)
        self.spectrum = np.zeros(length, dtype="complex128")
        self.amplitude = np.zeros(length)
        self.phase = np.zeros(length)

        # considering half overlap
        for i in range(self.nb_windows):
            idx_ini = int(i * self.M / 2)
            idx_end = int(i * self.M / 2 + self.M)

            # window signal
            signal_w = self.window * self.signal[idx_ini:idx_end]

            # fft window signal
            sig = signal_tools.Signal(self.time[idx_ini:idx_end], signal_w)
            sig.fft(nb_points=length, window=self.window)

            # spectrogram
            self.spectrogram[:, i] = sig.spectrum
            self.spectrogram_time[i] = (self.time[idx_ini] + self.time[idx_end-1]) / 2

        self.amplitude = np.abs(np.mean(self.spectrogram, axis=1))
        self.phase = np.abs(np.mean(np.unwrap(np.angle(self.spectrogram)), axis=1))

        # add to results
        self.frequency = sig.frequency
        self.log["FFT"] = True
        return

    def integration(self, rule="trap", baseline=False, moving=False, hp=False, ini_cond=False, fpass=0.5, n=6):
        """
        Numerical integration of signal with moving window

        Parameters
        ----------
        :param rule: integration rule  (optional: default trap)
        :param baseline: base line correction (optional: default False)
        "param moving: moving average correction (optional: default False)
        :param hp: highpass filter correction at fpass (optional: default False)
        :param ini_cond: initial conditions. (optional: default 0)
        :param fpass: cut off frequency [Hz]. only used if hp=True (optional: default 0.5)
        :param n: order of the filter. only used if hp=True (optional: default 6)
        """

        # signal to be integrated
        new_signal = np.zeros(len(self.signal))

        # considering half overlap
        for i in range(self.nb_windows):
            idx_ini = int(i * self.M / 2)
            idx_end = int(i * self.M / 2 + self.M)

            # window signal
            signal_w = self.window * self.signal[idx_ini: idx_end]

            # fft window signal
            sig = signal_tools.Signal(self.time[idx_ini: idx_end], signal_w, FS=self.Fs)
            sig.integrate(rule=rule, baseline=baseline, moving=moving, hp=hp, ini_cond=ini_cond, fpass=fpass, n=n)

            # add to result
            new_signal[idx_ini: idx_end] += sig.signal

        # update signal
        self.signal = new_signal

        # update log
        self.log["Integration"] = sig.log["Integration"]
        self.int_order += 1
        self.log["Integration"]["Order"] = self.int_order

        return

    def filter(self, Fpass, N, type="lowpass", rp=0.01, rs=60):
        """
        Filter signal with window

        Parameters
        ----------
        :param Fpass: cut off frequency [Hz]
        :param N: order of the filter
        :param type: type of the filter (optional: default lowpass)
        :param rp: maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number
                   default is 0.01
        :param rs: minimum attenuation required in the stop band. Specified in decibels, as a positive number
                   default is 60
        """

        # signal to be filtered
        new_signal = np.zeros(len(self.signal))

        # considering half overlap
        for i in range(self.nb_windows):
            idx_ini = int(i * self.M / 2)
            idx_end = int(i * self.M / 2 + self.M)

            # window signal
            signal_w = self.window * self.signal[idx_ini: idx_end]

            # fft window signal
            sig = signal_tools.Signal(self.time[idx_ini: idx_end], signal_w, FS=self.Fs)
            sig.filter(Fpass, N, type=type, rp=rp, rs=rs)

            # add to result
            new_signal[idx_ini: idx_end] += sig.signal

        # update signal
        self.signal = new_signal

        # update log
        self.log["Filter"] = sig.log["Filter"]
        return

    def plot_spectrogram(self, output_folder="./"):
        """
        Creates spectrogram plot

        :param output_folder: output folder to save figure (optional: default './')
        """
        # cannot compute without FFT
        if not self.log["FFT"]:
            return print(f"FFT needs to be run before creating spectrogram.\nNo figure created!")

        # if output folder does not exits -> creates
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # create fig
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.pcolormesh(self.spectrogram_time, self.frequency, np.abs(self.spectrogram),
                      cmap='Greys', shading='auto')

        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.grid()
        plt.savefig(os.path.join(output_folder, "spectrogram.png"))
        plt.savefig(os.path.join(output_folder, "spectrogram.pdf"))

        plt.close()

        return
