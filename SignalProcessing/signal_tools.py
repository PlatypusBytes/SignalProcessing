import sys
import numpy as np
from scipy import integrate, signal


class Signal:
    def __init__(self, time: np.ndarray, sig: np.ndarray, FS: [bool, int] = False) -> None:
        """
        Signal processing object

        Parameters
        ----------
        :param time: Time vector
        :param sig: Signal vector
        :param FS: Acquisition frequency (optional: default False. FS is computed based on time)
        """
        self.signal = sig  # signal to be processed
        self.signal_org = np.copy(sig)  # signal original
        self.time = time  # time

        # parameters FFT
        self.spectrum = []
        self.amplitude = []
        self.phase = []
        self.frequency = []

        # parameters PSD
        self.Pxx = []
        self.frequency_Pxx = []

        # parameters inverted FFT
        self.spectrum_inv = []
        self.signal_inv = []
        self.time_inv = []

        # acquisition frequency
        if not FS:
            FS = int(np.ceil(1 / np.mean(np.diff(time))))
        self.Fs = FS

        # log properties
        self.log = {"FFT": False,
                    "Half_FFT": False,
                    "IFFT": False,
                    "PSD": False,
                    "Integration": False,
                    "Filter": False}

        return

    def __str__(self):
        """
        Defines str object to print the log
        """
        return f"LOG description\n{self.log}"

    def fft(self, nb_points: [bool, int] = False, window: str = "rectangular", half_representation: bool = False) \
            -> None:
        """
        FFT of signal

        Parameters
        ----------
        :param nb_points: number of points for FFT (optional default False)
        :param window: type of window (optional: default 'rectangular') otherwise it must be a np.ndarray
        :param half_representation: true if fft should be computed in half representation (optional: default False)
        :return:
        """
        # check if number of points exits
        if not nb_points:
            # if nb_points is empty: nb_points is signal length
            nb_points = self.signal.shape[0]

        # if number is even
        if nb_points % 2 == 0:
            nfft = nb_points
            sig = self.signal
        else:
            nfft = nb_points + 1
            sig = np.append(self.signal, 0.)

        # type of windows
        if (isinstance(window, str)) and (window == "rectangular"):
            normalise_fct = len(sig[sig != 0.])
        elif isinstance(window, np.ndarray):
            normalise_fct = np.sum(window)
        else:
            sys.exit(f"Window {window} not valid for FFT")

        # compute spectrum
        self.spectrum = np.fft.fft(sig, nfft) / normalise_fct
        # compute amplitude
        self.amplitude = np.abs(self.spectrum)
        # compute phase
        self.phase = np.unwrap(np.angle(self.spectrum))
        # compute frequency
        self.frequency = np.linspace(0, 1, nfft) * self.Fs

        # half representation
        if half_representation:
            self.frequency = self.frequency[:int(nfft / 2)]
            self.amplitude = 2 * self.amplitude[:int(nfft / 2)]
            self.log["Half_FFT"] = True
        else:
            self.log["Half_FFT"] = False
        # log
        self.log["FFT"] = True
        return

    def inv_fft(self):
        """
        Inverse FFT of signal
        """

        # check if FFT is available. if not returns message
        if not self.log["FFT"]:
            return print("FFT needs to be available before inv FFT can be computed")
        if self.log["Half_FFT"]:
            return print("FFT needs to be available not in half representation")

        # compute spectrum
        self.spectrum_inv = np.fft.ifft(self.amplitude * np.exp(1j * self.phase), len(self.amplitude))
        # inverse of the FFT signal
        self.signal_inv = np.real(self.spectrum_inv) * len(self.amplitude)
        # time from frequency
        self.time_inv = np.cumsum(np.ones(len(self.amplitude)) * 1 / self.Fs) - 1 / self.Fs
        # log
        self.log["IFFT"] = True
        return

    def integrate(self, rule: str = "trap",
                    baseline: bool = False, moving: bool = False, hp: bool = False, ini_cond: bool = False,
                    fpass: float = 0.5, n: int = 6) -> None:
        """
        Numerical integration of signal

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
        # if log is False create dict for integration
        if not self.log["Integration"]:
            self.log["Integration"] = {"Integration": False,
                                       "Order": 0,
                                       "Baseline": False,
                                       "Moving": False,
                                       "High-pass": False}

        # rules allowed
        rules = ["trap"]
        # check if integration rule is supported
        if rule not in rules:
            sys.exit(f"ERROR: Integration rule '{rule}' not available")

        # mean average correction
        if moving:
            self.signal = self.signal - np.mean(self.signal)
            self.log["Moving"] = True

        # integration rule
        if rule == "trap":  # trapezoidal rule
            self.signal = integrate.cumtrapz(self.signal, self.time, initial=ini_cond)

        # baseline correction
        if baseline:
            fit = np.polyfit(self.time, self.signal, 2)
            fit_int = np.polyval(fit, self.time)
            self.signal = self.signal - fit_int
            self.log["Baseline"] = True

        # high pass filter
        if hp:
            self.filter(fpass, n, type_filter="highpass")
            self.log["High-pass"] = True

        # log
        self.log["Integration"]["Integration"] = True
        self.log["Integration"]["Order"] += 1
        return

    def filter(self, Fpass: float, N: int, type_filter: str = "lowpass", rp: float = 0.01, rs: int = 60):
        """
        Filter signal

        Parameters
        ----------
        :param Fpass: cut off frequency [Hz]
        :param N: order of the filter
        :param type_filter: type of the filter (optional: default lowpass)
        :param rp: maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number
                   default is 0.01
        :param rs: minimum attenuation required in the stop band. Specified in decibels, as a positive number
                   default is 60
        """

        # if log is False create dict for filter
        if not self.log["Filter"]:
            self.log["Filter"] = {"Type": [],
                                  "Cut-off": []
                                  }

        # types allowed
        types = ["lowpass", "highpass"]

        # check if filter type is supported
        if type_filter not in types:
            sys.exit(f"ERROR: Type filter '{type_filter}' not available")

        z, p, k = signal.ellip(N, rp, rs, Fpass / (self.Fs / 2), btype=type_filter, output='zpk')
        sos = signal.zpk2sos(z, p, k)

        self.signal = signal.sosfilt(sos, self.signal)
        self.signal = self.signal[::-1]
        self.signal = signal.sosfilt(sos, self.signal)
        self.signal = self.signal[::-1]

        # add to log
        self.log["Filter"]["Type"].append(type_filter)
        self.log["Filter"]["Cut-off"].append(Fpass)

        return

    def psd(self, length_w: int = 128) -> None:
        """
        PSD of signal

        Parameters
        ----------
        :param length_w: lenght of the window
        """
        # compute PSD using Welch method
        self.frequency_Pxx, self.Pxx = signal.welch(self.signal, fs=self.Fs, nperseg=length_w)

        # update log
        self.log["PSD"] = True
        return

