import sys
import numpy as np
from scipy import integrate, signal


class Signal:
    def __init__(self, time, sig, FS=False):
        """
        Signal processing object

        Parameters
        ----------
        :param time: Time vector
        :param sig: Signal vector
        :param FS: Acquisition frequency (optional: default False. FS is computed based on time)
        """
        self.signal_org = sig  # signal original
        self.signal = sig  # signal to be processed
        self.time = time  # time

        # parameters FFT
        self.spectrum = []
        self.amplitude = []
        self.phase = []
        self.frequency = []

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
                    "IFFT": False,
                    "Integration": {"Integration": False,
                                    "Order": 0,
                                    "Baseline": False,
                                    "Moving": False,
                                    "High-pass": False,
                                    },
                    "Filter": False}

        return

    def __str__(self):
        """
        Defines str object to print the log
        """
        return f"LOG description\n{self.log}"

    def fft(self, nb_points=False, window="rectangular"):
        """
        FFT of signal

        Parameters
        ----------
        :param nb_points: number of points for FFT (optional default False)
        :param window: type of window (optional: default 'rectangular') otherwise it must be a np.ndarray
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

        # compute spectrum
        self.spectrum_inv = np.fft.ifft(self.amplitude * np.exp(1j * self.phase), len(self.amplitude))
        # inverse of the FFT signal
        self.signal_inv = np.real(self.spectrum_inv) * len(self.amplitude)
        # time from frequency
        self.time_inv = np.cumsum(np.ones(len(self.amplitude)) * 1 / self.Fs) - 1 / self.Fs
        # log
        self.log["IFFT"] = True
        return

    def integrate(self, rule="trap", baseline=False, moving=False, hp=False, ini_cond=False, fpass=0.5, n=6):
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
            self.filter(fpass, n, type="highpass")
            self.log["High-pass"] = True

        # log
        self.log["Integration"]["Integration"] = True
        self.log["Integration"]["Order"] += 1
        return

    def filter(self, Fpass, N, type="lowpass", rp=0.01, rs=60):
        """
        Filter signal

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
        types = ["lowpass", "highpass"]

        # check if filter type is supported
        if type not in types:
            sys.exit(f"ERROR: Type filter '{type}' not available")

        z, p, k = signal.ellip(N, rp, rs, Fpass / (self.Fs / 2), btype=type, output='zpk')
        sos = signal.zpk2sos(z, p, k)

        self.signal = signal.sosfilt(sos, self.signal)
        self.signal = self.signal[::-1]
        self.signal = signal.sosfilt(sos, self.signal)
        self.signal = self.signal[::-1]

        return
