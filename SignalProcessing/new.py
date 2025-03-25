from typing import Union
import sys
import numpy as np
import numpy.typing as npt
from scipy import integrate, signal
from enum import Enum

class IntegrationRules(Enum):
    """
    Integration rules
    """
    TRAPEZOID = 1
    SIMPSON = 2


class SignalProcessing:

    def __init__(self, time: npt.NDArray[np.float64], signal: npt.NDArray[np.float64], FS: Union[None, int] = None):
        """
        Signal processing

        Parameters
        ----------
        :param time: Time vector
        :param signal: Signal vector
        :param FS: Acquisition frequency (optional: default None. FS is computed based on time)
        """
        self.time = time
        self.signal = signal
        self.signal_org = signal
        self.frequency = None
        self.amplitude = None
        self.phase = None
        self.spectrum = None
        self.Pxx = None
        self.frequency_Pxx = None
        self.signal_inv = None
        self.time_inv = None
        # acquisition frequency
        if not FS:
            FS = int(np.ceil(1 / np.mean(np.diff(time))))
        self.Fs = FS



    def fft(self, nb_points: Union[None, int] = None, window: str = "rectangular", half_representation: bool = True):
        """
        FFT of signal

        Parameters
        ----------
        :param nb_points: number of points for FFT (optional: default None)
        :param window: type of window (optional: default 'rectangular') otherwise it must be a np.ndarray
        :param half_representation: true if fft should be computed in half representation (optional: default False)
        """

         # if nb_points is empty: nb_points is signal length
        if not nb_points:
            nb_points = self.signal.shape[0]

        # if lenght is even
        if nb_points % 2 == 0:
            nfft = nb_points
            sig = self.signal
        else:
            nfft = nb_points + 1
            sig = np.append(self.signal, 0.)

        # type of windows
        if window == "rectangular":
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

    def inv_fft(self):
        """
        Inverse FFT of signal
        """

        if self.amplitude is None or self.phase is None:
            sys.exit("No FFT computed. Please compute FFT first.")

        # compute spectrum
        spectrum_inv = np.fft.ifft(self.amplitude * np.exp(1j * self.phase), len(self.amplitude))
        # inverse of the FFT signal
        self.signal_inv = np.real(spectrum_inv) * len(self.amplitude)
        # time from frequency
        self.time_inv = np.cumsum(np.ones(len(self.amplitude)) * 1 / self.Fs) - 1 / self.Fs

    def integrate(self, rule: IntegrationRules = IntegrationRules.TRAPEZOID,
                  baseline: bool = False, moving: bool = False, hp: bool = False, ini_cond: float = 0.,
                  fpass: float = 0.5, n: int = 6):
        """
        Numerical integration of signal

        Parameters
        ----------
        :param rule: integration rule (optional: default TRAPEZOID)
        :param baseline: base line correction (optional: default False)
        :param moving: moving average correction (optional: default False)
        :param hp: highpass filter correction at fpass (optional: default False)
        :param ini_cond: initial conditions. (optional: default 0.0)
        :param fpass: cut off frequency [Hz]. only used if hp=True (optional: default 0.5)
        :param n: order of the filter. only used if hp=True (optional: default 6)
        """
        # mean average correction
        if moving:
            self.signal = self.signal - np.mean(self.signal)
            self.log["Moving"] = True

        # integration rule
        if rule == IntegrationRules.TRAPEZOID:
            self.signal = integrate.cumulative_trapezoid(self.signal, self.time, initial=ini_cond)
        elif rule == IntegrationRules.SIMPSON:
            self.signal = integrate.cumulative_simpson(self.signal, x=self.time, initial=ini_cond)
        else:
            sys.exit("Integration rule not supported")

        # baseline correction
        if baseline:
            fit = np.polyfit(self.time, self.signal, 2)
            fit_int = np.polyval(fit, self.time)
            self.signal = self.signal - fit_int

        # high pass filter
        if hp:
            self.filter(fpass, n, type_filter="highpass")


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
        # types allowed
        types = ["lowpass", "highpass"]

        # check if filter type is supported
        if type_filter not in types:
            sys.exit(f"ERROR: Type filter '{type_filter}' not available\n"
                     "Filter type must be in {types}")

        z, p, k = signal.ellip(N, rp, rs, Fpass / (self.Fs / 2), btype=type_filter, output='zpk')
        sos = signal.zpk2sos(z, p, k)

        self.signal = signal.sosfilt(sos, self.signal)
        self.signal = self.signal[::-1]
        self.signal = signal.sosfilt(sos, self.signal)
        self.signal = self.signal[::-1]

    def psd(self, length_w: int = 128):
        """
        PSD of signal

        Parameters
        ----------
        :param length_w: lenght of the window for PSD (optional: default 128)
        """
        if length_w > len(self.signal):
            raise ValueError(f"Window length ({length_w}) cannot be greater than signal length ({len(self.signal)}).")

        # compute PSD using Welch method
        self.frequency_Pxx, self.Pxx = signal.welch(self.signal, fs=self.Fs, nperseg=length_w,
                                                    window='hamming', scaling="spectrum")

