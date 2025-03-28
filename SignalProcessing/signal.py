from typing import Union
import sys
import numpy as np
import numpy.typing as npt
from scipy import integrate, signal
from enum import Enum

# octave bands
BANDS = {"one-third": [[.08, .10],
                       [.10, .126],
                       [.126, .16],
                       [.16, .20],
                       [.20, .253],
                       [.253, .32],
                       [.32, .40],
                       [.40, .50],
                       [.50, .63]],
         }

class IntegrationRules(Enum):
    """
    Integration rules
    """
    TRAPEZOID = 1
    SIMPSON = 2

class Windows(Enum):
    """
    Windows types

    The values are the same as in `scipy.signal`. This is used for the PSD.
    """

    HANN = 'hann'
    HAMMING = 'hamming'
    BLACKMAN = 'blackman'
    RECTANGULAR = 'boxcar'
    TRIANG = 'triang'

class SignalProcessing:
    """
    Signal processing class
    """
    def __init__(self,
                 time: npt.NDArray[np.float64],
                 signal: npt.NDArray[np.float64],
                 FS: Union[None, int] = None,
                 window: Union[None, Windows] = None,
                 window_size: int = 0):
        """
        Signal processing

        Parameters
        ----------
        :param time: Time vector
        :param signal: Signal vector
        :param FS: Acquisition frequency (optional: default None. FS is computed based on time)
        :param window: Type of window to use (optional: default None - uses rectangular window for entire signal)
        :param window_size: Size of the window (optional: default 0 - uses signal length when window is None)
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
        self.v_eff = None

        # acquisition frequency
        if not FS:
            FS = int(np.ceil(1 / np.mean(np.diff(time))))
        self.Fs = FS

        # windowing
        signal_length = len(self.signal)

        # When window is None, process entire signal with rectangular window
        if window is None:
            self.window = np.ones(signal_length)
            self.window_size = signal_length
            self.window_type = Windows.RECTANGULAR
            self.nb_windows = 1
            self.use_window = False
        else:
            if window_size == 0:
                raise ValueError("When using a window the `window_size` must be specified")
            if window_size % 2 != 0:
                raise ValueError("Window length must be even")
            if window not in Windows:
                raise ValueError(f"Window type {window} not supported. Available types: {list(Windows)}")
            if window_size > signal_length:
                raise ValueError(f"Window length ({window_size}) cannot be greater than signal length ({signal_length}).")

            self.window = self.__create_window(window, window_size)
            self.window_size = window_size
            self.window_type = window
            self.nb_windows = int(np.ceil((signal_length / window_size) * 2 - 2))
            self.use_window = True

            # pad signal at the end if necessary to get full windows
            if signal_length % window_size != 0:
                self.signal = np.append(self.signal, np.zeros(window_size - (signal_length % window_size)))
                self.time = np.append(self.time, np.zeros(window_size - (signal_length % window_size)))

    @staticmethod
    def __create_window(window_type: Windows, size: int) -> npt.NDArray[np.float64]:
        """
        Create a window array of specified type and size

        Parameters
        ----------
        :param window_type: Type of window from Windows enum
        :param size: Size of the window
        :return: Window array of specified size
        """
        if window_type == Windows.RECTANGULAR:
            return np.ones(size)
        elif window_type == Windows.HANN:
            return np.hanning(size)
        elif window_type == Windows.HAMMING:
            return np.hamming(size)
        elif window_type == Windows.BLACKMAN:
            return np.blackman(size)
        elif window_type == Windows.TRIANG:
            return signal.triang(size)
        else:
            raise ValueError(f"Window type {window_type} not supported"
                             f"Available types: {list(Windows)}")

    def fft(self,
            nb_points: Union[None, int] = None,
            half_representation: bool = True):
        """
        FFT of signal

        Parameters
        ----------
        :param nb_points: number of points for FFT (optional: default None)
        :param half_representation: true if fft should be computed in half representation (optional: default True)
        """

        # if window is used, set nfft to window size
        if self.use_window:
            nfft = self.window_size
            sig = self.signal
        else:
            # if nb_points is None: nb_points is signal length
            if nb_points is None:
                nfft = len(self.signal)
                sig = self.signal
            else:
                # if length is even
                if nb_points % 2 == 0:
                    nfft = nb_points
                    sig = self.signal
                else:
                    nfft = nb_points + 1
                    sig = np.append(self.signal, 0.)
                    nfft = nb_points

        # Normalize by the sum of the window samples.
        # This compensates for the energy reduction caused by non-rectangular windows, aiming to preserve the
        # peak amplitude of stationary sinusoids.
        normalise_fct = np.sum(self.window)

        spectrum_w = np.zeros((nfft, self.nb_windows), dtype="complex128")
        hop_size = int(self.window_size * (1 - 0.5))

        # for each window
        for w in range(self.nb_windows):

            idx_ini = w * hop_size
            idx_end = idx_ini + self.window_size

            # window signal
            signal_w = self.window * sig[idx_ini:idx_end]

            # fft window signal
            spectrum_w[:, w] = np.fft.fft(signal_w, nfft) / normalise_fct


        self.amplitude = np.mean(np.abs(spectrum_w), axis=1)
        self.phase = np.unwrap(np.angle(np.mean(spectrum_w, axis=1)))

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

        # Applies twice the filter to the signal to avoid phase distortion
        self.signal = signal.sosfiltfilt(sos, self.signal)


    def psd(self):
        """
        PSD of signal
        """

        # check if window is initialized
        if not self.use_window:
            raise ValueError("No window defined. Please define a window when initialising SignalProcessing.")

        # compute PSD using Welch method
        self.frequency_Pxx, self.Pxx = signal.welch(self.signal, fs=self.Fs, nperseg=self.window_size,
                                                    window=self.window_type.value, scaling='density')


    def v_eff_SBR(self, n: int = 4, tau: int = 0.125):
        """
        Compute v_eff of signal based on SBR deel B Hinder voor personen in gebouwen (2006)

        Parameters
        ----------
        :param n:(optional, default = 4) number of time constants
        :param tau: (optional, default = 0.125) time constant for the exponential decay
        """

        # Create exponential decay function `g` for running RMS calculation
        fout = 1 / (1 - np.exp(-n))
        qsi = np.linspace(0, n * tau, int(n * tau * self.Fs + 1))
        g = fout * np.exp(-qsi / tau)


        # Frequency weighting parameters
        v0 = 1 / 1000  # Reference velocity [m/s]
        f0 = 5.6       # Reference frequency [Hz]

        # Handle even/odd signal length for FFT
        if self.signal.shape[0]  % 2 != 0:
            nv1 = int(self.signal.shape[0] / 2 + 0.5)
            nv2 = int(self.signal.shape[0] / 2 - 0.5)
        else:
            nv1 = int(self.signal.shape[0] / 2)
            nv2 = int(self.signal.shape[0] / 2)

        # Calculate frequency resolution
        df = 1 / (1 / self.Fs * self.signal.shape[0])
        freq = np.arange(df, (nv1 + 1) * df, df)

        # Create high-pass weighting filter (human perception curve)
        Hv = (1 / v0) * 1 / (np.sqrt(1 + (f0 / freq) ** 2))
        Hv = np.append(0, Hv)  # Add DC component

        # Create low-pass filter with 50 Hz cutoff
        cut_off_number = int(np.ceil(50 / df))
        if cut_off_number < nv1:
            Hv2 = np.zeros(Hv.shape[0])
            Hv2[:cut_off_number+1] = 1
        else:
            Hv2 = np.ones(Hv.shape[0])

        # Applies the frequency weighting functions
        Fv = np.fft.fft(self.signal)
        Fhv = Hv2 * Hv * Fv[:nv1+1]
        Fv = np.append(Fhv, np.flipud(np.conj(Fhv[1:nv2])))
        v_eff = np.real(np.fft.ifft(Fv))

        # moving root-mean-square through convolution with the exponential decay function `g`
        v_eff = np.sqrt( np.convolve(v_eff**2, g) * (1 / self.Fs) /tau)

        self.v_eff = v_eff[:self.signal.shape[0]]