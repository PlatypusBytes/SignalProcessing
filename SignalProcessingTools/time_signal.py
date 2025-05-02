from typing import Optional
import sys
import numpy as np
import numpy.typing as npt
from scipy import integrate, signal
from enum import Enum


class FilterDesign(Enum):
    """
    Filter design types
    """
    BUTTERWORTH = 1
    CHEBYSHEV = 2
    ELLIPTIC = 3

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

class TimeSignalProcessing:
    """
    Signal processing class for time signals
    """
    def __init__(self,
                 time: npt.NDArray[np.float64],
                 signal: npt.NDArray[np.float64],
                 Fs: Optional[int] = None,
                 window: Optional[Windows] = None,
                 window_size: int = 0):
        """
        Signal processing

        Parameters
        ----------
        :param time (npt.NDArray[np.float64]): Time vector
        :param signal (npt.NDArray[np.float64]): Signal vector
        :param Fs (Optional[int]): Acquisition frequency (optional: default None. Fs is computed based on time)
        :param window (Optional[Windows]): Type of window to use (optional: default None - uses rectangular
         window for entire signal)
        :param window_size (int): Size of the window (optional: default 0 - uses signal length when window is None)
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
        self.Sxx = None
        self.frequency_Sxx = None
        self.time_Sxx = None
        self.fft_settings = {"nb_points": None,
                             "half_representation": False}
        # Track operations performed on the signal
        self.operations = []

        # acquisition frequency
        if not Fs:
            Fs = int(np.ceil(1 / np.mean(np.diff(time))))
        self.Fs = Fs

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
                self.operations.append(f"Signal padded with zeros (original length: {signal_length}, new length: {len(self.signal)})")

    def __str__(self) -> str:
        """
        String representation of the SignalProcessing object
        showing the current state and operations performed.

        Returns
        -------
        str: A formatted string with information about the signal processing instance
        """
        # Create basic signal info
        info = [
            f"SignalProcessing Object",
            f"------------------------",
            f"Signal length: {len(self.signal)} samples",
            f"Sampling frequency: {self.Fs} Hz",
            f"Signal duration: {self.time[-1]:.3f} seconds",
        ]

        # Window information
        info.append(f"Window type: {self.window_type.name}")
        info.append(f"Window size: {self.window_size} samples")
        if self.use_window:
            info.append(f"Number of windows: {self.nb_windows}")

        # Show operations that have been performed
        if self.operations:
            info.append("\nOperations performed:")
            for i, op in enumerate(self.operations):
                info.append(f"- {i + 1}. {op}")
        else:
            info.append("\nNo operations performed yet")

        return "\n".join(info)

    @staticmethod
    def __create_window(window_type: Windows, size: int) -> npt.NDArray[np.float64]:
        """
        Create a window array of specified type and size

        Parameters
        ----------
        :param window_type (Windows): Type of window from Windows enum
        :param size (int): Size of the window
        :return (npt.NDArray[np.float64]): Window array of specified size
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
            nb_points: Optional[int] = None,
            half_representation: bool = True):
        """
        FFT of signal
        If window is used, the FFT is computed for each window and averaged.

        Parameters
        ----------
        :param nb_points (Optional[int]): number of points for FFT (optional: default None)
        :param half_representation (bool): true if fft should be computed in half representation
         (optional: default True)
        """

        # if window is used, set nfft to window size
        if self.use_window:
            nfft = self.window_size
            sig = self.signal
            odd_length = False
            normalise_fct = np.sum(self.window)
        else:
            # if nb_points is None: nb_points is signal length
            if nb_points is None:
                nfft = len(self.signal)
                sig = self.signal
                odd_length = False
                normalise_fct = np.sum(self.window)
            else:
                nfft = nb_points
                sig = np.zeros(nfft)
                sig[:len(self.signal)] = self.signal
                odd_length = False
                self.window = np.ones(nfft)
                self.window_size = nfft
                normalise_fct = len(self.signal)

            # if length is even
            if nfft % 2 != 0:
                nfft = len(self.signal) + 1
                sig = np.append(self.signal, 0.)
                self.window = np.ones(nfft)
                self.window_size = nfft
                odd_length = True

        spectrum_w = np.zeros((nfft, self.nb_windows), dtype="complex128")
        hop_size = int(self.window_size * (1 - 0.5))

        # for each window
        for w in range(self.nb_windows):

            idx_ini = w * hop_size
            idx_end = idx_ini + self.window_size

            # window signal
            signal_w = self.window * sig[idx_ini:idx_end]

            # fft window signal
            # Normalize by the sum of the window samples.
            # This compensates for the energy reduction caused by non-rectangular windows, aiming to preserve the
            # peak amplitude of stationary sinusoids.
            spectrum_w[:, w] = np.fft.fft(signal_w, nfft) / normalise_fct


        self.amplitude = np.mean(np.abs(spectrum_w), axis=1)
        # self.phase = np.unwrap(np.angle(np.mean(spectrum_w, axis=1)))
        self.phase = np.angle(np.mean(np.exp(1j * np.angle(spectrum_w)), axis=1))

        # compute frequency
        self.frequency = np.linspace(0, 1, nfft) * self.Fs

        # half representation
        if half_representation:
            self.frequency = self.frequency[:int(nfft / 2)]
            self.amplitude = 2 * self.amplitude[:int(nfft / 2)]
            self.phase = self.phase[:int(nfft / 2)]

        # FFT settings: needed to perform inverse FFT
        self.fft_settings = {"nb_points": nfft,
                             "half_representation": half_representation,
                             "odd_length": odd_length}

        # Add to operations list
        op_info = f"FFT (points: {nfft}, half representation: {half_representation})"
        self.operations.append(op_info)

    def inv_fft(self):
        """
        Inverse FFT of signal

        If the signal was processed with a window during FFT,
        the inverse FFT will also use the same windowed approach
        with proper overlap-add reconstruction.
        """
        # check if FFT was computed

        if self.amplitude is None or self.phase is None:
            raise ValueError("No FFT computed. Please compute FFT first.")

        if self.fft_settings["half_representation"]:
            raise  NotImplementedError("Half representation not supported for inverse FFT. " \
            "Please compute FFT with full representation.")

        if self.use_window:
            raise ValueError("Cannot perform inverse FFT on the windowed signal.")

        # get FFT settings
        odd_length = self.fft_settings["odd_length"]

        # get FFT
        amplitude = self.amplitude
        phase = self.phase

        # compute spectrum from amplitude and phase
        spectrum = amplitude * np.exp(1j * phase)
        spectrum_inv = np.fft.ifft(spectrum, len(spectrum))
        # inverse of the FFT signal
        self.signal_inv = np.real(spectrum_inv) * len(spectrum)
        # time from frequency
        self.time_inv = np.cumsum(np.ones(len(spectrum)) * 1 / self.Fs) - 1 / self.Fs

        if odd_length:
            # remove last sample
            self.signal_inv = self.signal_inv[:-1]
            self.time_inv = self.time_inv[:-1]

        # Add to operations list
        self.operations.append("Inverse FFT" + (" with windowing" if self.use_window else ""))

    def integrate(self, rule: IntegrationRules = IntegrationRules.TRAPEZOID,
                  baseline: bool = False, moving: bool = False, hp: bool = False, ini_cond: float = 0.,
                  fpass: float = 0.5, n: int = 6):
        """
        Numerical integration of signal

        Parameters
        ----------
        :param rule (IntegrationRules): integration rule (optional: default TRAPEZOID)
        :param baseline (bool): base line correction (optional: default False)
        :param moving (bool): moving average correction (optional: default False)
        :param hp (bool): highpass filter correction at fpass (optional: default False)
        :param ini_cond (float): initial conditions. (optional: default 0.0)
        :param fpass (float): cut off frequency [Hz]. only used if hp=True (optional: default 0.5)
        :param n (int): order of the filter. only used if hp=True (optional: default 6)
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

        # Add to operations list
        op_details = []
        op_details.append(f"rule: {rule.name}")
        if baseline:
            op_details.append("baseline correction")
        if moving:
            op_details.append("moving average correction")
        if hp:
            op_details.append(f"highpass filter (cutoff: {fpass} Hz, order: {n})")

        self.operations.append(f"Integration ({', '.join(op_details)})")


    def filter(self, Fpass: float, N: int, filter_design: FilterDesign = FilterDesign.ELLIPTIC,
               type_filter: str = "lowpass", rp: float = 0.01, rs: int = 60):
        """
        Filter signal

        Parameters
        ----------
        :param Fpass (float): cut off frequency [Hz]
        :param N (int): order of the filter
        :param filter_design (FilterDesign): filter design (optional: default ELLIPTIC)
        :param type_filter (str): type of the filter (optional: default lowpass)
        :param rp (float): maximum ripple allowed below unity gain in the passband. Specified in decibels, as a
         positive number. (optional: default 0.01)
        :param rs (int): minimum attenuation required in the stop band. Specified in decibels, as a positive number
         (optional: default 60)
        """
        # types allowed
        types = ["lowpass", "highpass", "bandpass"]

        # check if filter type is supported
        if type_filter not in types:
            sys.exit(f"ERROR: Type filter '{type_filter}' not available\n"
                     "Filter type must be in {types}")

        # design filter
        if filter_design == FilterDesign.ELLIPTIC:
            z, p, k = signal.ellip(N, rp, rs, np.array(Fpass) / (self.Fs / 2), btype=type_filter, output='zpk')
        elif filter_design == FilterDesign.BUTTERWORTH:
            z, p, k = signal.butter(N, np.array(Fpass) / (self.Fs / 2), btype=type_filter, output='zpk')
        elif filter_design == FilterDesign.CHEBYSHEV:
            z, p, k = signal.cheby1(N, rp, np.array(Fpass) / (self.Fs / 2), btype=type_filter, output='zpk')

        sos = signal.zpk2sos(z, p, k)

        # Applies twice the filter to the signal to avoid phase distortion
        self.signal = signal.sosfiltfilt(sos, self.signal)

        # Add to operations list
        self.operations.append(f"Filter ({type_filter}, cutoff: {Fpass} Hz, order: {N})")


    def psd(self, detrend: str = "linear", nb_points: Optional[int] = None):
        """
        PSD of signal

        Parameters
        ----------
        :param detrend (str): detrend method (optional: default linear)
        :param nb_points (Optional[int]): number of points for FFT (optional: default None)
        """

        if detrend not in ["linear", False]:
            raise ValueError("Detrend method not supported. Available methods: ['linear', False]")

        # check if window is initialized
        if not self.use_window:
            raise ValueError("No window defined. Please define a window when initialising SignalProcessing.")

        # if nb_points is None: nb_points is window length
        if nb_points is None:
            nfft = self.window_size
        else:
            nfft = nb_points

        # compute PSD using Welch method
        self.frequency_Pxx, self.Pxx = signal.welch(self.signal, fs=self.Fs, nperseg=self.window_size, nfft=nfft,
                                                    window=self.window_type.value, scaling='density', detrend=detrend)

        # Add to operations list
        self.operations.append(f"PSD (window: {self.window_type.name}, size: {self.window_size})")


    def v_eff_SBR(self, n: int = 4, tau: float = 0.125):
        """
        Compute v_eff of signal based on SBR deel B Hinder voor personen in gebouwen (2006)

        Parameters
        ----------
        :param n:(int) number of time constants. (optional: default 4)
        :param tau: (float) time constant for the exponential decay (optional: default 0.125)
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

        # Add to operations list
        self.operations.append(f"Effective velocity (SBR) (n={n}, tau={tau})")

    def reset(self):
        """
        Reset signal to original signal and clear all processing results.
        """

        # Reset signal to original
        self.signal = self.signal_org.copy()

        # Reset all processed data
        self.frequency = None
        self.amplitude = None
        self.phase = None
        self.spectrum = None
        self.Pxx = None
        self.frequency_Pxx = None
        self.signal_inv = None
        self.time_inv = None
        self.v_eff = None
        self.Sxx = None
        self.frequency_Sxx = None
        self.time_Sxx = None

        # Reset FFT settings
        self.fft_settings = {"nb_points": None, "half_representation": False}

        # Clear operations history
        self.operations = []

    def spectrogram(self):
        """
        Compute spectrogram of signal
        """
        # compute spectrogram
        f, t, Sxx = signal.spectrogram(self.signal, fs=self.Fs, window=self.window_type.value,
                                       nperseg=self.window_size, noverlap=self.window_size // 8)
        self.Sxx = Sxx
        self.frequency_Sxx = f
        self.time_Sxx = t

        # Add to operations list
        self.operations.append(f"Spectrogram (nperseg: {self.window_size}, noverlap: {self.window_size // 8})")

    def one_third_octave_bands(self):
        """
        Compute octave bands of the signal

        It uses the base 2 calculation for the one-third octave bands, according to ISO 18405:2017.

        """
        # ranges of the nominal one-third octave bands according to ISO 18405:2017
        # https://en.wikipedia.org/wiki/Octave_band
        initial_frequency_band_number = -20
        final_frequency_band_number = 33
        names = ("10", "12.5", "16", "20", "25", "31.5", "40", "50", "63", "80", "100", "125", "160", "200", "250", "315", "400",
                 "500", "630", "800", "1000", "1250", "1600", "2000", "2500", "3.150", "4000", "5000", "6300", "8000",
                 "10000", "12500", "16000", "20000")

        # compute centre frequencies of the bands
        f_centre = 1000 * (2 ** (np.arange(initial_frequency_band_number, final_frequency_band_number) / 3))
        f_upper = f_centre * (2 ** (1 / 6))
        f_lower = f_centre / (2 ** (1 / 6))

        # sum the signal for the bands
        if (self.Pxx is None) and (self.amplitude is None):
            raise ValueError("No PSD nor FFT computed. Please compute either first.")

        if self.Pxx is not None:
            # determine the frequency bands
            idx = np.where((f_lower < np.max(self.frequency_Pxx)) & (f_upper > np.min(self.frequency_Pxx)))[0]
            if len(idx) == 0:
                raise ValueError("No frequency bands found in the PSD. Please check the frequency bands.")

            delta_f = self.frequency_Pxx[1] - self.frequency_Pxx[0]

            # compute the PSD for the bands
            self.octave_bands_Pxx = np.zeros(len(idx))
            self.octave_bands_Pxx_power = np.zeros(len(idx))

            for i, val in enumerate(idx):
                self.octave_bands_Pxx[i] = float(names[val])
                mask = (self.frequency_Pxx >= f_lower[val]) & (self.frequency_Pxx < f_upper[val])
                self.octave_bands_Pxx_power[i] = np.sum(self.Pxx[mask] * delta_f)

        if self.amplitude is not None:
            # determine the frequency bands
            idx = np.where((f_lower < np.max(self.frequency)) & (f_upper > np.min(self.frequency)))[0]
            if len(idx) == 0:
                raise ValueError("No frequency bands found in the FFT. Please check the frequency bands.")

            # compute the FFT for the bands
            self.octave_bands_fft = np.zeros(len(idx))
            self.octave_bands_fft_power = np.zeros(len(idx))

            for i, val in enumerate(idx):
                self.octave_bands_fft[i] = float(names[val])
                mask = (self.frequency >= f_lower[val]) & (self.frequency < f_upper[val])
                self.octave_bands_fft_power[i] = np.sum(self.amplitude[mask]**2)
