from typing import Optional, List, Tuple
import numpy as np
import numpy.typing as npt
from .time_signal import TimeSignalProcessing, FilterDesign, Windows


class SpaceSignalProcessing:
    """
    SignalProcessing class for processing signals in space.
    """
    def __init__(self, x: npt.NDArray[np.float64], values: npt.NDArray[np.float64], Fs: Optional[float] = None):
        """
        Initializes the ProcessSignal object.

        Parameters
        ----------
        :param x (npt.NDArray[np.float64]): coordinates of the signal
        :param values (npt.NDArray[np.float64]): signal values
        :param Fs (Optional[float]): sampling frequency of the signal (optional: default None)
        """

        self.coordinates = x
        self.signal_raw = values
        self.n_points = values.shape[0]

        # acquisition frequency
        if Fs is None:
            self.fs = int(np.ceil(1 / np.mean(np.diff(x))))

        else:
            self.fs = Fs

        # track quality indexes
        self.d0 = None
        self.d1 = None
        self.d2 = None
        self.d3 = None
        # track descriptors
        self.rms_bands = None
        self.max_fast = None
        self.max_fast_Dx = None


    def compute_track_longitudinal_levels(self):
        """
        Computes the track longitudinal levels, following EN 13848-1:2006.

        The method computes the D0, D1, D2, and D3 components of the signal.
        It uses the following frequency bands:
        - D0: 1m < lambda <= 5m (1/5 Hz < f <= 1 Hz)
        - D1: 3m < lambda <= 25m (1/25 Hz < f <= 1/3 Hz)
        - D2: 25m < lambda <= 70m (1/70 Hz < f <= 1/25 Hz)
        - D3: 70m < lambda <= 150m (1/150 Hz < f <= 1/70 Hz)
        """

        sig = TimeSignalProcessing(self.coordinates, self.signal_raw, Fs=self.fs)
        sig.filter([1/5., 1.], 4, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)
        self.d0 = sig.signal

        sig.reset()
        sig.filter([1/25., 1/3.], 4, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)
        self.d1 = sig.signal
        sig.reset()

        sig.filter([1/70., 1/25.], 4, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)
        self.d2 = sig.signal
        sig.reset()

        sig.filter([1/150., 1/70.], 4, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)
        self.d3 = sig.signal
        sig.reset()

    def compute_Hmax(self, convert_m2mm: bool = True):
        """
        Computes the descriptor Hmax and Hrms according to Zandberg et al. (2022)
        'Deriving parameters for the characterisation of the railway track quality in
          relation to environmental vibration'

        Parameters
        ----------
        :param convert_m2mm (optional, default = True): if True, converts the results from m to mm
        """

        # octave bands used for the processing
        one_third_octave_bands = [[.08, .10],
                                  [.10, .126],
                                  [.126, .16],
                                  [.16, .20],
                                  [.20, .253],
                                  [.253, .32],
                                  [.32, .40],
                                  [.40, .50],
                                  [.50, .63],
                                  ]


        # setting for the processing
        self.DXmaxFast = 1
        nb_fft_min = 256  # minimum number of samples for the power spectral density
        derivative = [0, 0, 0, 2, 2, 2, 2, 2, 2]  # number of times that each frequency band is derived

        # RMS of the square root of the power spectral density
        self.rms_bands = np.zeros(len(one_third_octave_bands))
        # maximum effective value over the entire signal
        self.max_fast = np.zeros(len(one_third_octave_bands))
        # maximum effective value over the length Dx
        self.max_fast_Dx = np.zeros(len(one_third_octave_bands))

        # convert the signal from m to mm
        if convert_m2mm:
            self.signal = self.signal_raw * 1000

        # compute the power spectral density
        n_fft = int(np.max([2 ** (np.ceil(np.log2(len(self.signal)))), nb_fft_min]))
        # if signal is odd length, add a zero to make it even
        if len(self.signal) % 2 != 0:
            signal = np.append(self.signal, 0)
            coordinates = np.append(self.coordinates, self.coordinates[-1] + (self.coordinates[1] - self.coordinates[0]))
        else:
            signal = self.signal
            coordinates = self.coordinates
        sig = TimeSignalProcessing(coordinates, signal, Fs=self.fs, window=Windows.HAMMING,
                                   window_size=len(signal))
        sig.psd(nb_points=n_fft, detrend=False)

        # compute the rsm psd
        self.__rms_effective(sig.frequency_Pxx, sig.Pxx, one_third_octave_bands, derivative)
        # compute the effective values
        self.__effective_values(one_third_octave_bands, derivative)


    def __rms_effective(self, frequency: npt.NDArray[np.float64], Pxx: npt.NDArray[np.float64],
                        one_third_octave_bands: List[Tuple[float, float]], derivative: List[int]):
        """
        Computes RMS square root of power spectral density

        Parameters
        ----------
        :param frequency (npt.NDArray[np.float64]): frequency vector
        :param Pxx (npt.NDArray[np.float64]): power spectral density
        :param one_third_octave_bands (List[Tuple[float, float]]): frequency bands
        :param derivative (List[int]): derivative order
        """
        # frequency step
        delta_f = frequency[1] - frequency[0]

        # compute the rms value at each frequency band
        for i, band in enumerate(one_third_octave_bands):
            # find indexes where the bands exist
            idx = np.where((frequency >= band[0]) & (frequency < band[1]))[0]
            Pxx[idx] = (2 * np.pi * frequency[idx]) ** (2 * derivative[i]) * Pxx[idx]
            self.rms_bands[i] = np.sqrt(np.sum(Pxx[idx] * delta_f))

    def __effective_values(self, one_third_octave_bands: List[Tuple[float, float]], derivative: List[int]):
        """
        Computes the effective values of the signal

        Parameters
        ----------
        :param one_third_octave_bands (List[Tuple[float, float]]): frequency bands
        :param derivative (List[int]): derivative order
        """

        n = 4  # number of time constants
        tau = 2  # time constant

        fout = 1 / (1 - np.exp(-n))

        dx = self.coordinates[1] - self.coordinates[0]

        for i, band in enumerate(one_third_octave_bands):
            derivative_value = derivative[i]
            sig = TimeSignalProcessing(self.coordinates, self.signal, Fs=self.fs)
            sig.filter(np.array(band), N=3, type_filter="bandpass", filter_design=FilterDesign.BUTTERWORTH)
            new_signal = sig.signal

            while derivative_value != 0:
                new_signal = np.diff(new_signal) / dx
                derivative_value -= 1

            ksi = np.linspace(0, n * tau, int(n * tau / dx + 1))
            g = fout * np.exp(-ksi / tau)

            convoluted_signal = np.sqrt(np.convolve(new_signal**2, g) * dx / tau)
            self.max_fast[i] = np.max(convoluted_signal)
            idx = np.floor((len(self.signal) - np.floor(self.DXmaxFast / dx)) / 2) + \
                  np.linspace(0, np.floor(self.DXmaxFast / dx)-1, int(np.floor(self.DXmaxFast / dx)))

            self.max_fast_Dx[i] = np.max(convoluted_signal[idx.astype(int)])
