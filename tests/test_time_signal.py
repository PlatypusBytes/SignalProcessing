import pytest
import numpy as np

from SignalProcessingTools.time_signal import TimeSignalProcessing, IntegrationRules, Windows

TOL = 3e-3
FREQ = 6
AMP = 1.75

@pytest.fixture
def test_data():
    """
    Create test data
    """
    x = np.linspace(0, 100, 50001)
    omega = 2 * np.pi * FREQ
    y = AMP * np.sin(omega * x)
    y_noise = y + 0.01 * np.sin(120 * x)
    return x, y, y_noise

def test_fft(test_data):
    """
    Test the fft function
    """

    # results half representation
    x, y, _ = test_data
    sig = TimeSignalProcessing(x, y)
    sig.fft()

    # check if signal lenght has been adapted to window size
    assert len(sig.signal) == 50001
    assert len(sig.time) == 50001

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude)], FREQ, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude), AMP, 2)


    # results full representation
    sig.fft(half_representation=False)

    # check if signal lenght has been adapted to window size
    assert len(sig.signal) == 50001
    assert len(sig.time) == 50001

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude) / 2)])], FREQ, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude) / 2)]), AMP / 2, 2)

    # example with spectral leakage
    y = 1.75 * np.sin(2.675 * 2 * np.pi * x)
    sig = TimeSignalProcessing(x, y)
    sig.fft()

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude) / 2)])], 2.675, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude) / 2)]), 1.137, 2)

def test_fft_nb_points(test_data):
    """
    Test the fft function
    """

    # results half representation
    x, y, _ = test_data
    sig = TimeSignalProcessing(x, y)
    sig.fft(nb_points=2**18)

    # check if signal lenght has been adapted to window size
    assert len(sig.amplitude) == (2**18)/2
    assert len(sig.frequency) == (2**18)/2

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude)], FREQ, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude), AMP, 2)

    # results full representation
    sig.fft(nb_points=2**18, half_representation=False)

    # check if signal lenght has been adapted to window size
    assert len(sig.amplitude) == 2**18
    assert len(sig.frequency) == 2**18

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude) / 2)])], FREQ, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude) / 2)]), AMP / 2, 2)


def test_fft_window(test_data):
    """
    Test the fft function with window
    """
    x, y, _ = test_data

    # assert that sig raises a Value error
    with pytest.raises(ValueError, match="When using a window the `window_size` must be specified"):
        sig = TimeSignalProcessing(x, y, window=Windows.HAMMING)

    # test with window - half representation
    sig = TimeSignalProcessing(x, y, window=Windows.HAMMING, window_size=6000)
    sig.fft()

    # check if signal lenght has been adapted to window size
    assert len(sig.signal) == 54000
    assert len(sig.time) == 54000

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude))])], FREQ, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude))]), AMP, 2)

    # full representation
    sig.fft(half_representation=False)

    # check if signal lenght has been adapted to window size
    assert len(sig.signal) == 54000
    assert len(sig.time) == 54000

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude))])], FREQ, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude))]), AMP / 2, 2)


def test_ifft(test_data):
    """
    Test the fft function with window
    """
    x, y, _ = test_data

    # test with window - half representation
    sig = TimeSignalProcessing(x, y)
    sig.fft(half_representation=True)

    # assert that sig raises a Value error
    with pytest.raises(NotImplementedError, match="Half representation not supported for inverse FFT. Please compute FFT with full representation."):
        sig.inv_fft()

    # test with window - full representation
    sig = TimeSignalProcessing(x, y)
    sig.fft(half_representation=False)

    sig.inv_fft()

    # check if signal lenght has been adapted to window size
    assert len(sig.signal) == len(sig.signal_inv)

    rmse = np.sqrt(np.sum((sig.signal - sig.signal_inv) ** 2) / len(y))
    assert(rmse < TOL)

def test_ifft_window(test_data):
    """
    Test the fft function with window
    """
    x, y, _ = test_data

    # test with window - half representation
    sig = TimeSignalProcessing(x, y, window=Windows.HAMMING, window_size=6000)
    sig.fft(half_representation=True)

    # assert that sig raises a Value error
    with pytest.raises(NotImplementedError, match="Half representation not supported for inverse FFT. Please compute FFT with full representation."):
        sig.inv_fft()

    # test with window - half representation
    sig = TimeSignalProcessing(x, y, window=Windows.HAMMING, window_size=6000)
    sig.fft(half_representation=False)

    # assert that sig raises a Value error
    with pytest.raises(ValueError, match="Cannot perform inverse FFT on the windowed signal."):
        sig.inv_fft()


def test_int(test_data):
    """"
    Test the integration function
    """
    x, y, _ = test_data
    sig = TimeSignalProcessing(x, y)
    sig.integrate(baseline=True, hp=True, fpass=1, n=6)

    omega = 2 * np.pi * FREQ
    int_sig = -AMP * np.cos(omega * x) / omega

    rmse = np.sqrt(np.sum((sig.signal - int_sig) ** 2) / len(int_sig))
    assert(rmse < TOL)

    sig = TimeSignalProcessing(x, y)
    sig.integrate(baseline=True, hp=True, rule=IntegrationRules.SIMPSON, fpass=1, n=6)
    rmse = np.sqrt(np.sum((sig.signal - int_sig) ** 2) / len(int_sig))
    assert(rmse < TOL)

def test_filter(test_data):
    """
    Test the filter function
    """
    x, y, y_noise = test_data
    sig = TimeSignalProcessing(x, y_noise)
    sig.filter(10, 4, type_filter="lowpass")

    # compare between 200 and -200 to avoid edge effects
    rmse = np.sqrt(np.sum((sig.signal[200:-200] - y[200:-200]) ** 2) / len(y[200:-200]))
    assert(rmse < TOL)

def test_psd(test_data):
    """
    Test the psd function
    """
    x, y, _ = test_data
    with pytest.raises(ValueError, match="No window defined. Please define a window when initialising SignalProcessing."):
        sig = TimeSignalProcessing(x, y)
        sig.psd()

    sig = TimeSignalProcessing(x, y, window=Windows.HAMMING, window_size=4000)
    sig.psd()

    # power
    power_sinus_wave = AMP ** 2 / 2
    bin_width = sig.Fs / sig.window_size
    ENBW = np.sum(sig.window ** 2) / (np.sum(sig.window)**2) * sig.window_size
    peak_psd = power_sinus_wave / (ENBW * bin_width)

    np.testing.assert_almost_equal(sig.frequency_Pxx[np.argmax(sig.Pxx)], FREQ, 2)
    assert (np.abs((np.max(sig.Pxx) - peak_psd) / peak_psd) < 0.035)

def test_v_eff():
    """
    Test the v_eff function
    """
    # open the raw data
    with open("./tests/data/raw.csv") as fi:
        raw = fi.read().splitlines()
    raw = np.array([list(map(float, r.split(";"))) for r in raw])

    # open v_eff data
    with open("./tests/data/veff.csv") as fi:
        v_eff = fi.read().splitlines()
    v_eff = np.array([list(map(float, r.split(";"))) for r in v_eff])

    time = np.linspace(0, (raw.shape[0]-1) / 500, raw.shape[0])

    # compute veff
    for i in range(raw.shape[1]):
        sig = TimeSignalProcessing(time, raw[:, i])
        sig.v_eff_SBR()
        np.testing.assert_almost_equal(sig.v_eff, np.array(v_eff)[:, i], 2)

def test_str_representation(test_data):
    """
    Test the __str__ method to verify operations are tracked correctly
    """
    # Create signal instance
    x, y, _ = test_data
    sig = TimeSignalProcessing(x, y, window=Windows.HAMMING, window_size=600)

    # Get initial string representation
    str_repr = str(sig)

    # Verify basic information is in the representation
    assert "SignalProcessing Object" in str_repr
    assert f"Signal length: {len(sig.signal)}" in str_repr
    assert f"Sampling frequency: {sig.Fs}" in str_repr
    assert f"Window type: {sig.window_type.name}" in str_repr

    # Verify initial padding operation is tracked if window size doesn't divide signal length
    if len(x) % 600 != 0:
        assert "Signal padded with zeros" in str_repr

    # Perform operations
    sig.fft()
    sig.filter(10, 4, type_filter="lowpass")
    sig.psd()

    # Get updated string representation
    str_repr = str(sig)

    # Verify operations are tracked
    assert "FFT" in str_repr
    assert "Filter (lowpass" in str_repr
    assert "PSD" in str_repr

    # Check operations list directly
    assert len(sig.operations) >= 3
    assert any("FFT" in op for op in sig.operations)
    assert any("Filter" in op for op in sig.operations)
    assert any("PSD" in op for op in sig.operations)

def test_spectrogram(test_data):
    """
    Test the spectrogram function
    """
    x, y, _ = test_data
    sig = TimeSignalProcessing(x, y, window=Windows.HAMMING, window_size=600)
    sig.spectrogram()

    # check if signal lenght has been adapted to window size
    assert sig.Sxx.shape == (301, 95)
    assert sig.time_Sxx.shape == (95,)
    assert sig.frequency_Sxx.shape == (301,)

    np.testing.assert_almost_equal(sig.frequency_Sxx[np.where(sig.Sxx==np.max(sig.Sxx))[0][0]], FREQ, 0)
    np.testing.assert_almost_equal(np.max(sig.Sxx), 1.27, 2)

    # # plot spectrogram
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [s]')
    # plt.title('Spectrogram')
    # plt.colorbar(label='Intensity [dB]')
    # plt.show()


def test_reset(test_data):
    """
    Test the reset function to ensure it properly restores the object to its original state
    """
    # Create signal instance
    x, y, _ = test_data
    sig = TimeSignalProcessing(x, y, window=Windows.HAMMING, window_size=600)

    # Store original signal for comparison
    original_signal = sig.signal.copy()

    # Perform various operations
    sig.fft()
    sig.filter(10, 4, type_filter="lowpass")
    sig.integrate(baseline=True, hp=True, fpass=1, n=6)
    sig.psd()

    # Verify operations were tracked and signal was modified
    assert len(sig.operations) > 0
    assert sig.frequency is not None
    assert sig.amplitude is not None
    assert sig.Pxx is not None
    assert not np.array_equal(sig.signal, original_signal)

    # Reset the object
    sig.reset()

    # Verify signal is reset to original
    np.testing.assert_array_equal(sig.signal, sig.signal_org)

    # Verify all processing results are cleared
    assert sig.frequency is None
    assert sig.amplitude is None
    assert sig.phase is None
    assert sig.spectrum is None
    assert sig.Pxx is None
    assert sig.frequency_Pxx is None
    assert sig.signal_inv is None
    assert sig.time_inv is None
    assert sig.v_eff is None
    assert sig.Sxx is None
    assert sig.frequency_Sxx is None
    assert sig.time_Sxx is None

    # Verify FFT settings are reset
    assert sig.fft_settings == {"nb_points": None, "half_representation": False}

    # Verify operations history is cleared
    assert len(sig.operations) == 0

    # Verify string representation shows no operations
    assert "No operations performed yet" in str(sig)

def test_one_third_octave(test_data):
    """
    Test the one third octave function
    """
    x, y, _ = test_data

    # definition of frequencies used in the test
    freqs_used = [10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250]
    # compute the power at 20 Hz
    f_center = 1000 * (2 ** ((-17) / 3))
    f_max = f_center * (2 ** (1 / 6))
    f_min = f_center / (2 ** (1 / 6))

    # test FFT
    sig = TimeSignalProcessing(x, y)
    sig.fft()
    sig.one_third_octave_bands()

    assert all(sig.octave_bands_fft == freqs_used)

    idx = np.where((sig.frequency >= f_min) & (sig.frequency < f_max))[0]

    assert sig.octave_bands_fft_power[3] == np.sum(sig.amplitude[idx] ** 2)
    assert sig.octave_bands_fft[3] == 20

    # test PSDF
    sig = TimeSignalProcessing(x, y, window=Windows.HAMMING, window_size=4000)
    sig.psd()
    sig.one_third_octave_bands()
    assert all(sig.octave_bands_Pxx == freqs_used)

    idx = np.where((sig.frequency_Pxx >= f_min) & (sig.frequency_Pxx < f_max))[0]
    delta_freq = sig.frequency_Pxx[1] - sig.frequency_Pxx[0]
    assert sig.octave_bands_Pxx_power[3] == np.sum(sig.Pxx[idx] * delta_freq)
    assert sig.octave_bands_Pxx[3] == 20
