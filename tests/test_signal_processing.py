import pytest
import numpy as np
from SignalProcessing.signal import SignalProcessing, IntegrationRules, Windows

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
    sig = SignalProcessing(x, y)
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
    sig = SignalProcessing(x, y)
    sig.fft()

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude) / 2)])], 2.675, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude) / 2)]), 1.137, 2)

def test_fft_window(test_data):
    """
    Test the fft function with window
    """
    x, y, _ = test_data

    # assert that sig raises a Value error
    with pytest.raises(ValueError, match="When using a window the `window_size` must be specified"):
        sig = SignalProcessing(x, y, window=Windows.HAMMING)

    # test with window - half representation
    sig = SignalProcessing(x, y, window=Windows.HAMMING, window_size=6000)
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
    sig = SignalProcessing(x, y)
    sig.fft(half_representation=True)

    # assert that sig raises a Value error
    with pytest.raises(NotImplementedError, match="Half representation not supported for inverse FFT. Please compute FFT with full representation."):
        sig.inv_fft()

    # test with window - full representation
    sig = SignalProcessing(x, y)
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
    sig = SignalProcessing(x, y, window=Windows.HAMMING, window_size=6000)
    sig.fft(half_representation=True)

    # assert that sig raises a Value error
    with pytest.raises(NotImplementedError, match="Half representation not supported for inverse FFT. Please compute FFT with full representation."):
        sig.inv_fft()

    # test with window - half representation
    sig = SignalProcessing(x, y, window=Windows.HAMMING, window_size=6000)
    sig.fft(half_representation=False)

    # assert that sig raises a Value error
    with pytest.raises(ValueError, match="Cannot perform inverse FFT on the windowed signal."):
        sig.inv_fft()


def test_int(test_data):
    """"
    Test the integration function
    """
    x, y, _ = test_data
    sig = SignalProcessing(x, y)
    sig.integrate(baseline=True, hp=True, fpass=1, n=6)

    omega = 2 * np.pi * FREQ
    int_sig = -AMP * np.cos(omega * x) / omega

    rmse = np.sqrt(np.sum((sig.signal - int_sig) ** 2) / len(int_sig))
    assert(rmse < TOL)

    sig = SignalProcessing(x, y)
    sig.integrate(baseline=True, hp=True, rule=IntegrationRules.SIMPSON, fpass=1, n=6)
    rmse = np.sqrt(np.sum((sig.signal - int_sig) ** 2) / len(int_sig))
    assert(rmse < TOL)

def test_filter(test_data):
    """
    Test the filter function
    """
    x, y, y_noise = test_data
    sig = SignalProcessing(x, y_noise)
    sig.filter(10, 4, type_filter="lowpass")

    # compare between 200 and -200 to avoid edge effects
    rmse = np.sqrt(np.sum((sig.signal[400:-400] - y[400:-400]) ** 2) / len(y[400:-400]))
    assert(rmse < TOL)

def test_filter(test_data):
    """
    Test the filter function
    """
    x, y, y_noise = test_data
    sig = SignalProcessing(x, y_noise)
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
        sig = SignalProcessing(x, y)
        sig.psd()

    sig = SignalProcessing(x, y, window=Windows.HAMMING, window_size=4000)
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
        sig = SignalProcessing(time, raw[:, i])
        sig.v_eff_SBR()
        np.testing.assert_almost_equal(sig.v_eff, np.array(v_eff)[:, i], 2)

def test_str_representation(test_data):
    """
    Test the __str__ method to verify operations are tracked correctly
    """
    # Create signal instance
    x, y, _ = test_data
    sig = SignalProcessing(x, y, window=Windows.HAMMING, window_size=600)

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
