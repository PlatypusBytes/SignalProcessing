import pytest
import numpy as np
from SignalProcessing.new import SignalProcessing, IntegrationRules, Windows

TOL = 3e-3
FREQ = 6
AMP = 1.75

@pytest.fixture
def test_data():
    """
    Create test data
    """
    x = np.linspace(0, 100, 10001)
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

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude)], FREQ, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude), AMP, 2)

    # results full representation
    sig.fft(half_representation=False)

    # check if signal lenght has been adapted to window size
    assert len(sig.signal) == 10001
    assert len(sig.time) == 10001

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude) / 2)])], FREQ, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude) / 2)]), AMP / 2, 2)

    # example with spectral leakage
    y = 1.75 * np.sin(2.675 * 2 * np.pi * x)
    sig = SignalProcessing(x, y)
    sig.fft()

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude) / 2)])], 2.675, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude) / 2)]), 1.172, 2)

def test_fft_window(test_data):
    """
    Test the fft function with window
    """
    x, y, _ = test_data

    # assert that sig raises a Value error
    with pytest.raises(ValueError, match="When using a window the `window_size` must be specified"):
        sig = SignalProcessing(x, y, window=Windows.HAMMING)

    # test with window - half representation
    sig = SignalProcessing(x, y, window=Windows.HAMMING, window_size=600)
    sig.fft()

    # check if signal lenght has been adapted to window size
    assert len(sig.signal) == 10200
    assert len(sig.time) == 10200

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude))])], FREQ, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude))]), AMP, 2)

    # full representation
    sig.fft(half_representation=False)

    # check if signal lenght has been adapted to window size
    assert len(sig.signal) == 10200
    assert len(sig.time) == 10200

    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude))])], FREQ, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude))]), AMP / 2, 2)

def test_int(test_data):
    """"
    Test the integration function
    """
    x, y, _ = test_data
    sig = SignalProcessing(x, y)
    sig.integrate(baseline=True, hp=True)

    omega = 2 * np.pi * FREQ
    int_sig = -AMP * np.cos(omega * x) / omega

    rmse = np.sqrt(np.sum((sig.signal - int_sig) ** 2) / len(int_sig))
    assert(rmse < TOL)

    sig = SignalProcessing(x, y)
    sig.integrate(baseline=True, hp=True, rule=IntegrationRules.SIMPSON)
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
    rmse = np.sqrt(np.sum((sig.signal[200:-200] - y[200:-200]) ** 2) / len(y[200:-200]))
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

    sig = SignalProcessing(x, y, window=Windows.HAMMING, window_size=400)
    sig.psd()

    # power
    power_sinus_wave = AMP ** 2 / 2
    bin_width = sig.Fs / sig.window_size
    ENBW = np.sum(sig.window ** 2) / (np.sum(sig.window)**2) * sig.window_size
    peak_psd = power_sinus_wave / (ENBW * bin_width)

    np.testing.assert_almost_equal(sig.frequency_Pxx[np.argmax(sig.Pxx)], FREQ, 2)
    assert (np.abs((np.max(sig.Pxx) - peak_psd) / peak_psd) < 0.035)

@pytest.mark.skip(reason="Not implemented yet")
def test_v_eff():
    """
    Test the v_eff function
    """
    # open the raw data
    with open("./tests/data/raw.csv") as fi:
        raw = fi.read().splitlines()
    raw = np.array([list(map(float, r.split(";"))) for r in raw])

    time = np.linspace(0, (raw.shape[0]-1) / 500, raw.shape[0])

    # compute veff
    for i in range(raw.shape[1]):
        sig = SignalProcessing(time, raw[:, i])
        sig.v_eff()

        print(1)

    # open v_eff data
    with open("./tests/data/veff.csv") as fi:
        v_eff = fi.read().splitlines()
    v_eff = np.array([list(map(float, r.split(";"))) for r in v_eff])




    import matplotlib.pyplot as plt
    plt.plot(np.array(raw)[:, 0])
    plt.plot(np.array(raw)[:, 1])
    plt.plot(np.array(raw)[:, 2])
    plt.plot(np.array(raw)[:, 3])
    plt.show()
    print(1)