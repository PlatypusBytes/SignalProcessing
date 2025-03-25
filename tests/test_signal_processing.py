import pytest
import numpy as np
from SignalProcessing.new import SignalProcessing, IntegrationRules

TOL = 1e-3


@pytest.fixture
def test_data():
    """
    Create test data
    """
    x = np.linspace(0, 100, 10001)
    y = np.sin(20 * x)
    y_noise = np.sin(20 * x) + 0.1 * np.sin(120 * x)
    return x, y, y_noise

def test_fft(test_data):
    """
    Test the fft function
    """
    x, y, _ = test_data
    sig = SignalProcessing(x, y)
    sig.fft()

    # results half representation
    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude))])], 20 / 2 / np.pi, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude))]), 0.393*2, 2)

    # results full representation
    x, y, _ = test_data
    sig = SignalProcessing(x, y)
    sig.fft(half_representation=False)

    # results half representation
    np.testing.assert_almost_equal(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude) / 2)])], 20 / 2 / np.pi, 2)
    np.testing.assert_almost_equal(np.max(sig.amplitude[:int(len(sig.amplitude) / 2)]), 0.393, 2)

def test_int(test_data):
    """"
    Test the integration function
    """
    x, y, _ = test_data
    sig = SignalProcessing(x, y)
    sig.integrate(baseline=True, hp=True)

    int_sig = -np.cos(20 * x) / 20

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
    sig.filter(10, 20, type_filter="lowpass")

    # compare between 200 and -200 to avoid edge effects
    rmse = np.sqrt(np.sum((sig.signal[200:-200] - y[200:-200]) ** 2) / len(y[200:-200]))
    assert(rmse < TOL)

def test_psd(test_data):
    """
    Test the psd function
    """
    x, y, _ = test_data
    sig = SignalProcessing(x, y)
    sig.psd(length_w=4096)

    np.testing.assert_almost_equal(sig.frequency_Pxx[np.argmax(sig.Pxx[:int(len(sig.Pxx))])], 20 / 2 / np.pi, 2)
    np.testing.assert_almost_equal(np.max(sig.Pxx[:int(len(sig.Pxx))]), 0.393, 2)
