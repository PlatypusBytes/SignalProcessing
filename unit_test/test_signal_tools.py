import unittest
import numpy as np
from SignalProcessing import signal_tools, window

TOL = 0.025


class TestSignal(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 100, 10001)
        self.y = np.sin(20 * self.x)
        self.y_noise = np.sin(20 * self.x) + 0.1 * np.sin(120 * self.x)
        return

    def tearDown(self):
        pass

    def test_log(self):
        sig = signal_tools.Signal(self.x, self.y)

        log = {"FFT": False,
               "Half_FFT": False,
               "IFFT": False,
               "PSD": False,
               "Integration": False,
               "Filter": False}
        self.assertEqual(sig.log, log)

        sig.fft()
        log = {"FFT": True,
               "Half_FFT": False,
               "IFFT": False,
               "PSD": False,
               "Integration": False,
               "Filter": False}
        self.assertEqual(sig.log, log)

        sig.inv_fft()
        log = {"FFT": True,
               "Half_FFT": False,
               "IFFT": True,
               "PSD": False,
               "Integration": False,
               "Filter": False}
        self.assertEqual(sig.log, log)

        sig.integrate()
        log = {"FFT": True,
               "Half_FFT": False,
               "IFFT": True,
               "PSD": False,
               "Integration": {"Integration": True,
                               "Order": 1,
                               "Baseline": False,
                               "Moving": False,
                               "High-pass": False},
               "Filter": False}
        self.assertEqual(sig.log, log)

        sig.filter(1, 2)
        log = {"FFT": True,
               "Half_FFT": False,
               "IFFT": True,
               "PSD": False,
               "Integration": {"Integration": True,
                               "Order": 1,
                               "Baseline": False,
                               "Moving": False,
                               "High-pass": False},
               "Filter": {"Type": ["lowpass"],
                          "Cut-off": [1]
                          }}
        self.assertEqual(sig.log, log)

        sig.psd(length_w=10)
        log = {"FFT": True,
               "Half_FFT": False,
               "IFFT": True,
               "PSD": True,
               "Integration": {"Integration": True,
                               "Order": 1,
                               "Baseline": False,
                               "Moving": False,
                               "High-pass": False},
               "Filter": {"Type": ["lowpass"],
                          "Cut-off": [1]
                          }}
        self.assertEqual(sig.log, log)

        sig.fft(half_representation=True)
        log = {"FFT": True,
               "Half_FFT": True,
               "IFFT": True,
               "PSD": True,
               "Integration": {"Integration": True,
                               "Order": 1,
                               "Baseline": False,
                               "Moving": False,
                               "High-pass": False},
               "Filter": {"Type": ["lowpass"],
                          "Cut-off": [1]
                          }}
        self.assertEqual(sig.log, log)
        return

    def test_fft(self):
        sig = signal_tools.Signal(self.x, self.y)
        sig.fft()

        self.assertAlmostEqual(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude) / 2)])], 20 / 2 / np.pi, 2)
        self.assertAlmostEqual(np.max(sig.amplitude[:int(len(sig.amplitude) / 2)]), 0.393, 2)

        return

    def test_int(self):
        sig = signal_tools.Signal(self.x, self.y)
        sig.integrate(baseline=True, hp=True)

        int_sig = -np.cos(20 * self.x) / 20

        rmse = np.sqrt(np.sum((sig.signal - int_sig) ** 2) / len(int_sig))
        self.assertTrue(rmse < TOL)

        return

    def test_filter(self):
        sig = signal_tools.Signal(self.x, self.y_noise)
        sig.filter(10, 20, type_filter="lowpass")

        rmse = np.sqrt(np.sum((sig.signal - self.y) ** 2) / len(self.y))
        self.assertTrue(rmse < TOL)

        return

    def test_log_window(self):
        sig = window.Window(self.x, self.y, 10)

        log = {"FFT": False,
               "Half_FFT": False,
               "IFFT": False,
               "PSD": False,
               "Integration": False,
               "Filter": False,
               "Window": {"Type": "Hann",
                          "Length": 10}}
        self.assertEqual(sig.log, log)

        sig.fft_w()
        log = {"FFT": True,
               "Half_FFT": False,
               "IFFT": False,
               "PSD": False,
               "Integration": False,
               "Filter": False,
               "Window": {"Type": "Hann",
                          "Length": 10}}
        self.assertEqual(sig.log, log)

        sig.integrate_w()
        log = {"FFT": True,
               "Half_FFT": False,
               "IFFT": False,
               "PSD": False,
               "Integration": {"Integration": True,
                               "Order": 1,
                               "Baseline": False,
                               "Moving": False,
                               "High-pass": False},
               "Filter": False,
               "Window": {"Type": "Hann",
                          "Length": 10}}
        self.assertEqual(sig.log, log)

        sig.filter_w(1, 2)
        log = {"FFT": True,
               "Half_FFT": False,
               "IFFT": False,
               "PSD": False,
               "Integration": {"Integration": True,
                               "Order": 1,
                               "Baseline": False,
                               "Moving": False,
                               "High-pass": False},
               "Filter": {"Type": ["lowpass"],
                          "Cut-off": [1]
                          },
               "Window": {"Type": "Hann",
                          "Length": 10}}
        self.assertEqual(sig.log, log)
        return

    def test_fft_w(self):
        sig = window.Window(self.x, self.y, 512)
        sig.fft_w()

        self.assertAlmostEqual(sig.frequency[np.argmax(sig.amplitude[:int(len(sig.amplitude) / 2)])], 20 / 2 / np.pi, 1)
        self.assertAlmostEqual(np.max(sig.amplitude[:int(len(sig.amplitude) / 2)]), 0.026, 2)

        return

    def test_int_w(self):
        w_lenght = 256
        sig = window.Window(self.x, self.y, w_lenght)
        sig.integrate_w(baseline=True, hp=True)

        int_sig = -np.cos(20 * self.x) / 20

        rmse = np.sqrt(np.sum((sig.signal[w_lenght:len(self.y)-w_lenght] - int_sig[w_lenght:-w_lenght]) ** 2) / len(self.y[:w_lenght*2]))
        self.assertTrue(rmse < TOL)

        return

    def test_filter_w(self):
        w_lenght = 256
        sig = window.Window(self.x, self.y_noise, w_lenght)
        sig.filter_w(10, 20, type_filter="lowpass")

        rmse = np.sqrt(np.sum((sig.signal[w_lenght:len(self.y)-w_lenght] - self.y[w_lenght:-w_lenght]) ** 2) / len(self.y[:w_lenght*2]))
        self.assertTrue(rmse < TOL)
        return
