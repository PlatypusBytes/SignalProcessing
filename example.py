import numpy as np
import matplotlib.pylab as plt
from SignalProcessing import signal_tools, window


def signal_ex():
    x = np.linspace(0, 100, 1000)
    y = np.sin(20 * x) + np.random.random(len(x)) * 0.01

    sig = signal_tools.Signal(x, y)
    sig.fft()
    sig.inv_fft()

    plt.plot(sig.frequency, sig.amplitude)
    plt.grid()
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.show()

    plt.plot(x, y, marker="x")
    plt.plot(sig.time_inv, sig.signal_inv)
    plt.grid()
    plt.show()
    return


def integration_ex():
    time = np.linspace(0, 100, 10000)
    x = np.sin(20 * time)

    # check window integration
    sig = signal_tools.Signal(time, x, FS=1 / time[1])
    sig.integrate(hp=True)

    plt.plot(time, -np.cos(20 * time) / 20)
    plt.plot(sig.time, sig.signal)
    plt.ylabel("Integration")
    plt.show()
    return


def window_ex():
    time = np.linspace(0, 100, 10000)
    sig = np.sin(20 * time)

    # check window fft
    s = signal_tools.Signal(time, sig, FS=1 / time[1])
    s.fft()
    plt.plot(s.frequency, s.amplitude)

    w = window.Window(time, sig, 4096, FS=1/time[1])
    w.fft()
    plt.plot(w.frequency, np.mean(w.signal_fft, axis=1))
    plt.xlim(0, 10)
    plt.grid()
    plt.show()

    # check window integration
    w = window.Window(time, sig, 512, FS=1/time[1])
    w.integration()

    plt.plot(time, -np.cos(20 * time) / 20)
    plt.plot(w.time, w.signal_int)
    plt.show()

    return


if __name__ == "__main__":
    # example on how to use signal processing for fft
    signal_ex()
    # example on how to use signal processing for integration
    integration_ex()
    # example on how to use window for fft and integration
    window_ex()
