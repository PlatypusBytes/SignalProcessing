import numpy as np
import matplotlib.pylab as plt
from SignalProcessingTools.time_signal import TimeSignalProcessing, IntegrationRules, Windows

# Create test data
x = np.linspace(0, 100, 50001)
omega = 2 * np.pi * 6
y = 1.75 * np.sin(omega * x)
y_noise = y + 0.01 * np.sin(120 * x)

# Create a SignalProcessing object and demonstrate basic functionality
print("------------------------------------")
print("EXAMPLE 1: Basic FFT and integration")
print("------------------------------------")
sig = TimeSignalProcessing(x, y)
print(sig)

# Perform FFT
sig.fft()

plt.figure(figsize=(10, 6))
plt.plot(sig.frequency, sig.amplitude)
plt.xlim(0, 20)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('FFT of the signal')
plt.grid()
plt.show()

# Perform integration
sig.integrate(baseline=True, hp=True, fpass=1, n=6)

plt.figure(figsize=(10, 6))
plt.plot(sig.time, sig.signal)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Integrated signal')
plt.grid()
plt.show()

# Print the signal operations
print(sig)

print("RESET signal")
sig.reset()
print(sig)

# Example 2: Working with windowed signals and PSD
print("---------------------------------------")
print("EXAMPLE 2: Windowed processing and PSD")
print("---------------------------------------")
sig_window = TimeSignalProcessing(x,
                                  y_noise,
                                  window=Windows.HAMMING,
                                  window_size=4096)
print(sig_window)

# Calculate and plot PSD
sig_window.psd()
plt.figure(figsize=(10, 6))
plt.semilogy(sig_window.frequency_Pxx, sig_window.Pxx)
plt.xlim(0, 20)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power/Frequency [V²/Hz]')
plt.title('Power Spectral Density')
plt.grid()
plt.show()

# Calculate and plot spectrogram
sig_window.spectrogram()
plt.figure(figsize=(10, 6))
plt.pcolormesh(sig_window.time_Sxx,
               sig_window.frequency_Sxx,
               10 * np.log10(sig_window.Sxx),
               shading='gouraud')
plt.ylim(0, 20)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram')
plt.colorbar(label='Intensity [dB]')
plt.show()

print(sig_window)

# Example 3: Signal filtering
print("---------------------------")
print("EXAMPLE 3: Signal filtering")
print("---------------------------")
# Create a new signal with noise
sig_filter = TimeSignalProcessing(x, y_noise)
print(sig_filter)

plt.figure(figsize=(10, 6))
plt.plot(sig_filter.time[:500], sig_filter.signal[:500], label='Noisy signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Original signal with noise')
plt.grid()
plt.legend()
plt.show()

# Apply lowpass filter to remove high frequency noise
sig_filter.filter(10, 4, type_filter="lowpass")

plt.figure(figsize=(10, 6))
plt.plot(sig_filter.time[:500],
         sig_filter.signal[:500],
         label='Filtered signal')
plt.plot(x[:500], y[:500], '--', label='Original clean signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Low-pass filtered signal (cutoff at 10 Hz)')
plt.grid()
plt.legend()
plt.show()

print(sig_filter)

# Example 4: Inverse FFT
print("----------------------")
print("EXAMPLE 4: Inverse FFT")
print("----------------------")
sig_ifft = TimeSignalProcessing(x, y)
print(sig_ifft)

# Perform full representation FFT (needed for inverse FFT)
sig_ifft.fft(half_representation=False)

plt.figure(figsize=(10, 6))
plt.plot(sig_ifft.frequency[:500], np.abs(sig_ifft.amplitude[:500]))
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.title('FFT with full representation')
plt.grid()
plt.show()

# Perform inverse FFT
sig_ifft.inv_fft()

plt.figure(figsize=(10, 6))
plt.plot(sig_ifft.time[:500], sig_ifft.signal[:500], label='Original signal')
plt.plot(sig_ifft.time_inv[:500],
         sig_ifft.signal_inv[:500],
         '--',
         label='Reconstructed signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Signal Reconstruction with Inverse FFT')
plt.grid()
plt.legend()
plt.show()

# Calculate RMSE between original and reconstructed signals
rmse = np.sqrt(
    np.sum((sig_ifft.signal[:500] - sig_ifft.signal_inv[:500])**2) / 500)
print(f"RMSE between original and reconstructed signals: {rmse:.6f}")
print(sig_ifft)

# Example 5: Effective velocity calculation using SBR method
print("-------------------------------------------")
print("EXAMPLE 5: Effective velocity (SBR method)")
print("-------------------------------------------")

# Create a vibration signal (using more complex multi-frequency signal)
t = np.linspace(0, 10, 5001)
vib_signal = 0.5 * np.sin(2 * np.pi * 2 * t) + 0.3 * np.sin(
    2 * np.pi * 8 * t) + 0.2 * np.sin(2 * np.pi * 15 * t)

sig_veff = TimeSignalProcessing(t, vib_signal)
# Calculate effective velocity
sig_veff.v_eff_SBR()

plt.figure(figsize=(10, 6))
plt.plot(sig_veff.time, sig_veff.signal, label='Original vibration signal')
plt.plot(sig_veff.time, sig_veff.v_eff, label='Effective velocity')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Vibration Signal and Effective Velocity')
plt.grid()
plt.legend()
plt.show()

print(sig_veff)

# Example 6: Different integration rules
print("---------------------------------------")
print("EXAMPLE 6: Integration rules comparison")
print("---------------------------------------")

# Create a simple acceleration signal
t_acc = np.linspace(0, 5, 5001)
acc = np.sin(2 * np.pi * 1 * t_acc)

# Trapezoid integration
sig_trap = TimeSignalProcessing(t_acc, acc)
sig_trap.integrate(rule=IntegrationRules.TRAPEZOID, baseline=True)

# Simpson integration
sig_simp = TimeSignalProcessing(t_acc, acc)
sig_simp.integrate(rule=IntegrationRules.SIMPSON, baseline=True)

# Analytical integration for comparison (integral of sin(2πt) is -cos(2πt)/(2π))
vel_analytical = -np.cos(2 * np.pi * 1 * t_acc) / (2 * np.pi)
vel_analytical = vel_analytical - np.mean(
    vel_analytical)  # baseline correction for comparison

plt.figure(figsize=(10, 6))
plt.plot(t_acc, vel_analytical, 'k-', label='Analytical solution')
plt.plot(sig_trap.time, sig_trap.signal, 'b--', label='Trapezoid rule')
plt.plot(sig_simp.time, sig_simp.signal, 'r:', label='Simpson rule')
plt.xlabel('Time [s]')
plt.ylabel('Velocity')
plt.title('Comparison of Integration Methods')
plt.grid()
plt.legend()
plt.show()

print("Trapezoid integration:")
print(sig_trap)
print("\nSimpson integration:")
print(sig_simp)
