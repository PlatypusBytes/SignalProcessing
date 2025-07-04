import numpy as np
import matplotlib.pylab as plt
from SignalProcessingTools.space_signal import SpaceSignalProcessing
import pickle

# Create test data
x = np.linspace(0, 100, 50001)
omega = 2 * np.pi * 6
y = 1.75 * np.sin(omega * x)
y_noise = y + 0.01 * np.sin(120 * x)

# Create a SpaceSignalProcessing object and demonstrate basic functionality
print("----------------------------------------------")
print("EXAMPLE 1: Track longitudinal level processing")
print("----------------------------------------------")
sig = SpaceSignalProcessing(x, y_noise)
print(f"Initialized SpaceSignalProcessing with {sig.n_points} points")
print(f"Sampling frequency: {sig.fs} Hz")

# Compute track longitudinal levels
sig.compute_track_longitudinal_levels()

# Plot the original signal and the decomposed components
plt.figure(figsize=(12, 10))

plt.subplot(5, 1, 1)
plt.plot(sig.coordinates, sig.signal_raw)
plt.xlabel('Distance [m]')
plt.ylabel('Amplitude')
plt.title('Original signal')
plt.grid()

plt.subplot(5, 1, 2)
plt.plot(sig.coordinates, sig.d0)
plt.xlabel('Distance [m]')
plt.ylabel('D0')
plt.title('D0 component (1m < λ <= 5m)')
plt.grid()

plt.subplot(5, 1, 3)
plt.plot(sig.coordinates, sig.d1)
plt.xlabel('Distance [m]')
plt.ylabel('D1')
plt.title('D1 component (3m < λ <= 25m)')
plt.grid()

plt.subplot(5, 1, 4)
plt.plot(sig.coordinates, sig.d2)
plt.xlabel('Distance [m]')
plt.ylabel('D2')
plt.title('D2 component (25m < λ <= 70m)')
plt.grid()

plt.subplot(5, 1, 5)
plt.plot(sig.coordinates, sig.d3)
plt.xlabel('Distance [m]')
plt.ylabel('D3')
plt.title('D3 component (70m < λ <= 150m)')
plt.grid()

plt.tight_layout()
plt.show()

# Example 2: Computing Hmax parameters
print("------------------------------------")
print("EXAMPLE 2: Computing Hmax parameters")
print("------------------------------------")

# Create a new signal with more complex spatial variations
x_track = np.linspace(0, 500, 25001)  # 500m of track
track_irregularity = (
    0.002 * np.sin(2 * np.pi * 0.1 * x_track) +  # Long wavelength component
    0.001 * np.sin(2 * np.pi * 0.2 * x_track) +  # Medium wavelength component
    0.0005 * np.sin(2 * np.pi * 0.4 * x_track) +  # Short wavelength component
    0.0002 * np.random.randn(len(x_track))  # Random noise
)

sig_hmax = SpaceSignalProcessing(x_track, track_irregularity)
# Compute Hmax parameters
sig_hmax.compute_Hmax(convert_m2mm=True)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.bar(range(1, len(sig_hmax.rms_bands) + 1), sig_hmax.rms_bands)
plt.xlabel('Frequency Band')
plt.ylabel('RMS [mm]')
plt.title('RMS Bands')
plt.grid(axis='y')
plt.xticks(range(1, len(sig_hmax.rms_bands) + 1))

plt.subplot(3, 1, 2)
plt.bar(range(1, len(sig_hmax.max_fast) + 1), sig_hmax.max_fast)
plt.xlabel('Frequency Band')
plt.ylabel('Max Fast [mm]')
plt.title('Maximum Effective Values')
plt.grid(axis='y')
plt.xticks(range(1, len(sig_hmax.max_fast) + 1))

plt.subplot(3, 1, 3)
plt.bar(range(1, len(sig_hmax.max_fast_Dx) + 1), sig_hmax.max_fast_Dx)
plt.xlabel('Frequency Band')
plt.ylabel('Max Fast Dx [mm]')
plt.title('Maximum Effective Values over Dx')
plt.grid(axis='y')
plt.xticks(range(1, len(sig_hmax.max_fast_Dx) + 1))

plt.tight_layout()
plt.show()
