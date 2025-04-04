# SignalProcessing

![Tests](https://github.com/StemVibrations/STEM/actions/workflows/tests.yml/badge.svg)

A comprehensive Python package for time-domain signal processing operations with a focus on vibration analysis and frequency-domain transformations.

## Overview

SignalProcessing provides a suite of tools for analyzing, transforming, and processing time-series data.

Key features include:
* Fast Fourier Transforms (FFT) and inverse FFT
* Signal filtering (high-pass and low-pass)
* Integration methods (trapezoid, Simpson)
* Power Spectral Density (PSD) using Welch's method
* Spectrogram generation
* Effective velocity calculations using SBR method
* Windowing functions (Hann, Hamming, Blackman, etc.)

## Installation

### Install from source
```bash
git clone https://github.com/yourusername/SignalProcessing.git
cd SignalProcessing
pip install -e .
```

## Usage

### Basic Example

```python
import numpy as np
from SignalProcessing.time_signal import SignalProcessing, Windows

# Create a test signal
t = np.linspace(0, 10, 5001)
y = 1.75 * np.sin(2 * np.pi * 6 * t)

# Initialize the signal processor
sig = SignalProcessing(t, y)

# Perform FFT
sig.fft()

# Integrate the signal
sig.integrate(baseline=True, hp=True, fpass=1, n=6)
```

### Advanced Features

#### Windowed Processing and PSD
```python
# Create a signal processor with Hamming window
sig = SignalProcessing(t, y, window=Windows.HAMMING, window_size=4096)

# Calculate Power Spectral Density
sig.psd()

# Generate a spectrogram
sig.spectrogram()
```

#### Signal Filtering
```python
# Apply a low-pass filter to remove high frequency noise
sig.filter(10, 4, type_filter="lowpass")
```

#### Effective Velocity Calculation (SBR Method)
```python
# Calculate effective velocity using SBR method
sig.v_eff_SBR(n=4, tau=0.125)
```

## Example Files

A comprehensive example demonstrating all features is provided in [example.py](./example.py).


## License

This project is licensed under the MIT License - see the License file for details.

