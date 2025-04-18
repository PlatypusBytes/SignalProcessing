# SignalProcessingTools

![Tests](https://github.com//PlatypusBytes/SignalProcessing/actions/workflows/tests.yml/badge.svg)
[![PyPI version](https://img.shields.io/pypi/v/SignalProcessingTools.svg)](https://pypi.org/project/SignalProcessingTools/)
[![Python versions](https://img.shields.io/pypi/pyversions/SignalProcessingTools.svg)](https://pypi.org/project/SignalProcessingTools/)


A comprehensive Python package for time and space domain signal processing operations with a focus on vibration analysis and frequency-domain transformations.
The space domain operations focus on railway applications, while the time domain operations are more general.

## Overview

SignalProcessingTools provides a suite of tools for analyzing, transforming, and processing  data.

### Time domain operations:
* Fast Fourier Transforms (FFT) and inverse FFT
* Signal filtering
* Integration
* Power Spectral Density (PSD) using Welch's method
* Spectrogram generation
* Effective velocity calculations using SBR method
* 1/3 octave band analysis
* Windowing functions (Hann, Hamming, Blackman, etc.)


### Space domain operations:
* D0, D1, D2, and D3 track longitudinal levels, following EN 13848-1:2006.
* Hmax and Hrms according to Zandberg et al. (2022).

## Installation

### Install from PyPI
You can install the package directly from PyPI using pip:

```bash
pip install SignalProcessingTools
```

### Install from Source
To install the package from the source, clone the repository and run the following commands:

```bash
git clone https://github.com/PlatypusBytes/SignalProcessing.git
cd SignalProcessing
pip install -e .
```

## Usage

### Basic Example of Time Domain Operations

#### FFT and signal integration
```python
import numpy as np
from SignalProcessingTools.time_signal import SignalProcessing, Windows

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

#### Windowed Processing and PSD and spectrogram

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

#### Effective Velocity Calculation (SBR-B Method)

```python
# Calculate effective velocity using SBR method
sig.v_eff_SBR()
```

### Basic Example of Spatial Domain Operations

#### D0, D1, D2, and D3 Calculation

```python
import numpy as np
from SignalProcessingTools.space_signal import SpatialSignal
from SignalProcessingTools.space_signal import EN13848

# Create test data
x = np.linspace(0, 100, 50001)
omega = 2 * np.pi * 6
y = 1.75 * np.sin(omega * x)
y_noise = y + 0.01 * np.sin(120 * x)

sig = SpaceSignalProcessing(x, y_noise)
# Compute track longitudinal levels
sig.compute_track_longitudinal_levels()
```

#### Hmax and Hrms Calculation

```python
x_track = np.linspace(0, 500, 25001)
track_irregularity = (
    0.002 * np.sin(2 * np.pi * 0.1 * x_track) +
    0.001 * np.sin(2 * np.pi * 0.2 * x_track) +
    0.0005 * np.sin(2 * np.pi * 0.4 * x_track) +
    0.0002 * np.random.randn(len(x_track))
)

sig_hmax = SpaceSignalProcessing(x_track, track_irregularity)
# Compute Hmax parameters
sig_hmax.compute_Hmax(convert_m2mm=True)
```

## Example Files

A comprehensive example demonstrating all features is provided for the [time signal](./example_time_signal.py) and [space signal](./example_space_signal.py).


## License

This project is licensed under the MIT License - see the License file for details.

