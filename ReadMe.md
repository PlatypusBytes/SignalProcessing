# Signal processing tools

Package to perform operation in time signals.

The code includes:

* processing
* window
* spectral subtraction

## processing
Processing creates an object that performs operations in a signal. 
The supported operations are:

* FFT
* Inverse of FFT
* Filtering (high and low-pass)
* Integration

## window
Creates a moving window over a time history signal and performs operations. 
The available windows have an overlap of 50% (currently supported: "Hann", "Barthann", "Bartlett" and "Triang"). 
The supported operations for signal processing are:

* FFT
* Integration
* Filter
* Plot spectrogram

## spectral subtraction
To be done
