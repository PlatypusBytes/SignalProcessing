import pytest
import json
import numpy as np

from SignalProcessingTools.space_signal import SpaceSignalProcessing


@pytest.fixture
def test_data():
    """
    Create test data
    """
    with open("./tests/data/track_alignment.txt", "r") as fi:
        sig = fi.read().splitlines()

    sig = list(map(float, sig))
    dx = 0.25
    x = np.arange(0, len(sig) * dx, dx)

    return x, sig


def test_track_quality_index(test_data):
    """
    Test the track quality index function
    """
    x, level = test_data

    sig = SpaceSignalProcessing(np.array(x), np.array(level))
    sig.compute_track_longitudinal_levels()

    with open("./tests/data/track_alignment_results.txt", "r") as fi:
        data = json.load(fi)

    assert np.allclose(sig.coordinates, data["coordinates"], rtol=1e-5, atol=1e-8)
    assert np.allclose(sig.d0, data["D0"], rtol=1e-5, atol=1e-8)
    assert np.allclose(sig.d1, data["D1"], rtol=1e-5, atol=1e-8)
    assert np.allclose(sig.d2, data["D2"], rtol=1e-5, atol=1e-8)
    assert np.allclose(sig.d3, data["D3"], rtol=1e-5, atol=1e-8)

def test_Hmax(test_data):
    """
    Test the Hmax function
    """
    x, level = test_data

    sig = SpaceSignalProcessing(np.array(x), np.array(level))

    sig.compute_Hmax()


    rms_band_matlab = np.array([2087.55705457531, 1139.29553877343, 793.047457091564, 548.095561097181,
                                648.015015656438, 521.608760233929, 790.563948013129, 886.097705321285,
                                1342.44544888507])
    h_max_matlab = np.array([6557.20346546820, 3764.49040821894, 2505.54201229720, 1727.21064286318,
                                1208.73808675389, 1182.77849807824, 1996.50964604940, 2682.54554936051,
                                3681.41028314498])
    h_max_dx_matlab = np.array([293.263231128877, 783.641358934000, 1300.63422139907, 524.680132894043,
                                139.978582308885, 583.726452878631, 662.410999843909, 1240.10908108192,
                                1127.56942541745])



    assert np.allclose(sig.rms_bands, rms_band_matlab, rtol=1e-3, atol=1e-8)
    assert np.allclose(sig.max_fast, h_max_matlab, rtol=1e-3, atol=1e-8)
    assert np.allclose(sig.max_fast_Dx, h_max_dx_matlab, rtol=1e-3, atol=1e-8)