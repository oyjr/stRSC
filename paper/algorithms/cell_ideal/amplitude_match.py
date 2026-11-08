"""Derived fixed RNA-reliable amplitude; original RNA scaling is unchanged."""
import numpy as np


def match_prediction_amplitude(unit_prediction, mean_fullband_cross):
    """Use the 20-split mean full-band RNA cross-energy, fixed across splits.

    The mean is taken in the original per-split scaled RNA coordinates and is
    stored by the original calibration. Negative cross-energy gives zero amplitude while
    retaining the corresponding RNA cross/auto energies in subsequent pooling.
    """
    amplitude = np.sqrt(np.maximum(mean_fullband_cross, 0.0))
    return (unit_prediction * amplitude[None, :]).astype(np.float32)
