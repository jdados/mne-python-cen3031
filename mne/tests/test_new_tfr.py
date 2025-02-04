import numpy as np
import pytest

def _tfr_from_mt_old(x_mt, weights):
    weights = weights[np.newaxis, :, :, np.newaxis]
    tfr = weights * x_mt
    tfr *= tfr.conj()
    tfr = tfr.real.sum(axis=1)
    tfr *= 2 / (weights * weights.conj()).real.sum(axis=1)
    return tfr

def _tfr_from_mt_new(x_mt, weights):
    weights = weights[np.newaxis, ..., np.newaxis] # match dimensions to avoid broadcasting errors
    tfr = weights * x_mt
    tfr *= tfr.conj()
    tfr = tfr.real.sum(axis=-3)
    tfr *= 2 / (weights * weights.conj()).real.sum(axis=-3) #summing axis = -3 to avoid broadcasting errors
    return tfr

def random_data(shape):
    return np.random.randn(*shape) + 1j * np.random.randn(*shape)

# non-epoched data
n_channels, n_tapers, n_freqs, n_times = 2, 3, 5, 4
x_mt = random_data((n_channels, n_tapers, n_freqs, n_times))
print("random eeg data:", x_mt)
weights = np.random.rand(n_tapers, n_freqs)

result_old = _tfr_from_mt_old(x_mt, weights)
result_new = _tfr_from_mt_new(x_mt, weights)

print("non-epoched data shape:", result_new.shape)

# epoched data
n_epochs = 4
x_mt_epoched = random_data((n_epochs, n_channels, n_tapers, n_freqs, n_times))

# compare
result_old_epoched = np.array([_tfr_from_mt_old(epo_x, weights) for epo_x in x_mt_epoched])
result_new_epoched = _tfr_from_mt_new(x_mt_epoched, weights)

print("epoched data shape:", result_new_epoched.shape)

