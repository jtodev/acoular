# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Test cases for all convolve classes."""

import numpy as np
import pytest
from acoular import TimeConvolve, TimeSamples, tools
from pytest_cases import parametrize


@parametrize('extend_signal', [True, False], ids=['extend_signal-True', 'extend_signal-False'])
def test_time_convolve(time_data_source, extend_signal):
    """Compare results of timeconvolve with numpy convolve.

    Parameters
    ----------
    time_data_source : instance of acoular.sources.TimeSamples
        TimeSamples instance to be tested (see time_data_source fixture in conftest.py)
    """
    sig = tools.return_result(time_data_source)
    nc = time_data_source.num_channels
    kernel = np.random.rand(20 * nc).reshape(20, nc)
    conv = TimeConvolve(kernel=kernel, source=time_data_source, extend_signal=extend_signal)
    res = tools.return_result(conv)
    for i in range(time_data_source.num_channels):
        ref = np.convolve(np.squeeze(conv.kernel[:, i]), np.squeeze(sig[:, i]))
        np.testing.assert_allclose(np.squeeze(res[:, i]), ref[: res.shape[0]], rtol=1e-5, atol=1e-8)


@parametrize(
    'randn_params,error_msg',
    [
        ((10, 2, 2), 'Only one or two dimensional kernels accepted'),
        ((10, 3), 'Number of kernels must be either'),
    ],
    ids=['3d-kernel', 'mismatched-channels'],
)
def test_time_convolve_kernel_validation_errors(randn_params, error_msg):
    """
    Test TimeConvolve kernel validation errors.

    Tests the TimeConvolve class from tprocess.py.
    Covers the _validate_kernel() method:
    - Raises ValueError for kernels with more than 2 dimensions
    - Raises ValueError when kernel channel count doesn't match source
    """
    source = TimeSamples(sample_freq=51200, data=np.random.randn(100, 2))
    kernel = np.random.randn(*randn_params)
    conv = TimeConvolve(kernel=kernel, source=source)

    with pytest.raises(ValueError, match=error_msg):
        _ = tools.return_result(conv)


def test_time_convolve_output_blocks_exceed_signal_blocks():
    """
    Test TimeConvolve when output blocks exceed signal blocks (R > Q).

    Tests the TimeConvolve class from tprocess.py.
    Covers the for-loop: `for _ in range(R - Q):` in the result() method.
    This path is triggered when extend_signal=True and the convolution output
    requires more blocks than the input signal provides, necessitating additional
    processing iterations with zero-padded buffers.

    Specifically covers lines in result() method:
    - Loop iteration when R - Q > 0
    - _append_to_fdl() calls with zero-padded buffers
    - _spectral_sum() computation for tail blocks
    - Buffer shifting with zero concatenation
    - Yielding of tail blocks after signal exhausted
    """
    # Design parameters to ensure R > Q:
    # - Use extend_signal=True so output_size = L + M - 1
    # - Make kernel long relative to signal so R (output blocks) > Q (signal blocks)

    signal_samples = 100  # M = 100
    kernel_samples = 150  # L = 150
    block_size = 64  # num = 64

    # With extend_signal=True:
    # output_size = L + M - 1 = 150 + 100 - 1 = 249
    # Q = ceil(M / num) = ceil(100 / 64) = 2 (signal blocks)
    # R = ceil(output_size / num) = ceil(249 / 64) = 4 (output blocks)
    # R - Q = 4 - 2 = 2, so loop executes 2 times

    source = TimeSamples(sample_freq=51200, data=np.random.randn(signal_samples, 2))
    kernel = np.random.randn(kernel_samples, 2)

    conv = TimeConvolve(kernel=kernel, source=source, extend_signal=True)

    # Manually iterate to ensure all blocks are generated
    blocks = list(conv.result(num=block_size))

    # Should have R = 4 blocks total
    assert len(blocks) == 4, f'Expected 4 blocks, got {len(blocks)}'

    # Verify correctness against numpy convolve
    res = np.vstack(blocks)
    sig = tools.return_result(source)

    for i in range(2):
        ref = np.convolve(kernel[:, i], sig[:, i], mode='full')
        np.testing.assert_allclose(res[:, i], ref, rtol=1e-5, atol=1e-8)
