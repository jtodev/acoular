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
    """Test TimeConvolve kernel validation errors.

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


# @parametrize(
#     'kernel_shape,num_channels',
#     [((10,), 2), ((10, 1), 3)],
#     ids=['1d-kernel', 'broadcast-kernel'],
# )
# def test_time_convolve_kernel_broadcasting(kernel_shape, num_channels):
#     """Test TimeConvolve kernel broadcasting and reshaping.

#     Tests the TimeConvolve class from tprocess.py.
#     Covers:
#     - _validate_kernel() method for 1D kernels (automatic reshaping)
#     - Broadcasting behavior when kernel has shape (L, 1) and source has multiple channels
#     """
#     source = TimeSamples(sample_freq=51200, data=np.random.randn(100, num_channels))
#     kernel = np.random.randn(*kernel_shape)

#     conv = TimeConvolve(kernel=kernel, source=source)
#     res = tools.return_result(conv)

#     # Check that result has correct shape
#     assert res.shape[1] == num_channels

#     # Verify that the same kernel was applied to all channels
#     sig = tools.return_result(source)
#     for i in range(num_channels):
#         ref = np.convolve(conv.kernel[:, 0], sig[:, i])
#         np.testing.assert_allclose(res[:, i], ref[: res.shape[0]], rtol=1e-5, atol=1e-8)


# @parametrize(
#     'signal_samples,kernel_samples,block_size,extend_signal',
#     [
#         (10, 5, 128, True),  # very short signal (R==1)
#         (500, 200, 64, False),  # multiple kernel blocks (P>1)
#         (64, 64, 64, False),  # exact block alignment
#         (300, 50, 128, True),  # block_size=128
#         (300, 50, 64, True),  # block_size=64
#         (300, 50, 32, True),  # block_size=32
#     ],
#     ids=['short-signal', 'long-kernel', 'exact-alignment', 'num-128', 'num-64', 'num-32'],
# )
# def test_time_convolve_edge_cases(signal_samples, kernel_samples, block_size, extend_signal):
#     """Test TimeConvolve with various edge cases.

#     Tests the TimeConvolve class from tprocess.py.
#     Covers:
#     - Very short signals (R == 1 case) - early return after single block
#     - Multiple kernel blocks (P > 1) - kernel splitting in _get__kernel_blocks()
#     - Exact block alignment (last_size == 0) - final_len calculation
#     - Different block sizes - various paths through result() method
#     """
#     source = TimeSamples(sample_freq=51200, data=np.random.randn(signal_samples, 2))
#     kernel = np.random.randn(kernel_samples, 2)

#     conv = TimeConvolve(kernel=kernel, source=source, extend_signal=extend_signal)
#     res = tools.return_result(conv, num=block_size)

#     # Verify correctness against numpy convolve
#     sig = tools.return_result(source)
#     for i in range(2):
#         ref = np.convolve(kernel[:, i], sig[:, i])
#         expected_len = len(ref) if extend_signal else max(signal_samples, kernel_samples)
#         np.testing.assert_allclose(res[:, i], ref[:expected_len], rtol=1e-5, atol=1e-8)
