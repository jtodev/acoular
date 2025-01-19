# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Implements signal generators for the simulation of acoustic sources.

.. autosummary::
    :toctree: generated/

    SignalGenerator
    WNoiseGenerator
    PNoiseGenerator
    FiltWNoiseGenerator
    SineGenerator
    GenericSignalGenerator

"""

# imports from other packages
from abc import abstractmethod
from warnings import warn

from numpy import arange, array, log, pi, repeat, sin, sqrt, tile, zeros
from numpy.random import RandomState
from scipy.signal import resample, sosfilt, tf2sos
from traits.api import (
    ABCHasStrictTraits,
    Bool,
    CArray,
    CLong,
    Delegate,
    Float,
    Instance,
    Int,
    Property,
    cached_property,
)

# acoular imports
from .base import SamplesGenerator
from .deprecation import deprecated_alias
from .internal import digest


@deprecated_alias({'numsamples': 'num_samples'})
class SignalGenerator(ABCHasStrictTraits):
    """
    ABC for a simple one-channel signal generator.

    This ABC defines the common interface and attributes for all signal generator implementations.
    It provides a template for generating one-channel signals with specified amplitude,
    sampling frequency, and duration. Subclasses should implement the core functionality,
    including signal generation and computation of the internal identifier.

    Notes
    -----
    This class should not be instantiated directly. Instead, use a subclass that
    implements the required methods for signal generation.

    See Also
    --------
    :func:`scipy.signal.resample` : Used for resampling signals in the :meth:`usignal` method.
    """

    #: Root mean square (RMS) amplitude of the signal. For a point source,
    #: this corresponds to the RMS amplitude at a distance of 1 meter. Default is ``1.0``.
    rms = Float(1.0, desc='rms amplitude')

    #: Sampling frequency of the signal in Hz. Default is ``1.0``.
    sample_freq = Float(1.0, desc='sampling frequency')

    #: The number of samples to generate for the signal.
    num_samples = CLong

    #: A unique identifier based on the generator properties. (read-only)
    digest = Property(depends_on=['rms', 'sample_freq', 'num_samples'])

    @abstractmethod
    def _get_digest(self):
        """Returns the internal identifier."""

    @abstractmethod
    def signal(self):
        """
        Generate and return the signal.

        This method must be implemented by subclasses to provide the generated signal
        as a 1D array of samples.
        """

    def usignal(self, factor):
        """
        Resample the signal at a higher sampling frequency.

        This method uses Fourier transform-based resampling to deliver the signal at a
        sampling frequency that is a multiple of the original :attr:`sample_freq`.
        The resampled signal has a length of ``factor * num_samples``.

        Parameters
        ----------
        factor : int
            The resampling factor. Defines how many times larger the new sampling frequency is
            compared to the original :attr:`sample_freq`.

        Returns
        -------
        :class:`numpy.ndarray`
            The resampled signal as a 1D array of floats.

        Notes
        -----
        This method relies on the :func:`scipy.signal.resample` function for resampling.

        Examples
        --------
        Resample a signal by a factor of 4:

        >>> from acoular import SineGenerator  # Class extending SignalGenerator
        >>> sg = SineGenerator(rms=1.0, sample_freq=100.0, num_samples=1000)
        >>> resampled_signal = sg.usignal(4)
        >>> len(resampled_signal)
        4000
        """
        return resample(self.signal(), factor * self.num_samples)


class WNoiseGenerator(SignalGenerator):
    """
    Generate White noise signal.

    This class generates white noise signals with a specified
    :attr:`root mean square (RMS)<SignalGenerator.rms>` amplitude,
    :attr:`number of samples<SignalGenerator.num_samples>`, and
    :attr:`sampling frequency<SignalGenerator.sample_freq>`. The white noise is generated using a
    :obj:`random number generator<numpy.random.RandomState.standard_normal>` initialized with a
    :attr:`user-defined seed<seed>` for reproducibility.

    See Also
    --------
    :obj:`numpy.random.RandomState.standard_normal` :
        Used here to generate normally distributed noise.
    :class:`acoular.signals.PNoiseGenerator` : For pink noise generation.
    :class:`acoular.sources.UncorrelatedNoiseSource` : For per-channel noise generation.
    """

    #: Seed for random number generator. Default is ``0``.
    #: This parameter should be set differently for different instances
    #: to guarantee statistically independent (non-correlated) outputs.
    seed = Int(0, desc='random seed value')

    #: Internal identifier based on generator properties. (read-only)
    digest = Property(depends_on=['rms', 'num_samples', 'sample_freq', 'seed'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        """
        Generate and deliver the white noise signal.

        The signal is created using a Gaussian distribution with mean 0 and variance 1,
        scaled by the :attr:`RMS<SignalGenerator.rms>` amplitude of the generator.

        Returns
        -------
        :class:`numpy.ndarray`
            A 1D array of floats containing the generated white noise signal.
            The length of the array is equal to :attr:`~SignalGenerator.num_samples`.

        Examples
        --------
        Generate white noise with an RMS amplitude of 1.0 and 0.5:

        >>> from acoular import WNoiseGenerator
        >>> from numpy import mean
        >>>
        >>> # White noise with RMS of 1.0
        >>> gen1 = WNoiseGenerator(rms=1.0, num_samples=1000, seed=42)
        >>> signal1 = gen1.signal()
        >>>
        >>> # White noise with RMS of 0.5
        >>> gen2 = WNoiseGenerator(rms=0.5, num_samples=1000, seed=24)
        >>> signal2 = gen2.signal()
        >>>
        >>> mean(signal1) > mean(signal2)
        np.True_

        Ensure different outputs with different seeds:

        >>> gen1 = WNoiseGenerator(num_samples=3, seed=42)
        >>> gen2 = WNoiseGenerator(num_samples=3, seed=73)
        >>> gen1.signal() == gen2.signal()
        array([False, False, False])
        """
        rnd_gen = RandomState(self.seed)
        return self.rms * rnd_gen.standard_normal(self.num_samples)


class PNoiseGenerator(SignalGenerator):
    """
    Generate pink noise signal.

    The :class:`PNoiseGenerator` class generates pink noise signals,
    which exhibit a :math:`1/f` power spectral density. Pink noise is characterized by
    equal energy per octave, making it useful in various applications such as audio testing,
    sound synthesis, and environmental noise simulations.

    The pink noise simulation is based on the Voss-McCartney algorithm, which iteratively adds
    noise with increasing wavelength to achieve the desired :math:`1/f` characteristic.

    References
    ----------
    - S.J. Orfanidis: Signal Processing (2010), pp. 729-733 :cite:`Orfanidis2010`
    - Online discussion: http://www.firstpr.com.au/dsp/pink-noise/

    See Also
    --------
    :class:`acoular.signals.WNoiseGenerator` : For white noise generation.
    :class:`acoular.sources.UncorrelatedNoiseSource` : For per-channel noise generation.
    """

    #: Seed for the random number generator. Changing this value ensures statistically independent
    #: (non-correlated) outputs across different instances. Default is ``0``.
    seed = Int(0, desc='random seed value')

    #: "Octave depth" of the pink noise generation. Higher values result in a better approximation
    #: of the :math:`1/f` spectrum at low frequencies but increase computation time. The  maximum
    #: allowable value depends on the :attr:`number of samples<SignalGenerator.num_samples>`.
    #: Default is ``16``.
    depth = Int(16, desc='octave depth')

    # A unique identifier based on the generator properties. (read-only)
    digest = Property(depends_on=['rms', 'num_samples', 'sample_freq', 'seed', 'depth'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        """
        Generate and deliver the pink noise signal.

        The signal is computed using the Voss-McCartney algorithm, which generates noise
        with a :math:`1/f` power spectral density. The method ensures that the output has the
        desired :attr:`RMS<SignalGenerator.rms>` amplitude and spectrum.

        Returns
        -------
        :class:`numpy.ndarray`
            A 1D array of floats containing the generated pink noise signal. The length
            of the array is equal to :attr:`~SignalGenerator.num_samples`.

        Notes
        -----
        - The "depth" parameter controls the number of octaves included in the pink noise
          simulation. If the specified depth exceeds the maximum possible value based on
          the number of samples, it is automatically adjusted, and a warning is printed.
        - The output signal is scaled to have the same overall level as white noise by dividing
          the result by ``sqrt(depth + 1.5)``.
        """
        nums = self.num_samples
        depth = self.depth
        # maximum depth depending on number of samples
        max_depth = int(log(nums) / log(2))

        if depth > max_depth:
            depth = max_depth
            print(f'Pink noise filter depth set to maximum possible value of {max_depth:d}.')

        rnd_gen = RandomState(self.seed)
        s = rnd_gen.standard_normal(nums)
        for _ in range(depth):
            ind = 2**_ - 1
            lind = nums - ind
            dind = 2 ** (_ + 1)
            s[ind:] += repeat(rnd_gen.standard_normal(nums // dind + 1), dind)[:lind]
        # divide by sqrt(depth+1.5) to get same overall level as white noise
        return self.rms / sqrt(depth + 1.5) * s


class FiltWNoiseGenerator(WNoiseGenerator):
    """Filtered white noise signal following an autoregressive (AR), moving-average
    (MA) or autoregressive moving-average (ARMA) process.

    The desired frequency response of the filter can be defined by specifying
    the filter coefficients :attr:`ar` and :attr:`ma`.
    The RMS value specified via the :attr:`rms` attribute belongs to the white noise
    signal and differs from the RMS value of the filtered signal.
    For numerical stability at high orders, the filter is a combination of second order
    sections (sos).
    """

    ar = CArray(value=array([]), dtype=float, desc='autoregressive coefficients (coefficients of the denominator)')

    ma = CArray(value=array([]), dtype=float, desc='moving-average coefficients (coefficients of the numerator)')

    # internal identifier
    digest = Property(
        depends_on=[
            'ar',
            'ma',
            'rms',
            'num_samples',
            'sample_freq',
            'seed',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def handle_empty_coefficients(self, coefficients):
        if coefficients.size == 0:
            return array([1.0])
        return coefficients

    def signal(self):
        """Deliver the signal.

        Returns
        -------
        Array of floats
            The resulting signal as an array of length :attr:`~SignalGenerator.num_samples`.
        """
        rnd_gen = RandomState(self.seed)
        ma = self.handle_empty_coefficients(self.ma)
        ar = self.handle_empty_coefficients(self.ar)
        sos = tf2sos(ma, ar)
        ntaps = ma.shape[0]
        sdelay = round(0.5 * (ntaps - 1))
        wnoise = self.rms * rnd_gen.standard_normal(
            self.num_samples + sdelay,
        )  # create longer signal to compensate delay
        return sosfilt(sos, x=wnoise)[sdelay:]


class SineGenerator(SignalGenerator):
    """Sine signal generator with adjustable frequency and phase."""

    #: Sine wave frequency, float, defaults to 1000.0.
    freq = Float(1000.0, desc='Frequency')

    #: Sine wave phase (in radians), float, defaults to 0.0.
    phase = Float(0.0, desc='Phase')

    #: Amplitude of source signal (for point source: in 1 m distance).
    #: Defaults to 1.0.
    amplitude = Float(1.0)

    # internal identifier
    digest = Property(depends_on=['_amp', 'num_samples', 'sample_freq', 'freq', 'phase'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        """Deliver the signal.

        Returns
        -------
        array of floats
            The resulting signal as an array of length :attr:`~SignalGenerator.num_samples`.
        """
        t = arange(self.num_samples, dtype=float) / self.sample_freq
        return self.amplitude * sin(2 * pi * self.freq * t + self.phase)


class GenericSignalGenerator(SignalGenerator):
    """Generate signal from output of :class:`~acoular.base.SamplesGenerator` object.

    This class can be used to inject arbitrary signals into Acoular processing
    chains. For example, it can be used to read signals from a HDF5 file or create any signal
    by using the :class:`acoular.sources.TimeSamples` class.

    Example
    -------
    >>> import numpy as np
    >>> from acoular import TimeSamples, GenericSignalGenerator
    >>> data = np.random.rand(1000, 1)
    >>> ts = TimeSamples(data=data, sample_freq=51200)
    >>> sig = GenericSignalGenerator(source=ts)
    """

    #: Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')

    _num_samples = CLong(0)

    #: Number of samples to generate. Is set to source.num_samples by default.
    num_samples = Property()

    def _get_num_samples(self):
        if self._num_samples:
            return self._num_samples
        return self.source.num_samples

    def _set_num_samples(self, num_samples):
        self._num_samples = num_samples

    #: Boolean flag, if 'True' (default), signal track is repeated if requested
    #: :attr:`num_samples` is higher than available sample number
    loop_signal = Bool(True)

    # internal identifier
    digest = Property(
        depends_on=['source.digest', 'loop_signal', 'num_samples', 'rms'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        """Deliver the signal.

        Returns
        -------
        array of floats
            The resulting signal as an array of length :attr:`~GenericSignalGenerator.num_samples`.

        """
        block = 1024
        if self.source.num_channels > 1:
            warn(
                'Signal source has more than one channel. Only channel 0 will be used for signal.',
                Warning,
                stacklevel=2,
            )
        nums = self.num_samples
        track = zeros(nums)

        # iterate through source generator to fill signal track
        for i, temp in enumerate(self.source.result(block)):
            start = block * i
            stop = start + len(temp[:, 0])
            if nums > stop:
                track[start:stop] = temp[:, 0]
            else:  # exit loop preliminarily if wanted signal samples are reached
                track[start:nums] = temp[: nums - start, 0]
                break

        # if the signal should be repeated after finishing and there are still samples open
        if self.loop_signal and (nums > stop):
            # fill up empty track with as many full source signals as possible
            nloops = nums // stop
            if nloops > 1:
                track[stop : stop * nloops] = tile(track[:stop], nloops - 1)
            # fill up remaining empty track
            res = nums % stop  # last part of unfinished loop
            if res > 0:
                track[stop * nloops :] = track[:res]

        # The rms value is just an amplification here
        return self.rms * track
