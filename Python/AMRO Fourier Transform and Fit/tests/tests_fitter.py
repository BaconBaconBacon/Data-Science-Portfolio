from .utils.utils_fitter import *
from .utils.utils_fourier import *
from .utils.utils_loader import *


def _test_plot_sinebuilder():
    """ """
    # Test function
    f = [4, 2]
    amp = [2, 1]
    phase = [0, 0]
    offset = 1

    x = np.linspace(0, 2 * np.pi, 1000)
    y = self._sine_builder(x, amp, f, phase, offset)

    plt.scatter(x, y)
    return
