from functools import wraps

from nose import SkipTest


def if_matplotlib(func):
    """Test decorator that skips test if matplotlib not installed.

    Parameters
    ----------
    func
    """
    @wraps(func)
    def run_test(*args, **kwargs):
        try:
            import matplotlib
            matplotlib.use('Agg', warn=False)
            # this fails if no $DISPLAY specified
            import matplotlib.pyplot as plt
            plt.figure()
        except ImportError:
            raise SkipTest('Matplotlib not available.')
        else:
            return func(*args, **kwargs)
    return run_test
