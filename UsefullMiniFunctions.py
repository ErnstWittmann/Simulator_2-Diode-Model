import matplotlib.pyplot as plt
from contextlib import contextmanager
import time as time_lib

# Usefull mini_functions
def plt_U_I(ax=None):
    if ax is None:
        plt.xlabel("U/V", fontsize=16)
        plt.ylabel("I/A", fontsize=16)
        plt.grid()
        plt.tick_params(axis="both", labelsize=16)
    else:
        ax.set_xlabel("U/V", fontsize=16)
        ax.set_ylabel("I/A", fontsize=16)
        ax.grid()
        ax.tick_params(axis="both", labelsize=16)
def plt_U_P(ax=None):
    if ax is None:
        plt.xlabel("U/V", fontsize=16)
        plt.ylabel("P/W", fontsize=16)
        plt.grid()
        plt.tick_params(axis="both", labelsize=16)
    else:
        ax.set_xlabel("U/V", fontsize=16)
        ax.set_ylabel("P/W", fontsize=16)
        ax.grid()
        ax.tick_params(axis="both", labelsize=16)
def plt_T_U(ax=None):
    if ax is None:
        plt.xlabel("time", fontsize=16)
        plt.ylabel("U/V", fontsize=16)
        plt.grid()
        plt.tick_params(axis="both", labelsize=16)
        plt.tick_params(axis="x", rotation=45)
    else:
        ax.set_xlabel("time", fontsize=16)
        ax.set_ylabel("U/V", fontsize=16)
        ax.grid()
        ax.tick_params(axis="both", labelsize=16)
        ax.tick_params(axis="x", rotation=45)
def plt_T_I(ax=None):
    if ax is None:
        plt.xlabel("time", fontsize=16)
        plt.ylabel("I/A", fontsize=16)
        plt.grid()
        plt.tick_params(axis="both", labelsize=16)
        plt.tick_params(axis="x", rotation=45)
    else:
        ax.set_xlabel("time", fontsize=16)
        ax.set_ylabel("I/A", fontsize=16)
        ax.grid()
        ax.tick_params(axis="both", labelsize=16)
        ax.tick_params(axis="x", rotation=45)
def plt_T_P(ax=None):
    if ax is None:
        plt.xlabel("time", fontsize=16)
        plt.ylabel("P/W", fontsize=16)
        plt.grid()
        plt.tick_params(axis="both", labelsize=16)
        plt.tick_params(axis="x", rotation=45)
    else:
        ax.set_xlabel("time", fontsize=16)
        ax.set_ylabel("P/W", fontsize=16)
        ax.grid()
        ax.tick_params(axis="both", labelsize=16)
        ax.tick_params(axis="x", rotation=45)


@contextmanager
def _log_time_usage(prefix=""):
    '''log the time usage in a code block
    prefix: the prefix text to show
    '''
    start = time_lib.time()
    try:
        yield
    finally:
        end = time_lib.time()
        elapsed_seconds = float("%.5f" % (end - start))
        print(f'{prefix}: elapsed seconds: {elapsed_seconds}s')