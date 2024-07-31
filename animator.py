import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
from matplotlib import pyplot as plt
from IPython import display
from matplotlib_inline import backend_inline
from torchvision.transforms import functional as F
# 定义Animator类和其他辅助函数
def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats("svg")

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """For plotting data in animation."""
    def __init__(
        self,
        xlabels=[None, None],
        ylabels=[None, None],
        legends=[None, None],
        xlims=[None, None],
        ylims=[None, None],
        xscales=["linear", "linear"],
        yscales=["linear", "linear"],
        fmts=["c--", "m", "g--", "r"],
        nrows=1,
        ncols=2,
        figsize=(10, 4),
    ):
        use_svg_display()
        if legends is None:
            legends = [[], []]
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        self.config_axes = lambda: (
            set_axes(
                self.axes[0],
                xlabels[0],
                ylabels[0],
                xlims[0],
                ylims[0],
                xscales[0],
                yscales[0],
                legends[0],
            ),
            set_axes(
                self.axes[1],
                xlabels[1],
                ylabels[1],
                xlims[1],
                ylims[1],
                xscales[1],
                yscales[1],
                legends[1],
            ),
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        self.axes[1].cla()
        for i, (x, y, fmt) in enumerate(zip(self.X, self.Y, self.fmts)):
            if i < 2:
                self.axes[0].plot(x, y, fmt)
            else:
                self.axes[1].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
