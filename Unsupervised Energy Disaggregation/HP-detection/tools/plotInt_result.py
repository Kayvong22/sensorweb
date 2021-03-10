import calendar
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
import datetime
import numpy

ll_color = sns.cubehelix_palette(start=2.8, rot=.1)[-2:]
ll_color = [tuple(ll_color[i]) for i in range(len(ll_color))]


class plotInt():
    """Child class plotting the selected data (through Parent class
    Dict4plotInt) with the abillity to move along the month (slider below
    the graph)

    Attr:
        -- channels_dict: dictionary with the different key as channel. No signal
        channel as input. Only for comparison purpose. Nested keys appears on
        the same plot.

    Return:
        Interactive plots with slider on the bottom.

    """

    ll_apha = [1, 0.5]
    minwindow = 60 * 24  # sliders window, e.g. 24*60 for a day windows approx.
    yMaxAxis = 4000
    x_dt = 100
    test_1 = 0

    def __init__(self, channels_dict):
        self.channels_dict = channels_dict

    def suppl_param(self):
        """Supplementary parameters."""
        self.lkeys = list(self.channels_dict.keys())
        self.nbchannel = len(self.lkeys)

        # Nested status dictionary input
        try:
            self.channels_dict[self.lkeys[0]].keys()
            self.nested_dict_status = True
        except (KeyError, AttributeError):
            self.nested_dict_status = False

        # Updating the location of the subplot
        if isinstance(self.ax, (list, numpy.ndarray, tuple)) is not True:
            pass

        elif len(self.ax.shape) == 1:
            self.ll_loc_subplot = [(i, ) for i in range(self.ax.shape[0])]

        else:
            self.ll_loc_subplot = [(i, j) for i in range(self.ax.shape[0])
                                   for j in range(self.ax.shape[1])]

    def GenPlots(self, nrow, ncol):
        """Generate succesively all the subplots."""

        # Starting the plot
        self.fig, self.ax = plt.subplots(nrow, ncol)

        plt.subplots_adjust(left=0.05, bottom=0.07, right=0.95, top=1,
                            wspace=0.1, hspace=0.3)


    def minmax(self):
        """."""
        # Setting the min/max date for the SLIDER (located at the bottom of the plot)
        xmin = 0
        xmax = self.xmax

        return xmin, xmax

    def slider(self):
        """."""
        # Position and configure the slider at the bottom of the plot
        axpos = plt.axes([0.1, 0.01, 0.8, 0.03])  # slider position

        xmin, xmax = self.minmax()

        self.spos = Slider(axpos, '', xmin, xmax)

    def update(self, val):
        """Update function allows updating & moving the plot's windows."""
        pos = self.spos.val

        xmin_time = pos
        xmax_time = pos + self.x_dt

        if self.nested_dict_status == True:

            if isinstance(self.ax, (list, numpy.ndarray, tuple)) is not True:
                self.ax.axis([xmin_time, xmax_time +
                              self.x_dt, -50, self.yMaxAxis])

            else:
                for i, ll in enumerate(self.lkeys):
                    for j, mm in enumerate(self.channels_dict[ll].keys()):
                        self.ax[self.ll_loc_subplot[i]].axis([xmin_time, xmax_time + self.x_dt,
                                                              -50, self.yMaxAxis])

        elif self.nested_dict_status == False:

            if isinstance(self.ax, (list, numpy.ndarray, tuple)) is not True:
                self.ax.axis([xmin_time, xmax_time +
                              self.x_dt, -50, self.yMaxAxis])

            else:
                for i, ll in enumerate(self.lkeys):
                    self.ax[self.ll_loc_subplot[i]].axis([xmin_time, xmax_time + self.x_dt,
                                                          -50, self.yMaxAxis])

        self.fig.canvas.draw_idle()
