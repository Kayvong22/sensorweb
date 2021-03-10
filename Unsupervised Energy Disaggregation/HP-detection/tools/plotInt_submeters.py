import calendar
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd
import datetime

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

    # PLOTTING PARAMETERS

    ll_apha = [1, 0.5]
    minwindow = 60 * 24  # sliders window, e.g. 24*60 for a day windows approx.
    yMaxAxis = 4000
    x_dt = 1440
    test_1 = 0

    def __init__(self, channels_dict):
        self.channels_dict = channels_dict

    def suppl_param(self):
        """Supplementary parameters."""
        self.lkeys = list(self.channels_dict.keys())
        self.nbchannel = len(self.lkeys)

        if len(self.ax.shape) == 1:
            self.ll_loc_subplot = [(i, ) for i in range(self.ax.shape[0])]

        else:
            self.ll_loc_subplot = [(i, j) for i in range(self.ax.shape[0])
                                   for j in range(self.ax.shape[1])]

    def PlotMyPlot(self, channel, ax):
        """Compute the plot with a given Dictionary element and subplot."""

        x = channel

        ax.plot(x,
                linewidth=linewid,
                alpha=alpha,
                label=str(label),
                color=col
                )  # add the label
        ax.set(ylabel='Active Power [W]',
               xlabel='Time'
               )
        ax.legend(loc=1)

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
        # spos = Slider(axpos, ''.join([calendar.month_abbr[MONTH], '17']), matplotlib.dates.date2num(xmin),
        #               matplotlib.dates.date2num(xmax))

        xmin, xmax = self.minmax()

        # self.spos = Slider(axpos, ''.join([calendar.month_abbr[self.MONTH], '17']),
        # datetime_to_float(xmin), datetime_to_float(xmax))

        self.spos = Slider(axpos, '', xmin, xmax)

        # make the dates nicer, nicer better
        # plt.gcf().autofmt_xdate()

    def update(self, val):
        """Update function allows updating & moving the plot's windows."""
        pos = self.spos.val

        # xmin_time = float_to_datetime(pos)
        # xmax_time = float_to_datetime(pos) + self.x_dt

        xmin_time = pos
        xmax_time = pos + self.x_dt

        if len(self.ax.shape) == 1:
            for i in self.ll_loc_subplot:
                self.ax[i].axis([xmin_time, xmax_time +
                                 self.x_dt, -50, self.yMaxAxis])

        else:
            for i in self.ll_loc_subplot:
                self.ax[i].axis([xmin_time, xmax_time +
                                 self.x_dt, -50, self.yMaxAxis])

        self.fig.canvas.draw_idle()

    def fullprocess(self, yaxis_max=4000, delta_xaxis=1440):
        """."""

        self.yMaxAxis = yaxis_max
        self.x_dt = delta_xaxis

        self.GenPlots()

        self.suppl_param()

        self.slider()

        self.spos.on_changed(self.update)


# inst = plotInt(1, 2)
# inst.dict()
# inst.suppl_param()
# inst.GenPlots()
# inst.slider()
# inst.spos.on_changed(inst.update)
#
# plt.show()

# OR ##

# inst = plotInt(1, 2)
# inst.fullprocess()
