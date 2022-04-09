import matplotlib.pyplot as mpl


class DataVisualizer():
    def __init__(self):
        pass

    @staticmethod
    def twod_plot_versus(x, y, title, xlabel, ylabel):
        if min(x) > 0 or min(x) == 0:
            x_min = -5
        else:
            x_min = min(x) - 2

        # Set y-axis lower bound
        if min(y) > 0 or min(y) == 0:
            y_min = -5
        else:
            y_min = min(y) - 2

        fig = mpl.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        mpl.rcParams["figure.autolayout"] = True
        mpl.axvline(x=0, c="red", label="x=0")
        mpl.axhline(y=0, c="yellow", label="y=0")
        mpl.scatter(x, y, s=100, alpha=0.5, color="blue")
        mpl.title(title, fontsize=20)
        mpl.xlabel(xlabel, fontsize=16)
        mpl.ylabel(ylabel, fontsize=16)
        mpl.xlim(x_min, max(x) + 1)
        mpl.ylim(y_min, max(y) + 1)
        mpl.grid()
        mpl.show()
