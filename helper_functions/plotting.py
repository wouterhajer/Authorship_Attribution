import lir
from lir.calibration import IsotonicCalibrator
from lir import util
from lir.ece import plot_ece as ece
import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager
from functools import partial
from lir.bayeserror import plot_nbe as nbe

# Copied from https://github.com/NetherlandsForensicInstitute/lir/blob/master/lir/_plotting_new.py
# to modify some labels in plot to the variables used in the corresponding thesis

class Canvas:
    def __init__(self, ax):
        self.ax = ax

        self.calibrator_fit = partial(calibrator_fit, ax=ax)
        self.ece = partial(ece, ax=ax)
        self.lr_histogram = partial(lr_histogram, ax=ax)
        self.nbe = partial(nbe, ax=ax)
        self.pav = partial(pav, ax=ax)
        self.score_distribution = partial(score_distribution, ax=ax)
        self.tippett = partial(tippett, ax=ax)

    def __getattr__(self, attr):
        return getattr(self.ax, attr)

def savefig(path):
    """
    Creates a plotting context, write plot when closed.

    Example
    -------
    ```py
    with savefig(filename) as ax:
        ax.pav(lrs, y)
    ```

    A call to `savefig(path)` is identical to `axes(savefig=path)`.

    Parameters
    ----------
    path : str
        write a PNG image to this path
    """
    return axes(savefig=path)


def show():
    """
    Creates a plotting context, show plot when closed.

    Example
    -------
    ```py
    with show() as ax:
        ax.pav(lrs, y)
    ```

    A call to `show()` is identical to `axes(show=True)`.
    """
    return axes(show=True)


@contextmanager
def axes(savefig=None, show=None):
    """
    Creates a plotting context.

    Example
    -------
    ```py
    with axes() as ax:
        ax.pav(lrs, y)
    ```
    """
    fig = plt.figure()
    try:
        yield Canvas(ax=plt)
    finally:
        if savefig:
            fig.savefig(savefig)
        if show:
            plt.show()
        plt.close(fig)

def pav(lrs, y, add_misleading=0, show_scatter=True, ax=plt):
    """
    Generates a plot of pre- versus post-calibrated LRs using Pool Adjacent
    Violators (PAV).

    Parameters
    ----------
    lrs : numpy array of floats
        Likelihood ratios before PAV transform
    y : numpy array
        Labels corresponding to lrs (0 for Hd and 1 for Hp)
    add_misleading : int
        number of misleading evidence points to add on both sides (default: `0`)
    show_scatter : boolean
        If True, show individual LRs (default: `True`)
    ax : pyplot axes object
        defaults to `matplotlib.pyplot`
    ----------
    """
    pav = IsotonicCalibrator(add_misleading=add_misleading)
    pav_lrs = pav.fit_transform(lrs, y)

    with np.errstate(divide='ignore'):
        llrs = np.log10(lrs)
        pav_llrs = np.log10(pav_lrs)

    xrange = yrange = [llrs[llrs != -np.Inf].min() - .5, llrs[llrs != np.Inf].max() + .5]

    # plot line through origin
    ax.plot(xrange, yrange)

    # line pre pav llrs x and post pav llrs y
    line_x = np.arange(*xrange, .01)
    with np.errstate(divide='ignore'):
        line_y = np.log10(pav.transform(10 ** line_x))

    # filter nan values, happens when values are out of bound (x_values out of training domain for pav)
    # see: https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html
    line_x, line_y = line_x[~np.isnan(line_y)], line_y[~np.isnan(line_y)]

    # some values of line_y go beyond the yrange which is problematic when there are infinite values
    mask_out_of_range = np.logical_and(line_y >= yrange[0], line_y <= yrange[1])
    ax.plot(line_x[mask_out_of_range], line_y[mask_out_of_range])

    # add points for infinite values
    if np.logical_or(np.isinf(pav_llrs), np.isinf(llrs)).any():
        def adjust_ticks_labels_and_range(neg_inf, pos_inf, axis_range):
            ticks = np.linspace(axis_range[0], axis_range[1], 6).tolist()
            tick_labels = [str(round(tick, 1)) for tick in ticks]
            step_size = ticks[2] - ticks[1]

            axis_range = [axis_range[0] - (step_size * neg_inf), axis_range[1] + (step_size * pos_inf)]
            ticks = [axis_range[0]] * neg_inf + ticks + [axis_range[1]] * pos_inf
            tick_labels = ['-∞'] * neg_inf + tick_labels + ['+∞'] * pos_inf

            return axis_range, ticks, tick_labels

        def replace_values_out_of_range(values, min_range, max_range):
            # create margin for point so no overlap with axis line
            margin = (max_range - min_range) / 60
            return np.concatenate([np.where(np.isneginf(values), min_range + margin, values),
                                   np.where(np.isposinf(values), max_range - margin, values)])

        yrange, ticks_y, tick_labels_y = adjust_ticks_labels_and_range(np.isneginf(pav_llrs).any(),
                                                                       np.isposinf(pav_llrs).any(),
                                                                       yrange)
        xrange, ticks_x, tick_labels_x = adjust_ticks_labels_and_range(np.isneginf(llrs).any(),
                                                                       np.isposinf(llrs).any(),
                                                                       xrange)

        mask_not_inf = np.logical_or(np.isinf(llrs), np.isinf(pav_llrs))
        x_inf = replace_values_out_of_range(llrs[mask_not_inf], xrange[0], xrange[1])
        y_inf = replace_values_out_of_range(pav_llrs[mask_not_inf], yrange[0], yrange[1])

        ax.yticks(ticks_y, tick_labels_y)
        ax.xticks(ticks_x, tick_labels_x)

        ax.scatter(x_inf,
                   y_inf, facecolors='none', edgecolors='#1f77b4', linestyle=':')

    ax.axis(xrange + yrange)
    # pre-/post-calibrated lr fit

    if show_scatter:
        ax.scatter(llrs, pav_llrs)  # scatter plot of measured lrs

    ax.set_xlabel("pre-calibrated log$_{10}$(LR)")
    ax.set_ylabel("post-calibrated log$_{10}$(LR)")


def lr_histogram(lrs, y, bins=20, weighted=True, ax=plt):
    """
    plots the 10log lrs

    Parameters
    ----------
    lrs : the likelihood ratios
    y : a numpy array of labels (0 or 1)
    bins: number of bins to divide scores into
    weighted: if y-axis should be weighted for frequency within each class
    ax: axes to plot figure to

    """
    log_lrs = np.log10(lrs)

    bins = np.histogram_bin_edges(log_lrs, bins=bins)
    points0, points1 = util.Xy_to_Xn(log_lrs, y)
    weights0, weights1 = (np.ones_like(points) / len(points) if weighted else None
                          for points in (points0, points1))
    ax.hist(points1, bins=bins, alpha=.25, weights=weights1)
    ax.hist(points0, bins=bins, alpha=.25, weights=weights0)
    ax.set_xlabel('log$_{10}$(LR)')
    ax.set_ylabel('count' if not weighted else 'relative frequency')


def tippett(lrs, y, plot_type=1, ax=plt):
    """
    plots empirical cumulative distribution functions of same-source and
        different-sources lrs

    Parameters
    ----------
    lrs : the likelihood ratios
    y : a numpy array of labels (0 or 1)
    plot_type : an integer, must be either 1 or 2.
        In type 1 both curves show proportion of lrs greater than or equal to the
        x-axis value, while in type 2 the curve for same-source shows the
        proportion of lrs smaller than or equal to the x-axis value.
    ax: axes to plot figure to
    """
    log_lrs = np.log10(lrs)

    lr_0, lr_1 = lir.util.Xy_to_Xn(log_lrs, y)
    xplot = np.linspace(np.min(log_lrs), np.max(log_lrs)+0.0001, 10000)
    perc0 = (sum(i >= xplot for i in lr_0) / len(lr_0)) * 100
    if plot_type == 1:
        perc1 = (sum(i >= xplot for i in lr_1) / len(lr_1)) * 100
    elif plot_type == 2:
        perc1 = (sum(i <= xplot for i in lr_1) / len(lr_1)) * 100
    else:
        raise ValueError("plot_type must be either 1 or 2.")


    ax.plot(xplot, perc1, color='b', label='LRs given $\mathregular{H_p}$')
    ax.plot(xplot, perc0, color='r', label='LRs given $\mathregular{H_d}$')
    ax.axvline(x=0, color='k', linestyle='--')
    ax.set_xlabel('log$_{10}$(LR)')
    ax.set_ylabel('Cumulative proportion')
    ax.legend()


def score_distribution(scores, y, bins: int = 20, weighted: bool = True, ax=plt):
    """
    Plots the distributions of scores calculated by the (fitted) lr_system.

    If `weighted` is `True`, the y-axis represents the probability density
    within the class, and `inf` is the fraction of instances. Otherwise, the
    y-axis shows the number of instances.

    Parameters
    ----------
    scores : scores of (fitted) lr_system (1d-array)
    y : a numpy array of labels (0 or 1, 1d-array of same length as `scores`)
    bins: number of bins to divide scores into
    weighted: if y-axis should be the probability density within each class,
        instead of counts
    ax: axes to plot figure to

    """
    ax.rcParams.update({'font.size': 15})
    bins = np.histogram_bin_edges(scores[np.isfinite(scores)], bins=bins)
    bin_width = bins[1] - bins[0]

    # flip Y-classes to achieve blue bars for H1-true and orange for H2-true
    y_classes = np.flip(np.unique(y))
    # create weights vector so y-axis is between 0-1
    scores_by_class = [scores[y == cls] for cls in y_classes]
    if weighted:
        weights = [np.ones_like(data) / len(data) for data in scores_by_class]
    else:
        weights = [np.ones_like(data) for data in scores_by_class]

    # handle inf values
    if np.isinf(scores).any():
        prop_cycle = ax.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        x_range = np.linspace(min(bins), max(bins), 6).tolist()
        labels = [str(round(tick, 1)) for tick in x_range]
        step_size = x_range[2] - x_range[1]
        bar_width = step_size / 4
        plot_args_inf = []

        if np.isneginf(scores).any():
            x_range = [x_range[0] - step_size] + x_range
            labels = ['-∞'] + labels
            for i, s in enumerate(scores_by_class):
                if np.isneginf(s).any():
                    plot_args_inf.append(
                        (colors[i], x_range[0] + bar_width if i else x_range[0], np.sum(weights[i][np.isneginf(s)])))

        if np.isposinf(scores).any():
            x_range = x_range + [x_range[-1] + step_size]
            labels.append('∞')
            for i, s in enumerate(scores_by_class):
                if np.isposinf(s).any():
                    plot_args_inf.append(
                        (colors[i], x_range[-1] - bar_width if i else x_range[-1], np.sum(weights[i][np.isposinf(s)])))

        ax.xticks(x_range, labels)

        for color, x_coord, y_coord in plot_args_inf:
            ax.bar(x_coord, y_coord, width=bar_width, color=color, alpha=0.3, hatch='/')

    for cls, weight in zip(y_classes, weights):
        if cls == 1:
            ax.hist(scores[y == cls], bins=bins, alpha=.3,
                label='Samples of $\mathregular{H_p}$', weights=weight / bin_width if weighted else None)
        else:
            ax.hist(scores[y == cls], bins=bins, alpha=.3,
                label='Samples of $\mathregular{H_d}$', weights=weight / bin_width if weighted else None)

        ax.xlabel('score')
    if weighted:
        ax.ylabel('probability density')
    else:
        ax.ylabel('count')


def calibrator_fit(calibrator, score_range=(0, 1), resolution=100, ax=plt):
    """
    plots the fitted score distributions/score-to-posterior map
    (Note - for ELUBbounder calibrator is the firststepcalibrator)

    TODO: plot multiple calibrators at once
    """
    ax.rcParams.update({'font.size': 15})

    x = np.linspace(score_range[0], score_range[1], resolution)
    calibrator.transform(x)

    ax.plot(x, calibrator.p1, color='tab:blue', label='KDE $\mathregular{\hat{f}_p}$')
    ax.plot(x, calibrator.p0, color='tab:orange', label='KDE $\mathregular{\hat{f}_d}$')
