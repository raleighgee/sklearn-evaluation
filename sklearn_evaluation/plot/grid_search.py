from functools import reduce
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from six import string_types

from ..util import (_group_by, _get_params_value, _mapping_to_tuple_pairs,
                    default_heatmap, _sorted_map_iter, _flatten_list)


def grid_search(grid_scores, change, keep_fixed=None, ax=None, kind='line',
                cmap=None):
    """
    Plot results from a sklearnn Grid Search by changing two parameters at most
    and keeping the rest fixed.

    Parameters
    ----------
    grid_scores : list of named tuples
        Results from a sklearn Grid Search (get them using the
        grid_scores_ parameter)
    change : str or iterable with len<=2
        Parameter to change
    keep_fixed : dictionary-like
        parameter-value pairs, such values will keep fixed. You can
        specify more than one value per parameter
        (e.g. {'n_estimartors': [1, 10]}), if None all combinations will be
        used.
    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses current Axes
    kind : ['line', 'bar']
        This only applies whe change is a single parameter. Changes the
        type of plot
    cmap : matplotlib Colormap
        This only applies when change are two parameters. Colormap used for
        the matrix. If None uses a modified version of matplotlib's OrRd
        colormap.

    Returns
    -------
    ax: matplotlib Axes
        Axes containing the plot

    """
    if change is None:
        raise ValueError(('change can\'t be None, you need to select at least'
                          ' one value to make the plot.'))

    if ax is None:
        ax = plt.gca()

    if cmap is None:
        cmap = default_heatmap()

    if isinstance(change, string_types) or len(change) == 1:
        return _grid_search_single(grid_scores, change, keep_fixed, ax, kind)
    elif len(change) == 2:
        return _grid_search_double(grid_scores, change, keep_fixed, ax, cmap)
    else:
        raise ValueError('change must have length 1 or 2 or be a string')


def _grid_search_single(grid_scores, change, keep_fixed, ax, kind):
    # the logic of this function is to group the grid scores acording
    # to certain rules and subsequently remove the elements that we are
    # not interested in, until we have only the elements that the user
    # wants to plot

    # get a set with all the parameters
    params = set(grid_scores[0].parameters.keys())

    # remove parameter to vary from the list
    # since we are not filtering on that parameter
    try:
        params.remove(change)
    except KeyError:
        raise ValueError('{} is not a valid parameter'.format(change))

    # now need need to filter out the grid_scores that the user
    # didn't select, for that we have to cases, the first one is when
    # the user explicitely selected some values
    if keep_fixed:
        # group the grid_scores based on the values the user selected
        # in keep_fixed
        groups = _group_by(grid_scores, _get_params_value(keep_fixed.keys()))
        keys = _mapping_to_tuple_pairs(keep_fixed)
        groups = {k: v for k, v in _sorted_map_iter(groups) if k in keys}
        grid_scores = _flatten_list(groups.values())
        groups = _group_by(grid_scores, _get_params_value(params))
        if not groups:
            raise ValueError(('There wasn\'t any match with the data provided'
                              ' check that the values in keep_fixed are'
                              ' correct.'))
    # if the user didn't select any values don't filter anything
    # just group the grid_scores depending on the values they
    # have for the parameters
    else:
        groups = _group_by(grid_scores, _get_params_value(params))

    if kind == 'bar':
        change_unique = len(set([g.parameters[change] for g in grid_scores]))
        # bar shifter is just a wrapper around matplotlib's bar plot
        # to automatically calculate the left position on each bar
        bar_shifter = BarShifter(g_number=change_unique, g_size=len(groups),
                                 ax=ax)

    for params_kv, group in _sorted_map_iter(groups):
        # get the x and y values for each grid_score on this group
        # also calculate the std
        x = [element.parameters[change] for element in group]
        y = [element.mean_validation_score for element in group]
        stds = [element.cv_validation_scores.std() for element in group]

        # take (param, value) and convert them to 'param: value'
        label = ['{}: {}'.format(*t) for t in params_kv]
        # now convert it to one string
        label = reduce(lambda x, y: x+', '+y, label)

        if kind == 'bar':
            bar_shifter(y, yerr=stds, label=label)
        elif kind == 'line':
            is_categorical = isinstance(x[0], string_types)
            if is_categorical:
                ints = range(len(x))
                ax.errorbar(ints, y, yerr=stds, label=label)
                ax.set_xticks(ints)
            else:
                ax.errorbar(x, y, yerr=stds, label=label)
                ax.set_xticks(x)

    ax.set_xticklabels(x)
    ax.set_title('Grid search results')
    ax.set_ylabel('Mean score')
    ax.set_xlabel(change)
    ax.legend(loc="best")
    ax.margins(0.05)
    return ax


def _grid_search_double(grid_scores, change, keep_fixed, ax, cmap):
    # if a value in keep_fixed was passed, use it to filter the groups
    # if not use the set of parameters but remove the ones the user wants
    # to vary
    if keep_fixed is not None:
        groups = _group_by(grid_scores, _get_params_value(keep_fixed.keys()))
        keys = _mapping_to_tuple_pairs(keep_fixed)
        groups = {k: v for k, v in _sorted_map_iter(groups) if k in keys}
    else:
        keep_fixed = list(grid_scores[0].parameters.keys())
        for p in change:
            try:
                keep_fixed.remove(p)
            except ValueError:
                raise ValueError('{} is not a valid parameter.'.format(p))
        groups = _group_by(grid_scores, _get_params_value(keep_fixed))

    # there should be just one group at this point
    if len(groups) > 1:
        raise ValueError(('More than one result matched your criteria.'
                          ' Make sure you specify parameters using change'
                          ' and keep_fixed so only one group matches.'))
    elif len(groups) == 0:
        raise ValueError(('There wasn\'t any match with the data provided'
                          ' check that the values in keep_fixed are correct.'))

    group = list(groups.values())[0]

    # group by every possible combination in change
    matrix_elements = _group_by(group, _get_params_value(change))

    # for v in matrix_elements.values():
    #     if len(v) > 1:
    #         raise ValueError('Group with more than one element')

    # on each group there must be only one element, get it
    matrix_elements = {k: v[0] for k, v in matrix_elements.items()}

    # get the unique values for each element
    # and sort the results to make sure the matrix axis
    # is showed in increasing order
    row_names = sorted(set([t[0] for t in matrix_elements.keys()]),
                       key=itemgetter(1))
    col_names = sorted(set([t[1] for t in matrix_elements.keys()]),
                       key=itemgetter(1))

    # size of the matrix
    cols = len(col_names)
    rows = len(row_names)

    # map values to coordinates to populate the marix
    x_coord = {k: v for k, v in zip(col_names, range(cols))}
    y_coord = {k: v for k, v in zip(row_names, range(rows))}

    # replace keys in matrix_elements with their corresponding indexes
    # improve variable naming
    m = {(x_coord[k[1]], y_coord[k[0]]): v for k, v in matrix_elements.items()}

    matrix = np.zeros((rows, cols))

    for (j, i), v in m.items():
        matrix[i][j] = v.mean_validation_score

    # ticks for the axis
    row_labels = ['{}={}'.format(*x) for x in row_names]
    col_labels = ['{}={}'.format(*y) for y in col_names]

    im = ax.imshow(matrix, interpolation='nearest', cmap=cmap)

    # set text on cells
    for (x, y), v in m.items():
        label = '{:.3}'.format(v.mean_validation_score)
        ax.text(x, y, label, horizontalalignment='center',
                verticalalignment='center')

    ax.set_xticks(range(cols))
    ax.set_xticklabels(col_labels, rotation=45)
    ax.set_yticks(range(rows))
    ax.set_yticklabels(row_labels)
    plt.colorbar(im, ax=ax)
    ax.get_figure().tight_layout()
    return ax


class BarShifter:
    # bar shifter is just a wrapper arounf matplotlib bar chart
    # which automatically computes the position of the bars
    # you need to specify how many groups of bars are gonna be
    # and the size of such groups. The frst time you call it,
    # will plot the first element for every group, then the second one
    # and so on
    def __init__(self, g_number, g_size, ax, scale=0.8):
        self.g_number = g_number
        self.g_size = g_size
        self.ax = ax
        self.i = 0
        self.width = (1.0/g_size)*scale
        self.colors = plt.get_cmap()(np.linspace(0, 1, self.g_size))

    def __call__(self, height, bottom=None, hold=None, **kwargs):
        left = [x+self.i*self.width for x in range(self.g_number)]
        self.ax.bar(left, height, self.width, color=self.colors[self.i],
                    ecolor=self.colors[self.i],
                    **kwargs)
        self.i += 1
        if self.i == self.g_size:
            n = range(self.g_number)
            ticks_pos = [x+(self.width*self.g_size)/2.0 for x in n]
            self.ax.set_xticks(ticks_pos)
