import json
import unittest

from matplotlib.testing.decorators import image_comparison, cleanup

from sklearn_evaluation import plot
from sklearn_evaluation.util import _grid_scores_from_dicts

# Parameters:
# n_estimators': [1, 10, 50, 100]
# criterion': ['gini', 'entropy']
# max_features': ['sqrt', 'log2']

with open('static/sample_scores.json') as f:
    grid_scores = _grid_scores_from_dicts(json.loads(f.read()))

with open('static/sample_scores_2_params.json') as f:
    grid_scores_2_params = _grid_scores_from_dicts(json.loads(f.read()))

# plot tests


@image_comparison(baseline_images=['single_numeric_line'],
                  extensions=['png'],
                  remove_text=True)
def test_single_numeric_line():
    change = 'n_estimators'
    plot.grid_search(grid_scores, change, kind='line')


@image_comparison(baseline_images=['single_numeric_line'],
                  extensions=['png'],
                  remove_text=True)
def test_single_numeric_line_with_tuple():
    change = ('n_estimators')
    plot.grid_search(grid_scores, change, kind='line')


@image_comparison(baseline_images=['single_numeric_bar'],
                  extensions=['png'],
                  remove_text=True)
def test_single_numeric_bar():
    change = 'n_estimators'
    plot.grid_search(grid_scores, change, kind='bar')


@image_comparison(baseline_images=['single_categorical_line'],
                  extensions=['png'],
                  remove_text=True)
def test_single_categorial_line():
    change = 'n_estimators'
    plot.grid_search(grid_scores, change, kind='line')


@image_comparison(baseline_images=['single_categorical_bar'],
                  extensions=['png'],
                  remove_text=True)
def test_single_categorial_bar():
    change = 'n_estimators'
    plot.grid_search(grid_scores, change, kind='bar')


@image_comparison(baseline_images=['single_numeric_partially_restricted'],
                  extensions=['png'],
                  remove_text=True)
def test_single_numeric_partially_restricted():
    change = 'n_estimators'
    keep_fixed = {'max_features': 'sqrt'}
    plot.grid_search(grid_scores, change, keep_fixed, kind='bar')


@image_comparison(baseline_images=['single_numeric_restricted_single'],
                  extensions=['png'],
                  remove_text=True)
def test_single_numeric_restricted_single():
    change = 'n_estimators'
    keep_fixed = {'max_features': 'sqrt', 'criterion': 'gini'}
    plot.grid_search(grid_scores, change, keep_fixed, kind='bar')


@image_comparison(baseline_images=['single_numeric_restricted_multi'],
                  extensions=['png'],
                  remove_text=True)
def test_single_numeric_restricted_multi():
    change = 'n_estimators'
    keep_fixed = {'max_features': ['sqrt', 'log2'], 'criterion': 'gini'}
    plot.grid_search(grid_scores, change, keep_fixed, kind='bar')


@image_comparison(baseline_images=['double'],
                  extensions=['png'],
                  remove_text=True)
def test_double_ignores_kind_line():
    change = ('n_estimators', 'criterion')
    keep_fixed = {'max_features': 'sqrt'}
    plot.grid_search(grid_scores, change, keep_fixed, kind='line')


@image_comparison(baseline_images=['double'],
                  extensions=['png'],
                  remove_text=True)
def test_double_ignores_kind_bar():
    change = ('n_estimators', 'criterion')
    keep_fixed = {'max_features': 'sqrt'}
    plot.grid_search(grid_scores, change, keep_fixed, kind='bar')


# API tests

class TestGridSearchAPI(unittest.TestCase):

    @cleanup
    def test_list_with_len_three_raises_exception(self):
        l = ['a', 'b', 'c']
        with self.assertRaises(ValueError):
            plot.grid_search(grid_scores, l)

    @cleanup
    def test_none_change_raises_exception(self):
        with self.assertRaises(ValueError):
            plot.grid_search(grid_scores, None)

    @cleanup
    def test_can_send_tuple_len_one(self):
        change = ('n_estimators')
        plot.grid_search(grid_scores, change)

    @cleanup
    def test_can_send_string(self):
        change = 'n_estimators'
        plot.grid_search(grid_scores, change)

    @cleanup
    def test_raise_exception_when_parameter_set_is_not_fully_specified(self):
        with self.assertRaises(ValueError):
            change = ('n_estimators', 'criterion')
            plot.grid_search(grid_scores, change=change, keep_fixed=None)

    @cleanup
    def test_keep_fixed_can_be_none_when_parameter_set_is_fully_specified(self):
        change = ('n_estimators', 'criterion')
        plot.grid_search(grid_scores_2_params, change=change, keep_fixed=None)

    @cleanup
    def test_raise_exception_when_parameter_does_not_exist(self):
        with self.assertRaises(ValueError):
            change = ('this_is_not_a_parameter')
            keep_fixed = {'criterion': 'gini',
                          'max_features': 'sqrt'}
            plot.grid_search(grid_scores, change=change, keep_fixed=keep_fixed)

    @cleanup
    def test_raise_exception_when_parameter_does_not_exist_double(self):
        with self.assertRaises(ValueError):
            change = ('n_estimators', 'this_is_not_a_parameter')
            keep_fixed = {'criterion': 'gini',
                          'max_features': 'sqrt'}
            plot.grid_search(grid_scores, change=change, keep_fixed=keep_fixed)

    @cleanup
    def test_raise_exception_when_invalid_value_in_keep_fixed_double(self):
        with self.assertRaises(ValueError):
            change = ('n_estimators', 'max_features')
            keep_fixed = {'criterion': 'not_a_value'}
            plot.grid_search(grid_scores, change=change, keep_fixed=keep_fixed)

    @cleanup
    def test_raise_exception_when_invalid_value_in_keep_fixed(self):
        with self.assertRaises(ValueError):
            change = 'n_estimators'
            keep_fixed = {'criterion': 'not_a_value'}
            plot.grid_search(grid_scores, change=change, keep_fixed=keep_fixed)
