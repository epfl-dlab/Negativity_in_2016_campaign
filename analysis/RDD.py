import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from statsmodels.stats.outliers_influence import summary_table
import statsmodels.formula.api as smf
import sys
from typing import Any, Dict, List, Tuple, Union
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from utils.plots import timeLinePlot, ONE_COL_FIGSIZE

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Folder with aggregates', required=True)
parser.add_argument('--save', help='Folder to save RDD fits to',  required=True)

CM = 1/2.54
FONTSIZE = 14
DAY_ZERO = datetime(2008, 11, 1)
KINK = datetime(2015, 6, 15)
ROW_NAMES = {
    'Negemo': 'Negative Emotions',
    'Anx': 'Anxiety',
    'Anger': 'Anger',
    'Sad': 'Sadness',
    'Swear': 'Swearing Terms',
    'Posemo': 'Positive Emotions',
    'Certain': 'Certainty',
    'Tentat': 'Tentativeness',
    'positive emotion': 'Pos. Emotion (empath)',
    'negative emotion': 'Neg. Emotion (empath)',
    'swearing terms': 'Swearing (empath)'
}


def get_first(x):
    """
    Gets string of a float up to the first non-zero decimal.
    Code by Manoel Horta Ribeiro: https://github.com/epfl-dlab/platform_bans/blob/main/helpers/regression_helpers.py
    """
    if int(x) == x:  # Gotta "love" python
        return str(int(x))
    x_ = abs(x)

    decimal_part = abs(x_ - float(int(x_)))
    max0s = 3
    if x_ > 1:
        max0s = 1
    if x_ > 1000:
        max0s = 0

    first_non0 = math.ceil(abs(np.log10(decimal_part)))

    strv = str(round(x, min(first_non0, max0s)))
    if strv[-2:] == ".0":
        strv = strv[:-2]

    return strv


def pvalstars(x):
    if x <= 0.001:
        return "***"
    if x <= 0.01:
        return "**"
    if x <= 0.05:
        return "*"
    return ""


def aicc(aic, k, n):
    """Corrected Akaike Information Criterion for sample size n and k parameters"""
    return aic + (2 * k ** 2 + 2 * k) / (n - k - 1)


def w_i(aicc_i: float, aicc_all: List[float]):
    """Get the weight of evidence for model based on AICc"""
    delta_i = aicc_i - min(aicc_all)
    delta_all = [a - min(aicc_all) for a in aicc_all]

    def p(f):
        return np.exp(-.5 * f)

    return p(delta_i) / sum(p(m) for m in delta_all)


class RDD:

    def __init__(self, data: pd.DataFrame, rdd_dict: Dict, lin_reg_dict: Dict):
        self.rdd_metrics = None
        self.data = self.add_date_or_pass(data)
        self.rdd = self.rdd_results_to_df(rdd_dict)
        self.rdd_err_summary = {feature: rdd_dict[feature]['summary'] for feature in rdd_dict}
        self.rdd_fit = {feature: rdd_dict[feature]['res'] for feature in rdd_dict}
        self.lin_reg = self.lin_reg_results_to_df(lin_reg_dict)

    @staticmethod
    def add_date_or_pass(data: pd.DataFrame):
        """Adds a date column to the data.
        If no date was present before, it is based on the first day of Quotebank + the time_delta column.
        """
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            return data

        if 'time_delta' not in data.columns:
            raise ValueError('Assumed there would be a time_delta column.')

        data = data.copy(deep=True)
        dates = [DAY_ZERO + relativedelta(months=i) for i in data['time_delta']]
        data['date'] = dates
        return data

    @staticmethod
    def lin_reg_results_to_df(results: Dict) -> pd.DataFrame:
        """Translates a dictionary of linear regression results to a pandas Data Frame"""
        columns = sorted(results.keys())
        first_row = results[columns[0]]
        rows = list(k for k in first_row.keys() if k != 'splits')

        values = np.empty([len(rows), len(columns)])
        for i, c in enumerate(columns):
            try:
                values[:, i] = np.asarray([results[c][r] for r in rows])
            except ValueError:
                print(values, results[c], rows)
                raise ValueError
        return pd.DataFrame(data=values, index=rows, columns=columns)

    @staticmethod
    def rdd_results_to_df(rdd_results: Dict) -> pd.DataFrame:
        """Translates a dictionary of RDD results to a pandas Data Frame"""
        columns = sorted(rdd_results.keys())
        first_row = rdd_results[columns[0]]
        sample_splits = list(first_row['splits'])
        rows = first_row['parameters'].index.tolist() + [f'split_{i}' for i in range(len(sample_splits))] + ['r2', 'r2_adj', 'f_score', 'SSR', 'aic']

        values = np.empty([len(rows), len(columns)])
        for i, c in enumerate(columns):  #
            data = rdd_results[c]
            splits = list(data['splits'])
            r2 = data['r2']
            r2_adj = data['r2_adjust']
            f_score = data['f_score']
            SSR = data['SSR']
            aic = data['aic']
            values[:, i] = np.asarray([*data['parameters'].values, *splits, r2, r2_adj, f_score, SSR, aic])
        return pd.DataFrame(data=values, index=rows, columns=columns)

    def _get_approx_date(self, delta: float):
        """Returns a date based on a time delta value. This works for deltas that are not present in the original data
        and floats - but it rounds only to full days."""
        if delta in self.data['time_delta'].values:
            return self.data[self.data['time_delta'] == delta]['date'].values[0]

        closest = np.argmin(np.abs(self.data['time_delta'].values - delta))
        diff = self.data['time_delta'].values[closest] - delta
        approx_days = 30 * diff
        return self.data.date[closest] + relativedelta(days=approx_days)

    def _get_rdd_plot_data(self, feature: str) -> Tuple[np.array, np.array]:
        """Uses the internal RDD data to generate the points needed for plotting"""
        rdd = self.rdd[feature]
        intercept = rdd['Intercept']
        slope_base = rdd['time_delta']
        num_disc = len([idx for idx in rdd.index if 'split_' in idx])
        num_lines = num_disc + 1
        splits = [rdd[f'split_{i}'] for i in range(num_disc)]
        threshold_effect = {i: rdd[f'C(threshold)[T.{i}]'] for i in range(1, num_lines)}
        threshold_dt_effect = {i: rdd[f'C(threshold)[T.{i}]:time_delta'] for i in range(1, num_lines)}

        X = np.empty([2 * num_lines])
        Y = np.empty([2 * num_lines])

        # temporary offsetting x
        X[0] = - splits[0]
        X[1] = -1
        Y[1] = intercept
        Y[0] = Y[1] + slope_base * X[0]

        for i in range(1, num_lines):
            slope = slope_base + threshold_dt_effect[i]
            X[2 * i] = X[2 * i - 1] + 1
            try:
                X[2 * i + 1] = splits[i] + X[0]
            except IndexError:
                X[2 * i + 1] = max(self.data.time_delta) + X[0]

            Y[2 * i] = intercept + threshold_effect[i] + X[2 * i] * slope
            Y[2 * i + 1] = intercept + threshold_effect[i] + X[2 * i + 1] * slope

        # Resetting x-values: RDD works centered around the first discontinuity, but the other models don't
        X = X + splits[0]
        return X, Y

    def _rdd_confidence(self, feature: str) -> Tuple[List[np.array], List[np.array]]:
        """
        https://stackoverflow.com/questions/17559408/confidence-and-prediction-intervals-with-statsmodels
        Returns 95% Confidence Intervals for the RDD
        """
        st, data, ss2 = self.rdd_err_summary[feature]
        lower, upper = data[:, 4:6].T
        splits = self.rdd[feature][self.rdd.index.map(lambda x: 'split' in x)].values

        def find_closest(lst: List, value: Union[int, float]):
            arr = np.asarray(lst)
            return int(np.argmin(np.abs(arr - value)))

        split_indices = np.asarray([find_closest(self.data.time_delta.tolist(), s) for s in splits])
        return np.split(lower, split_indices), np.split(upper, split_indices)

    def _get_lin_reg_plot_data(self, feature: str) -> Tuple[np.array, np.array]:
        slope = self.lin_reg[feature]['time_delta']
        intercept = self.lin_reg[feature]['Intercept']

        X = np.asarray([0, max(self.data['time_delta'])])
        Y = np.asarray([intercept, intercept + slope * X[1]])

        return X, Y

    def get_table(self, features: List[str] = None, CI_features: List[str] = None, asPandas=False) -> str:
        """
        Makes a Latex table from the internal RDD rsults.
        Parameters
        ----------
        features: Include only these features. If not given, will be all features the RDD was fitted on.
        CI_features: Include Confidence Intervals only for these features. If None, will be all.
        asPandas: If True, will not return a Latex table but the Pandas table created as an intermediate step. Useful for plotting.
        Returns
        -------
        A Latex Tabular including RDD parameters and Confidence as used in the Paper appendix.
        """
        stats = list()
        if features is None:
            features = list(self.rdd_fit.keys())
        for feature in features:
            res = self.rdd_fit[feature]
            summary = pd.read_html(res.summary().tables[1].as_html(), header=0, index_col=0)[0]
            summary = summary.rename(index={'Intercept': r'$\alpha_0$', 'C(threshold)[T.1]': r'$\alpha_1$',
                                            'time_delta': r'$\beta_0$', 'C(threshold)[T.1]:time_delta': r'$\beta_1$',
                                            'governing_party': 'governing', 'congress_member': 'congress'},
                                     columns={'P>|t|': 'p-value', '[0.025': 'CI_lower', '0.975]': 'CI_upper'}
                                     )[['coef', 'p-value', 'CI_lower', 'CI_upper']]
            stats.append(summary.T.stack())

        df = pd.concat(stats, axis=1).T
        df.index = pd.Series(features).map(lambda x: ' '.join(x.split('_')[1:]))
        coef = df['coef'].applymap(get_first)
        p_val = df['p-value'].applymap(pvalstars)
        conf = '(' + df['CI_lower'].applymap(get_first) + ', ' + df['CI_upper'].applymap(get_first) + ')'
        if CI_features is not None:
            ignore = [col for col in conf.columns if col not in CI_features]
            conf[ignore] = ''
        table = coef + p_val + conf
        if asPandas:
            return table

        return table.rename(index=ROW_NAMES).to_latex(sparsify=True, escape=False, multicolumn_format='c')

    def plot(self, feature, ax=None, parameters=False, **kwargs) -> Tuple[plt.axis, plt.figure]:
        """
        Creates an RDD plot as displayed in the Paper.
        Parameters
        ----------
        feature: Feature which shall be plotted.
        ax: If given, will plot on provided axis instead of creating a new one.
        parameters: If True, will annotate the plot with important RDD parameters.
        kwargs: Plotting parameters like color, scatter_color and more.
        Returns
        -------
        A fig, ax tuple.
        """
        if ax is not None:
            fig = plt.gcf()
        else:
            fig, ax = plt.subplots(figsize=ONE_COL_FIGSIZE)

        color = kwargs.get('color', 'black')

        s = 15 if kwargs.get('lean', False) else 30
        scatter_color = kwargs.get('scatter_color', 'tab:grey') if kwargs.get('lean', False) else 'black'
        timeLinePlot(self.data.date, self.data[feature], ax=ax, snsargs={'s': s, 'color': scatter_color}, kind='scatter')

        # RDD
        linewidth = 3 if kwargs.get('lean') else 5
        linewidth = kwargs.get('linewidth', linewidth)
        X_rdd, Y_rdd = self._get_rdd_plot_data(feature)
        dates_rdd = [self._get_approx_date(x) for x in X_rdd]
        for i in range(len(dates_rdd) // 2):
            timeLinePlot([dates_rdd[2 * i], dates_rdd[2 * i + 1]], [Y_rdd[2 * i], Y_rdd[2 * i + 1]], ax=ax,
                         snsargs={'label': 'RDD' if i == 0 else '', 'color': color, 'linewidth': linewidth},
                         timeDelta='1y')

        # RDD Confidence
        if not kwargs.get('lean', False) and kwargs.get('ci', True):
            lower, upper = self._rdd_confidence(feature)
            conf_deltas = self.data[~self.data[feature].isna()].time_delta.values
            from_date = 0
            for low, up in zip(lower, upper):
                to_date = from_date + len(low)
                dates_ci = [self._get_approx_date(x) for x in conf_deltas[from_date:to_date]]
                ax.fill_between(dates_ci, low, up, alpha=.2, color=color)
                from_date = to_date

        # Linear Regression
        X_lin, Y_lin = self._get_lin_reg_plot_data(feature)
        dates_lin = [self._get_approx_date(x) for x in X_lin]
        timeLinePlot(dates_lin, Y_lin, ax=ax, clean_dates=False,
                     snsargs={'label': 'Linear Regression', 'color': 'black', 'linewidth': 2, 'linestyle': '-.'},
                     timeDelta='1y')

        # Performance Annotations
        if kwargs.get('annotate', True):
            aicc_rdd = aicc(self.rdd[feature].aic, 4, len(self.data))
            aicc_lin = aicc(self.lin_reg[feature].aic, 2, len(self.data))
            w_rdd = w_i(aicc_rdd, [aicc_rdd, aicc_lin])
            annotations = ', '.join([
                f'$r^2_{"{RDD, adj}"}$ : {self.rdd[feature].loc["r2_adj"]:.2f}',
                f'$r^2_{"{lin, adj}"}$ : {self.lin_reg[feature].loc["r2_adjust"]:.2f}',
                f'$F$ : {self.rdd[feature].loc["f_score"]:.1f}',
                '$w_{RDD}$: ' + f'{w_rdd:.3f}',
                r'$\Delta_{AICc}$: ' + f'{aicc_lin - aicc_rdd:.2f}'
            ])
            ax.set_title(annotations, fontsize=FONTSIZE)

        # Parameter Annotations
        if parameters:
            param_annotation = ', '.join([
                r'$\alpha_1$=' + self.get_table(asPandas=True)[r'$\alpha_1$'].loc[
                    ' '.join(feature.split('_')[1:])],
                r'$\beta_1$=' + self.get_table(asPandas=True)[r'$\beta_1$'].loc[' '.join(feature.split('_')[1:])]
            ])
            # ax.annotate(param_annotation, (1.02, 0), fontsize=FONTSIZE,
            #             xycoords=trans, rotation=90)
            box_props = dict(boxstyle='square', facecolor='white', alpha=0.5)
            ax.text(0.98, 0.975, param_annotation, transform=ax.transAxes, fontsize=FONTSIZE, verticalalignment='top',
                    horizontalalignment='right', bbox=box_props)

        # Visuals
        if kwargs.get('visuals', True):
            data_values = [val for val in self.data[feature] if not np.isnan(val)]
            ax.set_ylim(min(data_values) - 1, max(data_values) + 1)
            plt.legend(fontsize=FONTSIZE, loc='lower left', framealpha=1, fancybox=False, ncol=2)
            plt.tight_layout()

        return fig, ax


def __try_float(s: str) -> Union[None, float]:
    try:
        return float(s)
    except ValueError:
        return None


def RDD_statsmodels(data: pd.DataFrame, t: str = 'time_delta') -> Dict[str, Any]:
    """
    A Regression Discontinuity Design implemented using statsmodels.
    Parameters
    ----------
    data: Data Frame containing the data to be fitted
    t: The column name of the time column in data.
    Returns
    -------
    A dictionary of RDD parameters for fitted RDD on all features. Can be used by the RDD class.
    """
    language_features = [c for c in data.columns if ('empath' in c) or ('liwc' in c)]  # liwc_X and empath_X

    results = dict()

    for feature in language_features:
        tmp = data.copy(deep=True)  # Don't alter input data
        tmp['date'] = pd.to_datetime(tmp['date'])  # Make sure date formatting is uniform
        tmp['threshold'] = tmp['date'].apply(lambda dt: int(dt >= KINK))
        delta_at_kink = tmp.loc[tmp.date == KINK][t].values[0]
        tmp[t] = tmp[t] - delta_at_kink

        # Will automatically fit on all of the speaker attributes as well if they are given.
        speaker_attributes = [c for c in ('party', 'gender', 'congress_member', 'governing_party') if c in tmp.columns]
        formula = f'{feature} ~ C(threshold) * {t}'  # RDD base formula
        if len(speaker_attributes) > 0:
            formula += ' + ' + ' + '.join(speaker_attributes)

        try:
            mod = smf.ols(formula=formula, data=tmp, missing='drop')
        except ValueError:
            print(feature)
            print(tmp.head(150))
            sys.exit(100)
        res = mod.fit()

        results[feature] = {
            'splits': [delta_at_kink],
            'f_score': res.fvalue,
            'r2': res.rsquared,
            'r2_adjust': res.rsquared_adj,
            'RMSE': (res.ssr / len(tmp)) ** 0.5,
            'SSR': res.ssr,
            'summary': summary_table(res, alpha=0.05),
            'aic': res.aic,
            'res': res
        }

        results[feature].update({'parameters': res.params})

    return results


def linear_regression(data: pd.DataFrame, t='time_delta'):
    """
    A wrapper for a linear regression that takes the same input as the RDD function and returns a similar dictionary
    """
    language_features = [c for c in data.columns if ('empath' in c) or ('liwc' in c)]  # liwc_X and empath_X
    results = dict()

    for feature in language_features:
        speaker_attributes = [c for c in ('party', 'gender', 'congress', 'governing_party') if c in data.columns]
        formula = f'{feature} ~ {t}'
        if len(speaker_attributes) > 0:
            formula += ' + ' + ' + '.join(speaker_attributes)
        mod = smf.ols(formula=formula, data=data)
        res = mod.fit()

        results[feature] = {
            'f_score': res.fvalue,
            'r2': res.rsquared,
            'r2_adjust': res.rsquared_adj,
            'RMSE': (res.ssr / len(data)) ** 0.5,
            'SSR': res.ssr,
            'aic': res.aic
        }

        results[feature].update({p: res.params[p] for p in res.params.index})

    return results


def remove_outliers(data: pd.DataFrame, thresh: float, center: bool = True) -> pd.DataFrame:
    """
    Takes a dataframe and sets every value, where abs(value) >= threshold to np.nan.
    Parameters
    ----------
    data: Numeric dataframe
    thresh: Threshold value
    center: If true, the mean of the data will be subtracted for getting the threshold values. The returned
    data will be the original, uncentered data.
    """
    tmp = data.copy(deep=True)
    features = [c for c in data.columns if ('empath' in c) or ('liwc' in c)]
    if center:
        centered = tmp - tmp.mean()
        mask = centered[features].applymap(lambda val: abs(val) >= thresh)
    else:
        mask = tmp[features].applymap(lambda val: abs(val) >= thresh)

    tmp[features][mask] = np.nan
    return tmp


def main():
    args = parser.parse_args()
    data_folder = Path(args.data)
    storage_folder = Path(args.save)
    storage_folder.mkdir(exist_ok=True)

    for file in tqdm(list(data_folder.iterdir())):
        if not file.name.endswith('.csv'):
            continue
        aggregates = pd.read_csv(str(file)).sort_values('time_delta')
        outliers_removed = remove_outliers(aggregates, 2)
        fname = file.name.split('.')[0]

        masks = dict()
        make_folder = None
        if 'verbosity' in aggregates.columns:
            make_folder = 'verbosity'
            for verb in aggregates['verbosity'].unique():
                masks['_verbosity_{}'.format(verb)] = aggregates.verbosity == verb

        elif ('party' in aggregates.columns) and not ('gender' in aggregates.columns):  # Party Only split
            make_folder = 'parties'
            masks['_republicans'] = aggregates.party == 0
            masks['_democrats'] = aggregates.party == 1

        else:
            masks = {'': np.ones(len(aggregates), dtype=bool)}  # Default, no mask

        for data, kind in zip((aggregates, outliers_removed), ('original', 'outliers')):
            for prefix, mask in masks.items():
                tmp = data[mask]
                if prefix in ('_democrats', '_republicans'):
                    tmp = tmp.drop('party', axis=1)
                rdd_results = RDD_statsmodels(tmp)
                lin_reg = linear_regression(tmp)
                reg = RDD(tmp, rdd_results, lin_reg)

                save_in = storage_folder
                if make_folder is not None:
                    save_in = storage_folder.joinpath(make_folder)
                    save_in.mkdir(exist_ok=True)

                if kind == 'outliers':
                    prefix += '_outliers'

                pickle.dump(reg, save_in.joinpath(fname + '_RDD' + prefix + '.pickle').open('wb'))
                with save_in.joinpath(fname + '_tabular' + prefix + '.tex').open('w') as tab:
                    tab.write(reg.get_table())


if __name__ == '__main__':
    main()
