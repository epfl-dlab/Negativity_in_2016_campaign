import argparse
from copy import deepcopy

import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import re
import sys
from typing import Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from utils.plots import saveFigure, timeLinePlot, ONE_COL_FIGSIZE, TWO_COL_FIGSIZE
from analysis.RDD import RDD, aicc

# TODO: Outliers
parser = argparse.ArgumentParser()
parser.add_argument('--rdd', help='Folder containing fitted RDDs', required=True)
parser.add_argument('--img', help='Folder to write images to.', required=True)

FONTSIZE = 14
NAMES = {
    'liwc_Negemo': 'Negative Emotions',
    'liwc_Anx':    'Anxiety',
    'liwc_Anger':  'Anger',
    'liwc_Sad':    'Sadness',
    'liwc_Swear':  'Swearing Terms',
    'liwc_Posemo': 'Positive Emotions',
    'linreg':      'Linear Regression',
    'liwc_Certain':'Certainty',
    'liwc_Tentat': 'Tentativeness',
    'empath_negative_emotion': 'Neg. Emotion (empath)',
    'empath_positive_emotion': 'Pos. Emotion (empath)',
    'empath_swearing_terms': 'Swearing (empath)'
}
STYLES = {
    'liwc_Negemo': {'color': 'tab:red', 'linewidth': 4},
    'liwc_Anx':    {'color': 'darkorange', 'linewidth': 3},
    'liwc_Anger':  {'color': 'darkorange', 'linewidth': 3},
    'liwc_Sad':    {'color': 'darkorange', 'linewidth': 3},
    'liwc_Swear':  {'color': 'black', 'linewidth': 3},
    'liwc_Posemo': {'color': 'tab:green', 'linewidth': 4},
    'linreg':      {'color': 'black', 'linewidth': 2, 'linestyle': '-.'}
}


def _conf_only(ax: plt.axis, feature: str, model: RDD, color: str):
    lower, upper = model._rdd_confidence(feature)
    from_date = 0
    for low, up in zip(lower, upper):
        to_date = from_date + len(low)
        ax.fill_between(model.data.date[from_date:to_date], low, up, alpha=.2, color=color)
        from_date = to_date


def _rdd_only(ax: plt.axis, feature: str, model: RDD, kwargs: Dict):
    kwargs = deepcopy(kwargs)
    X_rdd, Y_rdd = model._get_rdd_plot_data(feature)
    dates_rdd = [model._get_approx_date(x) for x in X_rdd]
    for i in range(len(dates_rdd) // 2):
        if i > 0:
            kwargs['label'] = ''
        timeLinePlot([dates_rdd[2 * i], dates_rdd[2 * i + 1]], [Y_rdd[2 * i], Y_rdd[2 * i + 1]], ax=ax, snsargs=kwargs)


def _scatter_only(ax: plt.axis, feature: str, model: RDD, color: str):
    timeLinePlot(model.data.date, model.data[feature], ax=ax, snsargs={'s': 15, 'color': color}, kind='scatter')


def _grid_annotate(ax: plt.axis, model: RDD, feature: str, title: str):
    ax.set_title(title, fontsize=FONTSIZE, fontweight='bold')

    aicc_rdd = aicc(model.rdd[feature].aic, 4, len(model.data))
    aicc_lin = aicc(model.lin_reg[feature].aic, 2, len(model.data))

    fit = ',  '.join([
        f'$r^2_{"{RDD, adj}"}$={model.rdd[feature].loc["r2_adj"]:.2f}',
        f'$r^2_{"{lin, adj}"}$={model.lin_reg[feature].loc["r2_adjust"]:.2f}',
        r'$\Delta_{AICc}$: ' + f'{aicc_lin - aicc_rdd:.2f}'
    ])
    params = ',  '.join([
        r'$\alpha_1$=' + model.get_table(asPandas=True)[r'$\alpha_1$'].loc[' '.join(feature.split('_')[1:])],
        r'$\beta_1$=' + model.get_table(asPandas=True)[r'$\beta_1$'].loc[' '.join(feature.split('_')[1:])]
    ])
    box_props = dict(boxstyle='round', facecolor='white', alpha=1)

    ax.text(0.025, 0.05, fit, transform=ax.transAxes, fontsize=FONTSIZE, verticalalignment='bottom',
            horizontalalignment='left', bbox=box_props)
    ax.text(0.975, 0.95, params, transform=ax.transAxes, fontsize=FONTSIZE, verticalalignment='top',
            horizontalalignment='right', bbox=box_props)


def basic_model_plots(model_file: Path, base: Path):
    """
    Creates a single plot for every feature the RDD was fitted on.
    Parameters
    ----------
    model_file: Path to an RDD Model
    base: Base folder to store results in
    -------

    """
    model = pickle.load(model_file.open('rb'))
    features = [col for col in model.data.columns if ('empath' in col) or ('liwc' in col)]

    # Single-Feature Plots
    for f in sorted(features):
        fig, ax = model.plot(f, parameters=True)
        saveFigure(fig, base.joinpath(f))
        plt.close()

    # 2 x 3 Grid
    selected_features = ['liwc_Negemo', 'liwc_Posemo', 'liwc_Anger', 'liwc_Sad', 'liwc_Anx', 'liwc_Swear']
    fig, axs = plt.subplots(figsize=TWO_COL_FIGSIZE, ncols=3, nrows=2, sharex='all', sharey='all')
    ymin = np.inf
    ymax = - np.inf
    for i, feature in enumerate(selected_features):
        COL = i % 2
        ROW = i // 2
        ax = axs[COL][ROW]
        model.plot(feature, ax=ax, annotate=False, visuals=False, **STYLES[feature])
        ax.set_xlim(13926.0, 18578.0)
        _grid_annotate(ax, model, feature, NAMES[feature])
        ymin = min(ymin, min(model.data[feature]))
        ymax = max(ymax, max(model.data[feature]))

    ydiff = ymax - ymin
    for row in axs:
        for ax in row:
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
            ax.get_legend().remove()
            ax.set_ylim(ymin - 0.25 * ydiff, ymax + 0.25 * ydiff)

    fig.autofmt_xdate(rotation=75)
    plt.minorticks_off()
    saveFigure(fig, base.joinpath('grid'))
    plt.close()


def party_plots(folder: Path, base: Path):
    """
    Creates party comparison plots
    Parameters
    ----------
    folder: Folder that contains verbosity-grouped RDD fits.
    base: Base path to store plots in
    """
    def _get_party_name(path: Path) -> str:
        model_name = path.name.split('.')[0]
        return model_name.split('_')[-1]

    model_files = [file for file in folder.iterdir() if file.name.endswith('pickle')]
    models = {_get_party_name(p): pickle.load(p.open('rb')) for p in model_files}
    features = [col for col in models['democrats'].data.columns if ('empath' in col) or ('liwc' in col)]

    party_styles = {
        'democrats': {'color': 'tab:blue', 'linewidth': 3, 'label': 'Democrats'},
        'republicans': {'color': 'tab:red', 'linewidth': 3, 'label': 'Republicans'},
    }

    for feature in features:
        fig, ax = plt.subplots(figsize=ONE_COL_FIGSIZE)
        lower, upper = (13926.0, 18578.0)  # Hard coded numeric Quotebank Date limits + margin

        y_min = np.Inf
        y_max = - np.Inf
        for party, model in models.items():
            # Adapt y-limits of the plot to the scatter values
            y_min = min(y_min, min(model.data[feature]))
            y_max = max(y_max, max(model.data[feature]))
            _scatter_only(ax, feature, model, party_styles[party]['color'])
            _rdd_only(ax, feature, model, party_styles[party])
            _conf_only(ax, feature, model, party_styles[party]['color'])

        y_diff = y_max - y_min
        ax.set_xlim(lower, upper)
        ax.set_ylim(y_min - y_diff, y_max + y_diff)
        ax.legend(loc='lower left', ncol=2, fontsize=FONTSIZE)

        title = ',  '.join([
            f'$r^2_{"{DEM, adj}"}$={models["democrats"].rdd[feature].loc["r2_adj"]:.2f}',
            f'$r^2_{"{REP, adj}"}$={models["republicans"].rdd[feature].loc["r2_adj"]:.2f}',
            r'$\sigma_{DEM}$: ' + f'{models["democrats"].data[feature].std():.2f}',
            r'$\sigma_{REP}$: ' + f'{models["republicans"].data[feature].std():.2f}',
        ])
        ax.set_title(title, fontsize=FONTSIZE)

        box = '\n'.join([
            ',  '.join([
                r'$\alpha_{1, DEM}$: ' + str(models["democrats"].get_table(asPandas=True).loc['Negemo'][r'$\alpha_1$']),
                r'$\beta_{1, DEM}$: ' + str(models["democrats"].get_table(asPandas=True).loc['Negemo'][r'$\beta_1$'])
            ]), ',  '.join([
                r'$\alpha_{1, REP}$: ' + str(models["republicans"].get_table(asPandas=True).loc['Negemo'][r'$\alpha_1$']),
                r'$\beta_{1, REP}$: ' + str(models["republicans"].get_table(asPandas=True).loc['Negemo'][r'$\beta_1$'])
            ])
        ])
        box_props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.975, 0.95, box, transform=ax.transAxes, fontsize=FONTSIZE, multialignment='center',
                verticalalignment='top', horizontalalignment='right', bbox=box_props)

        fig.autofmt_xdate(rotation=75)
        plt.minorticks_off()

        saveFigure(fig, base.joinpath(feature))
        plt.close()


def verbosity_plots(folder: Path, base: Path, verbosity_groups: Tuple[int] = (0, 3)):
    """
    Creates Verbosity comparison plots
    Parameters
    ----------
    folder: Folder that contains verbosity-grouped RDD fits.
    base: Base path to store plots in
    verbosity_groups: Selection of verbosity groups that shall be plotted
    """
    def _get_verbosity_number(path: Path) -> int:
        return int(re.search('[0-9]+', path.name)[0])

    model_files = [file for file in folder.iterdir() if file.name.endswith('pickle')]
    models = {_get_verbosity_number(p): pickle.load(p.open('rb')) for p in model_files}
    features = [col for col in models[verbosity_groups[0]].data.columns if ('empath' in col) or ('liwc' in col)]

    plasma = get_cmap('plasma')
    V_style = {
        0: {'color': plasma(0.), 'linewidth': 3, 'label': 'Verbose Politicians'},
        1: {'color': plasma(.25), 'linewidth': 3, 'label': '2nd Most Verbose Politicians'},
        2: {'color': plasma(.5), 'linewidth': 3, 'label': '3rd Most Verbose Politicians'},
        3: {'color': plasma(.75), 'linewidth': 3, 'label': 'Unknown Politicians'}
    }

    for feature in features:
        fig, ax = plt.subplots(figsize=ONE_COL_FIGSIZE)
        lower, upper = (13926.0, 18578.0)  # Hard coded numeric Quotebank Date limits + margin

        y_min = np.Inf
        y_max = - np.Inf
        for i in verbosity_groups:
            model = models[i]
            # Adapt y-limits of the plot to mean
            _, Y = model._get_rdd_plot_data(feature)
            y_min = min(y_min, min(Y))
            y_max = max(y_max, max(Y))
            _scatter_only(ax, feature, model, V_style[i]['color'])
            _rdd_only(ax, feature, model, V_style[i])
            _conf_only(ax, feature, model, V_style[i]['color'])

        y_diff = y_max - y_min
        ax.set_xlim(lower, upper)
        ax.set_ylim(y_min - y_diff, y_max + y_diff)
        ax.legend(loc='lower left', ncol=2, fontsize=FONTSIZE)

        title = ',  '.join([
            f'$r^2_{"{Ver, adj}"}$={models[0].rdd[feature].loc["r2_adj"]:.2f}',
            f'$r^2_{"{Unk, adj}"}$={models[3].rdd[feature].loc["r2_adj"]:.2f}',
            r'$\sigma_{Ver}$: ' + f'{models[0].data[feature].std():.2f}',
            r'$\sigma_{Unk}$: ' + f'{models[3].data[feature].std():.2f}',
        ])
        ax.set_title(title, fontsize=FONTSIZE)

        box = '\n'.join([
            ',  '.join([
                r'$\alpha_{1, Ver}$: ' + str(models[0].get_table(asPandas=True).loc['Negemo'][r'$\alpha_1$']),
                r'$\beta_{1, Ver}$: ' + str(models[0].get_table(asPandas=True).loc['Negemo'][r'$\beta_1$'])
            ]), ',  '.join([
                r'$\alpha_{1, Unk}$: ' + str(models[3].get_table(asPandas=True).loc['Negemo'][r'$\alpha_1$']),
                r'$\beta_{1, Unk}$: ' + str(models[3].get_table(asPandas=True).loc['Negemo'][r'$\beta_1$'])
            ])
        ])
        box_props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.975, 0.95, box, transform=ax.transAxes, fontsize=FONTSIZE, multialignment='center',
                verticalalignment='top', horizontalalignment='right', bbox=box_props)

        fig.autofmt_xdate(rotation=75)
        plt.minorticks_off()

        saveFigure(fig, base.joinpath(feature))
        plt.close()


def main():
    args = parser.parse_args()
    data = Path(args.rdd)
    img = Path(args.img)

    # Map a file or folder name to a plotting utility.
    NAME_2_FUNCTION = {
        'verbosity': verbosity_plots,
        'parties': party_plots,
        'QuotationAggregation_RDD': basic_model_plots,
        'SpeakerAggregation_RDD': basic_model_plots
    }

    for path in data.iterdir():
        if path.name.endswith('tex'):
            continue

        base_name = path.name.split('.')[0]
        if base_name not in NAME_2_FUNCTION:
            continue

        print(base_name)
        base_folder = img.joinpath(base_name)
        plot = NAME_2_FUNCTION[base_name]
        plot(path, base_folder)


if __name__ == '__main__':
    main()
