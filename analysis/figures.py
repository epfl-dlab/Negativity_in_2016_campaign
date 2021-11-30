import argparse
from copy import deepcopy
import collections
from datetime import datetime
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pickle
from pathlib import Path
import re
import seaborn as sns
import string
import sys
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from utils.plots import saveFigure, timeLinePlot, ONE_COL_FIGSIZE, TWO_COL_FIGSIZE, NARROW_TWO_COL_FIGSIZE
from analysis.RDD import RDD, aicc, KINK

parser = argparse.ArgumentParser()
parser.add_argument('--rdd', help='Folder containing fitted RDDs', required=True)
parser.add_argument('--img', help='Folder to write images to.', required=True)
parser.add_argument('--SI', help='SI material', required=False)

FONTSIZE = 14
CORE_FEATURES = ['liwc_Negemo', 'liwc_Anger', 'liwc_Anx', 'liwc_Sad']
LIWC_FEATURES = CORE_FEATURES + ['liwc_Posemo', 'liwc_Swear']
NAMES = {
    'liwc_Negemo': 'Negative emotion',
    'liwc_Anx': 'Anxiety',
    'liwc_Anger': 'Anger',
    'liwc_Sad': 'Sadness',
    'liwc_Swear': 'Swear words',
    'liwc_Posemo': 'Positive emotion',
    'linreg': 'Linear Regression',
    'liwc_Certain': 'Certainty',
    'liwc_Tentat': 'Tentativeness',
    'empath_negative_emotion': 'Neg. Emotion (empath)',
    'empath_positive_emotion': 'Pos. Emotion (empath)',
    'empath_science': 'Science (empath)',
    'empath_swearing_terms': 'Swearing (empath)',
    'verbosity_vs_alpha': r'RDD $\alpha$ over verbosity',
    'verbosity_vs_beta': r'RDD $\beta$ over verbosity',
    'r2_adj': 'Adjusted $r^2$ score',
    'r2_adj_posemo': 'Adjusted $r^2$ score for positive emotions'
}


def _default_style(): return {'color': 'tab:grey', 'linewidth': 2.5}


STYLES = collections.defaultdict(_default_style)
STYLES['liwc_Negemo'] = {'color': 'tab:red', 'linewidth': 3}
STYLES['liwc_Anx'] = {'color': 'darkorange', 'linewidth': 2.5}
STYLES['liwc_Anger'] = {'color': 'darkorange', 'linewidth': 2.5}
STYLES['liwc_Sad'] = {'color': 'darkorange', 'linewidth': 2.5}
STYLES['liwc_Swear'] = {'color': 'black', 'linewidth': 2.5}
STYLES['liwc_Posemo'] = {'color': 'tab:green', 'linewidth': 3}
STYLES['linreg'] = {'color': 'black', 'linewidth': 1.5, 'linestyle': '-.'}

PARTY_STYLES = {
    'democrats': {'color': 'tab:blue', 'linewidth': 2.5, 'scatter_color': 'tab:blue', 'label': 'Democrats'},
    'republicans': {'color': 'tab:red', 'linewidth': 2.5, 'scatter_color': 'tab:red', 'label': 'Republicans'}
}
POLITICIAN_IDS = {
    'Q76': 'Barack Obama',
    'Q6294': 'Hillary Clinton',
    'Q22686': 'Donald Trump',
    'Q69319': 'John Kasich',
    'Q221997': 'Jeb Bush',
    'Q324546': 'Marco Rubio',
    'Q359442': 'Bernie Sanders ',
    'Q816459': 'Ben Carson',
    'Q2036942': 'Ted Cruz'
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
    del kwargs['scatter_color']
    X_rdd, Y_rdd = model._get_rdd_plot_data(feature)
    dates_rdd = [model._get_approx_date(x) for x in X_rdd]
    for i in range(len(dates_rdd) // 2):
        if i > 0:
            kwargs['label'] = ''
        timeLinePlot([dates_rdd[2 * i], dates_rdd[2 * i + 1]], [Y_rdd[2 * i], Y_rdd[2 * i + 1]], ax=ax, snsargs=kwargs)


def _scatter_only(ax: plt.axis, feature: str, model: RDD, color: str, s: int = 25):
    timeLinePlot(model.data.date, model.data[feature], ax=ax, snsargs={'s': s, 'color': color}, kind='scatter')


def _grid_annotate(ax: plt.axis, model: RDD, feature: str, title: str):
    ax.set_title(title, fontsize=FONTSIZE, fontweight='bold')

    txt = r"{\mathrm{adj}}"
    r2 = f'$R^2_{txt}$={model.rdd[feature].loc["r2_adj"]:.2f}'
    params = ',  '.join([
        r'$\alpha$=' + model.get_table(asPandas=True)[r'$\alpha$'].loc[' '.join(feature.split('_')[1:])].split('(')[0],
        r'$\beta$=' + model.get_table(asPandas=True)[r'$\beta$'].loc[' '.join(feature.split('_')[1:])].split('(')[0]
    ])
    box_props = dict(boxstyle='round', facecolor='white', alpha=1)

    ax.text(0.025, 0.05, r2, transform=ax.transAxes, fontsize=FONTSIZE, verticalalignment='bottom',
            horizontalalignment='left', bbox=box_props)
    ax.text(0.975, 0.95, params, transform=ax.transAxes, fontsize=FONTSIZE, verticalalignment='top',
            horizontalalignment='right', bbox=box_props)


def grid(models: Dict[str, RDD], ncols: int, nrows: int, features: List[str], style: Dict, gridspec: bool = False,
         **kwargs):
    fontsize = kwargs.get('fontsize', FONTSIZE)
    names = kwargs.get('names', NAMES)
    if gridspec:
        assert len(features) == 5, "Other gridspecs not supported."
        fig = plt.figure(figsize=TWO_COL_FIGSIZE)
        gs = fig.add_gridspec(2, 6)
        axs = [
            fig.add_subplot(gs[0, :3]),
            fig.add_subplot(gs[0, 3:6]),
            fig.add_subplot(gs[1, :2]),
            fig.add_subplot(gs[1, 2:4]),
            fig.add_subplot(gs[1, 4:6]),
        ]
    elif kwargs.get('axs', None) is not None:
        fig = plt.gcf()
        axs = kwargs.get('axs')
    else:
        fig, axs = plt.subplots(figsize=NARROW_TWO_COL_FIGSIZE, ncols=ncols, nrows=nrows, sharex='all', sharey='all')
    ymin = np.inf
    ymax = - np.inf
    for name, model in models.items():
        for i, feature in enumerate(features):
            COL = i % nrows
            ROW = i // nrows
            if isinstance(axs[0], np.ndarray):
                ax = axs[COL][ROW]
            else:
                ax = axs[i]
            ax.set_title(names[feature], fontsize=fontsize, fontweight='bold')
            ax.set_title(string.ascii_lowercase[i] + ')', fontfamily='serif', loc='left', fontsize=FONTSIZE + 4,
                         fontweight='bold')  # Subplot naming
            try:
                selectedStyle = style[feature]
            except KeyError:
                try:
                    selectedStyle = style[name][feature]
                except KeyError:
                    selectedStyle = style[name]

            model.plot(feature,
                       ax=ax,
                       annotate=kwargs.get('annotate', False),
                       lin_reg=kwargs.get('lin_reg', False),
                       visuals=kwargs.get('visuals', False),
                       **selectedStyle)
            if kwargs.get('grid_annotate', False):
                _grid_annotate(ax, model, feature, NAMES[feature])
            if kwargs.get('mean_adapt', False):
                _, Y = model._get_rdd_plot_data(feature)
                ymin = min(ymin, min(Y))
                ymax = max(ymax, max(Y))
            else:
                ymin = min(ymin, min(model.data[feature]))
                ymax = max(ymax, max(model.data[feature]))

    ydiff = ymax - ymin
    if gridspec:
        axs = [axs]  # Quick hack to allow the following iteration
    for row in axs:
        if not isinstance(row, np.ndarray):
            row = [row]
        for ax in row:
            if kwargs.get('legend', False):
                ax.legend(fontsize=fontsize, loc='lower left', framealpha=1, fancybox=False,
                          ncol=kwargs.get('legendCols', 1))
            else:
                try:
                    ax.get_legend().remove()
                except AttributeError:
                    pass  # There was no legend to begin with
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            if kwargs.get('mean_adapt', False):
                ax.set_ylim(ymin - ydiff, ymax + ydiff)
            else:
                ax.set_ylim(ymin - 0.25 * ydiff, ymax + 0.25 * ydiff)
            ax.set_xlim(13926.0, 18578.0)

    fig.autofmt_xdate(rotation=75)
    plt.minorticks_off()
    return fig, axs


def basic_model_plots(model_file: Path, base: Path, ylims: Tuple[float, float] = None):
    """
    Creates a single plot for every feature the RDD was fitted on.
    Parameters
    ----------
    model_file: Path to an RDD Model
    base: Base folder to store results in
    ylims: Another way to control the y axis limits: They can be directly provided here.
    -------
    """
    model = pickle.load(model_file.open('rb'))

    # Single-Feature Plots
    # for f in sorted(features):
    #     lin_reg = f != 'liwc_Posemo'
    #     fig, ax = model.plot(f, parameters='all', **baseStyle, lin_reg=lin_reg)
    #     if ylims is not None:
    #         ax.set_ylim(ylims[0], ylims[1])
    #     ax.set_xlim(13926.0, 18578.0)  # QB
    #     saveFigure(fig, base.joinpath(f))
    #     plt.close()

    # 5x grid
    fig, axs = grid({'quoteAggregation': model}, 5, 1, CORE_FEATURES + ['liwc_Swear'], STYLES, grid_annotate=True)
    saveFigure(fig, base.joinpath('negative_and_swear_grid'))

    # 2 x 3 Grid only scatter
    fig, axs = plt.subplots(figsize=TWO_COL_FIGSIZE, ncols=3, nrows=2, sharex='all', sharey='all')
    ymin, ymax = np.inf, -np.inf
    for i, feature in enumerate(LIWC_FEATURES):
        COL = i % 2
        ROW = i // 2
        ax = axs[COL][ROW]
        _scatter_only(ax, feature, model, 'black')
        ax.set_title(NAMES[feature], fontweight='bold', fontsize=FONTSIZE)
        ymin = min(ymin, min(model.data[feature]))
        ymax = max(ymax, max(model.data[feature]))

    ydiff = ymax - ymin
    for row in axs:
        for ax in row:
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
            ax.set_ylim(ymin - 0.25 * ydiff, ymax + 0.25 * ydiff)
            if ylims is not None:
                ax.set_ylim(ylims[0], ylims[1])

    fig.autofmt_xdate(rotation=75)
    plt.minorticks_off()
    saveFigure(fig, base.joinpath('scatter_grid'))
    plt.close()


def outlier_plots(model_file: Path, store: Path):
    """
    Creates a comparison between "basic" and "outliers removed" for every feature the RDD was fitted on.
    Parameters
    ----------
    model_file: Path to an RDD Model, including outliers. Naming of the non-outlier model must follow project structure.
    store: Base folder to store results in
    -------

    """
    outliers = pickle.load(model_file.open('rb'))
    base_model = pickle.load(model_file.parent.joinpath(re.sub('_outliers', '', model_file.name)).open('rb'))

    models = collections.OrderedDict()
    models['With Outliers'] = base_model
    models['Without Outliers'] = outliers

    outlierStyle = {k: STYLES[k] for k in STYLES}
    baseStyle = {k: _default_style() for k in STYLES}
    for k in baseStyle:
        outlierStyle[k]['label'] = 'With Outliers'
        outlierStyle[k]['scatter_color'] = STYLES[k]['color']
        baseStyle[k]['label'] = 'Without Outliers'
        baseStyle[k]['scatter_color'] = 'tab:grey'

    fig, axs = grid(models, 5, 1, CORE_FEATURES + ['liwc_Swear'],
                    {'With Outliers': outlierStyle, 'Without Outliers': baseStyle}, mean_adapt=True, legend=True)

    saveFigure(fig, store.joinpath('negative_and_swear_grid'))


def individual_plots(folder: Path, base: Path):
    """
    Creates all feature plots for every feature for every individual RDD in the given folder.
    Parameters
    ----------
    folder: Parent folder, containing RDDs fitted on individual aggregates
    base: Base folder to store figures in
    """
    for file in folder.iterdir():
        if not file.name.endswith('pickle'):
            continue
        qid = file.name.split('_')[0]
        if qid not in POLITICIAN_IDS:
            continue
        clearname = POLITICIAN_IDS[qid]
        if 'outlier' in file.name:
            # outlier_plots(file, save_in.joinpath(clearname + '_outlier'))
            pass  # We don't really need that
        else:
            basic_model_plots(file, base.joinpath(clearname), ylims=(-10, 20))


def verbosity_vs_parameter(folder: Path, base: Path, kind: str, alpha_CI: float = 0.05):
    """
    Plots RDD parameters (y-Axis) vs speaker verbosity (x-axis)
    Parameters
    ----------
    folder: RDD models folder
    base: Img storage folder
    kind: Either "individual" or "ablation" - changes style adjustments
    alpha: Confidence Interval parameter
    """
    verbosity = folder.parent.parent.joinpath('speaker_counts.csv')
    assert verbosity.exists(), "To create the scatter plot influence / verbosity, there needs to be a speaker count file."
    base_model_path = folder.parent.joinpath('QuotationAggregation_RDD.pickle')
    assert base_model_path.exists(), "To create the scatter plot influence / verbosity, there needs to be a Quotation Aggregation file."
    base_model = pickle.load(base_model_path.open('rb'))

    def _get_qid(s: str) -> str:
        return re.search('Q[0-9]+', s)[0]

    names = list(set([_get_qid(speaker.name) for speaker in folder.iterdir() if 'outlier' not in speaker.name]))

    verbosity_df = pd.read_csv(verbosity)
    verbosity_df = verbosity_df[verbosity_df.QID.isin(names)]
    plot_data = {
        feature: pd.DataFrame(columns=names,
                              index=['aCORE_FEATURESlpha', 'beta', 'verbosity', 'alpha_low', 'alpha_high', 'beta_low',
                                     'beta_high'])
        for feature in CORE_FEATURES + ['liwc_Swear']
    }

    for speaker in folder.iterdir():
        if not (speaker.name.endswith('.pickle')):
            continue
        qid = _get_qid(speaker.name)
        speaker_data = pickle.load(speaker.open('rb'))
        # params = speaker_data.rdd
        for feature in plot_data:
            summary = pd.read_html(speaker_data.rdd_fit[feature].summary(alpha=alpha_CI).tables[1].as_html(), header=0,
                                   index_col=0)[0]
            lower_CI, upper_CI = summary.columns[4:6]
            alpha_low, alpha_high = summary[lower_CI].loc['C(threshold)[T.1]'], summary[upper_CI].loc[
                'C(threshold)[T.1]']
            beta_low, beta_high = summary[lower_CI].loc['C(threshold)[T.1]:time_delta'], summary[upper_CI].loc[
                'C(threshold)[T.1]:time_delta']
            alpha = summary['coef'].loc['C(threshold)[T.1]']
            beta = summary['coef'].loc['C(threshold)[T.1]:time_delta']
            plot_data[feature].at['alpha', qid] = alpha
            plot_data[feature].at['alpha_low', qid] = alpha_low
            plot_data[feature].at['alpha_high', qid] = alpha_high
            plot_data[feature].at['beta', qid] = beta
            plot_data[feature].at['beta_low', qid] = beta_low
            plot_data[feature].at['beta_high', qid] = beta_high
            plot_data[feature].at['verbosity', qid] = verbosity_df[verbosity_df.QID == qid]['Unique Quotations'].values[
                0]

    for param in ('alpha', 'beta'):
        fig, axs = plt.subplots(figsize=NARROW_TWO_COL_FIGSIZE, ncols=5, nrows=1, sharex='all', sharey='all')
        for i, (feature, data) in enumerate(plot_data.items()):
            ax = axs[i]
            ax.set_xscale('log')
            if kind == 'individual':
                ax.axhline(y=0, linestyle='--', color='black', linewidth=0.8)
            else:
                key = 'C(threshold)[T.1]' if param == 'alpha' else 'C(threshold)[T.1]:time_delta'
                baseline = base_model.rdd[feature][key]
                summary = \
                    pd.read_html(base_model.rdd_fit[feature].summary(alpha=alpha_CI).tables[1].as_html(), header=0,
                                 index_col=0)[0]
                base_low, base_high = summary[lower_CI].loc[key], summary[upper_CI].loc[key]
                ax.axhline(y=baseline, linestyle='--', color='black')
                ax.fill_between((min(verbosity_df['Unique Quotations']), max(verbosity_df['Unique Quotations'])),
                                base_low, base_high, color='grey', alpha=0.3)
            for qid in data.columns:
                isTrump = int(qid == 'Q22686')
                CI_low, CI_high = data[qid].loc[param + '_low'], data[qid].loc[param + '_high']
                # Highlight "significant" points where CIs share a sign
                if kind == 'individual':
                    color = 'tab:red' if (CI_low * CI_high > 0) else 'grey'
                else:
                    color = 'tab:red' if (base_low > CI_high) else 'grey'
                ax.plot((data[qid].loc['verbosity'], data[qid].loc['verbosity']), (CI_low, CI_high), '-', color=color,
                        linewidth=0.3 + 1 * isTrump)
                ax.scatter(x=data[qid].loc['verbosity'], y=data[qid].loc[param], c=color, s=7.5 * (1 + 3 * isTrump))
            if kind == 'individual':
                annot = r'$\left|\{' + rf'\{param}>0' + r'\}' + rf'\right|={(data.loc[param] > 0).sum()}$' + \
                        r', $\left|\{' + rf'\{param}<0' + r'\}' + rf'\right|={(data.loc[param] < 0).sum()}$'
                box_props = dict(boxstyle='round', facecolor='white', alpha=1)
                ax.text(0.975, 0.05, annot, transform=ax.transAxes, fontsize=FONTSIZE, multialignment='center',
                        verticalalignment='bottom', horizontalalignment='right', bbox=box_props)

            ax.set_title(NAMES[feature], fontsize=FONTSIZE, fontweight='bold')
            ax.set_title(string.ascii_lowercase[i] + ')', fontfamily='serif', loc='left', fontsize=FONTSIZE + 4,
                         fontweight='bold')
            if i > 0:
                # ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', left=False, right=False)
            else:
                ax.set_ylabel(r'$\{}$'.format(param))
                if (param == 'alpha') and (kind == 'individual'):
                    ax.set_ylim(-20, 20)
                    ax.set_yticks([-10, 10])
                    ax.set_yticklabels(['-10', '10'])
            # if i <= 2:
            #     ax.set_xticklabels([])
            #     ax.tick_params(axis='x', which='both', bottom=False)
            ax.set_xlabel('Uttered Quotations')
        if alpha_CI != 0.05:
            fig.suptitle('{:.2f}% Confidence Intervals'.format(1 - alpha_CI), fontweight='bold', fontsize=FONTSIZE + 2)
        saveFigure(fig, base.joinpath('verbosity_vs_{}'.format(param)))
        plt.close()


def ablation_plots(folder: Path, base: Path):
    """
    Generates two kinds of plots: One summarizing ablation plots, showing verbosity vs. the influence of leaving one
    speaker out on the overall parameters and then an "individual" plot for the newly fitted lines for all speakers
    that can be mapped to their real world names.
    Parameters
    ----------
    folder: Parent folder, containing RDDs fitted on data from all-but-one-individual each
    base: Base folder where to store figures in
    """
    verbosity_vs_parameter(folder, base, 'ablation', alpha_CI=0.17)


def individuals(folder: Path, base: Path):
    """
    Creates all feature plots for every feature for every individual RDD in the given folder.
    Parameters
    ----------
    folder: Parent folder, containing RDDs fitted on individual aggregates
    base: Base folder to store figures in
    """
    individual_plots(folder, base)
    verbosity_vs_parameter(folder, base, 'individual')


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

    model_files = [file for file in folder.iterdir() if file.name.endswith('pickle') and ('outliers' not in file.name)]
    models = {_get_party_name(p): pickle.load(p.open('rb')) for p in model_files}
    features = [col for col in models['democrats'].data.columns if ('empath' in col) or ('liwc' in col)]

    for feature in features:
        fig, ax = plt.subplots(figsize=ONE_COL_FIGSIZE)
        lower, upper = (13926.0, 18578.0)  # Hard coded numeric Quotebank Date limits + margin

        y_min = np.Inf
        y_max = - np.Inf
        for party, model in models.items():
            # Adapt y-limits of the plot to the scatter values
            y_min = min(y_min, min(model.data[feature]))
            y_max = max(y_max, max(model.data[feature]))
            _scatter_only(ax, feature, model, PARTY_STYLES[party]['color'], s=40)
            _rdd_only(ax, feature, model, PARTY_STYLES[party])
            _conf_only(ax, feature, model, PARTY_STYLES[party]['color'])

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

        hrFeature = ' '.join(feature.split('_')[1:])  # Human readable, as used in the tables
        box = '\n'.join([
            ',  '.join([
                r'$\alpha_{0, DEM}$: ' + str(
                    models["democrats"].get_table(asPandas=True).loc[hrFeature][r'$\alpha_0$']),
                r'$\beta_{0, DEM}$: ' + str(models["democrats"].get_table(asPandas=True).loc[hrFeature][r'$\beta_0$']),
                r'$\alpha_{DEM}$: ' + str(models["democrats"].get_table(asPandas=True).loc[hrFeature][r'$\alpha$']),
                r'$\beta_{DEM}$: ' + str(models["democrats"].get_table(asPandas=True).loc[hrFeature][r'$\beta$']),
            ]), ',  '.join([
                r'$\alpha_{0, REP}$: ' + str(
                    models["republicans"].get_table(asPandas=True).loc[hrFeature][r'$\alpha_0$']),
                r'$\beta_{0, REP}$: ' + str(
                    models["republicans"].get_table(asPandas=True).loc[hrFeature][r'$\beta_0$']),
                r'$\alpha_{REP}$: ' + str(models["republicans"].get_table(asPandas=True).loc[hrFeature][r'$\alpha$']),
                r'$\beta_{REP}$: ' + str(models["republicans"].get_table(asPandas=True).loc[hrFeature][r'$\beta$'])
            ])
        ])
        box_props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.975, 0.95, box, transform=ax.transAxes, fontsize=FONTSIZE - 4, multialignment='center',
                verticalalignment='top', horizontalalignment='right', bbox=box_props)

        fig.autofmt_xdate(rotation=75)
        plt.minorticks_off()

        saveFigure(fig, base.joinpath(feature))
        plt.close()

    # Grid
    fig, axs = grid(models, 5, 1, CORE_FEATURES + ['liwc_Swear'], PARTY_STYLES, legend=True)
    saveFigure(fig, base.joinpath('grid'))
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

    model_files = [file for file in folder.iterdir() if file.name.endswith('pickle') and ('outliers' not in file.name)]
    models = {_get_verbosity_number(p): pickle.load(p.open('rb')) for p in model_files}
    features = [col for col in models[verbosity_groups[0]].data.columns if ('empath' in col) or ('liwc' in col)]

    plasma = get_cmap('plasma')
    V_style = {
        0: {'color': plasma(0.), 'linewidth': 3, 'label': 'Most prominent speaker quartile'},
        1: {'color': plasma(.25), 'linewidth': 3, 'label': '2nd most prominent speaker quartile'},
        2: {'color': plasma(.5), 'linewidth': 3, 'label': '3rd most prominent speaker quartile'},
        3: {'color': plasma(.75), 'linewidth': 3, 'label': 'Least most prominent speaker quartile'}
    }

    lower, upper = (13926.0, 18578.0)  # Hard coded numeric Quotebank Date limits + margin
    for feature in features:
        fig, axs = plt.subplots(figsize=NARROW_TWO_COL_FIGSIZE, ncols=4)
        y_min = np.Inf
        y_max = - np.Inf

        for verbosity, model in models.items():
            ax = axs[verbosity]
            model.plot(feature, ax=ax, parameters=True, visuals=False, color=STYLES[feature]['color'], lin_reg=False,
                       annSize=FONTSIZE - 2)
            # Adapt y-limits of the plot to mean
            _, Y = model._get_rdd_plot_data(feature)
            y_min = min(y_min, min(Y))
            y_max = max(y_max, max(Y))
            ax.get_legend().remove()
            ax.set_title(NAMES[feature] + '\n' + V_style[verbosity]['label'], fontsize=FONTSIZE - 2, fontweight='bold')
            txt = r'{\mathrm{adj}}'
            r2 = f'$R^2_{txt}$={model.rdd[feature].loc["r2_adj"]:.2f}'
            box_props = dict(boxstyle='round', facecolor='white', alpha=1)
            ax.text(0.025, 0.05, r2, transform=ax.transAxes, fontsize=FONTSIZE, verticalalignment='bottom',
                    horizontalalignment='left', bbox=box_props)

        y_diff = y_max - y_min
        for ax in axs:
            ax.set_xlim(lower, upper)
            ax.set_ylim(y_min - y_diff, y_max + y_diff)

        fig.autofmt_xdate(rotation=75)
        plt.minorticks_off()

        saveFigure(fig, base.joinpath(feature))
        plt.close()


def attribute_plots(model_path: Path, base: Path):
    attributes = ['party', 'governing_party', 'gender', 'congress_member',
                  'C(threshold)[T.1]', 'C(threshold)[T.1]:time_delta']
    model = pickle.load(model_path.open('rb'))

    ticks = {
        'gender': ['Male', 'Female'],
        'party': ['Republican', 'Democratic'],
        'congress_member': ['Others', 'Congress'],
        'governing_party': ['Opposition', 'Government']
    }
    titles = {
        'gender': r'$\mathbf{\eta}$: gender',
        'party': r'$\mathbf{\gamma}$: party affiliation',
        'congress_member': r'$\mathbf{\zeta}$: Congress membership',
        'governing_party': r'$\mathbf{\delta}$: party\'s federal role',
        'C(threshold)[T.1]': r'$\mathbf{\alpha}$',
        'C(threshold)[T.1]:time_delta': r'$\mathbf{\beta}$'
    }

    styles_cpy = {NAMES[key]: val for key, val in STYLES.items()}

    ORDER = ['liwc_Negemo', 'liwc_Anger', 'liwc_Anx', 'liwc_Sad']

    fig, axs = plt.subplots(figsize=TWO_COL_FIGSIZE, ncols=3, nrows=2)
    for i, att in enumerate(attributes):
        ROW = i % 2
        COL = i // 2
        ax = axs[ROW][COL]
        df = pd.DataFrame(data=None, index=CORE_FEATURES, columns=['mean', 'low', 'high'])

        for feat in CORE_FEATURES:
            summary = pd.read_html(model.rdd_fit[feat].summary().tables[1].as_html(), header=0, index_col=0)[0]
            lower, upper = summary['[0.025'].loc[att], summary['0.975]'].loc[att]
            mean = summary['coef'].loc[att]
            df.loc[feat] = (mean, lower, upper)

        df = df.reindex(ORDER[::-1])

        for _, r in df.iterrows():
            name = NAMES[r.name]
            color = styles_cpy[name]['color']
            ax.plot((r.low, r.high), (name, name), '|-', color='black', linewidth=1.33)
            ax.plot(r['mean'], name, 'o', color=color, markersize=7.5)

        if 'time_delta' not in att:  # All but beta
            ax.set_xlim([-2, 4.5])
            ax.set_xticks([-1, 0, 1, 2, 3, 4])
            # ax.text(x=0.05, y=-0.2, s=r'{}$\longleftarrow$'.format(ticks[att][0]), fontsize=FONTSIZE, ha='left', va='bottom', transform=ax.transAxes)
            # ax.text(x=0.95, y=-0.2, s=r'$\longrightarrow${}'.format(ticks[att][1]), fontsize=FONTSIZE, ha='right', va='bottom', transform=ax.transAxes)
            ax.text(x=0.5, y=-0.2, s='Pre-campaign SD', fontsize=FONTSIZE, ha='center', va='bottom',
                    transform=ax.transAxes)

        ax.tick_params(axis='both', labelsize=FONTSIZE)
        ax.set_title(titles[att], fontsize=FONTSIZE, fontweight='bold')
        ax.set_title(string.ascii_lowercase[i] + ')', fontfamily='serif', loc='left', fontsize=FONTSIZE + 4,
                     fontweight='bold')
        ax.axvline(x=0, linestyle='dashed', color='black', linewidth=0.5)  # TODO: Distance between dashes
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower - .5, upper + .5)

        if COL == 1:
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', left=False, right=False)

        if COL == 2:
            ax.yaxis.tick_right()
            if ROW == 1:
                locs, labels = ax.get_xticks(), ax.get_xticklabels()
                ax.set_xticks(locs[1::2])
                ax.set_xticklabels([f'{loc:.3f}' for loc in locs[1::2]])

    saveFigure(fig, base.joinpath('selected_attributes'))
    plt.close()


def plot_quantities(data_path: Path, base: Path):
    """
    Plots some quantitative overview stats.
    Parameters
    ----------
    data_path: csv file
    base: Image folder
    """
    df = pd.read_csv(data_path)
    df['dt_month'] = df.apply(lambda r: datetime(int(r.year), int(r.month), 15), axis=1)
    fig, axs = plt.subplots(figsize=TWO_COL_FIGSIZE, nrows=3)
    for i, col, name in ((0, 'num_speaker', 'Monthly Speakers'), (1, 'num_quotes', 'Unique Monthly Quotations'),
                         (2, 'total_domains', 'Total monthly quotation domains')):
        fig, ax = timeLinePlot(x=df.dt_month,
                               y=df[col],
                               kind='scatter',
                               title=name,
                               ax=axs[i])

    saveFigure(fig, base.joinpath('quantities.pdf'))


def RDD_kink_performance(data_path: Path, base: Path):
    """
    Plots the performance of the RDD, depending on the date of the discontinuity / kink
    """
    data = pickle.load(data_path.open('rb'))
    dates = list(data.keys())
    fig, ax = plt.subplots(figsize=TWO_COL_FIGSIZE)

    rdd_style = deepcopy(STYLES)
    rdd_style['liwc_Anx']['linestyle'] = '--'
    rdd_style['liwc_Anger']['linestyle'] = '-.'
    rdd_style['liwc_Sad']['linestyle'] = 'dotted'

    for feature in CORE_FEATURES + ['liwc_Swear']:
        timeLinePlot(
            x=dates,
            y=[data[dt][feature]['r2_adjust'] for dt in dates],
            snsargs=dict(label=NAMES[feature], **rdd_style[feature]),
            ax=ax,
            includeElections=False
        )
    ax.axvline(mdates.date2num(KINK), color='black', linestyle='--')
    plt.legend(fontsize=FONTSIZE - 2, loc='upper right', framealpha=1, fancybox=False, ncol=3)
    saveFigure(fig, base.joinpath('r2_adj'))

    fig, ax = plt.subplots(figsize=TWO_COL_FIGSIZE)
    timeLinePlot(
        x=dates,
        y=[data[dt]['liwc_Posemo']['r2_adjust'] for dt in dates],
        snsargs=dict(label=NAMES['liwc_Posemo'], **rdd_style['liwc_Posemo']),
        ax=ax,
        includeElections=False)
    ax.axvline(mdates.date2num(KINK), color='black', linestyle='--')
    plt.legend(fontsize=FONTSIZE - 2, loc='upper right', framealpha=1, fancybox=False, ncol=3)
    saveFigure(fig, base.joinpath('r2_adj_posemo'))
    plt.close('all')


def main_figure(president_verbosity: pd.DataFrame, raw_negemo: pd.DataFrame):
    """
    Plots the raw liwc score of the negative emotion liwc category vs. the relative share of quotes
    uttered by Donald Trump vs Barack Obama
    """
    dates = sorted(raw_negemo['date'].unique())
    shares = pd.DataFrame(data=None, columns=['trump', 'obama', 'total'],
                          index=dates)
    for dt in shares.index:
        total = president_verbosity[president_verbosity.date == dt]['numQuotes'].sum()
        trump = president_verbosity[(president_verbosity.date == dt) & (president_verbosity.qid == 'Q22686')] \
                ['numQuotes'].iloc[0]
        obama = president_verbosity[(president_verbosity.date == dt) & (president_verbosity.qid == 'Q76')] \
            ['numQuotes'].iloc[0]
        shares.at[dt, 'trump'] = trump
        shares.at[dt, 'obama'] = obama
        shares.at[dt, 'total'] = total

    leftStyle = {'s': 40, 'color': STYLES['liwc_Negemo']['color']}
    fig, left_axis = plt.subplots(figsize=TWO_COL_FIGSIZE)
    right_axis = left_axis.twinx()
    fig, left_axis = timeLinePlot(x=dates, y=raw_negemo.liwc_Negemo,
                                  snsargs=leftStyle,
                                  kind='scatter',
                                  ax=left_axis)

    rightStyle = {'linewidth': 2.5, 'color': 'tab:blue'}
    fig, right_axis = timeLinePlot(x=dates, y=np.asarray((shares.trump - shares.obama) / shares.total, dtype=float),
                                   ax=right_axis,
                                   snsargs=rightStyle)

    left_axis.set_ylabel('Negative emotion score', color=leftStyle['color'])
    left_axis.set_title('TITLE NEEDED')
    left_axis.set_ylim([0, 2 * raw_negemo.liwc_Negemo.mean()])
    left_axis.set_xlim(13956.0, 18548.0)

    right_axis.axhline(y=0, linestyle='--', color='black', linewidth=1)
    right_axis.set_ylabel('Share of captured quotes', color=rightStyle['color'])
    right_axis.set_ylim([-1.1, 1.1])
    right_axis.set_yticks([-1, 0, 1])
    right_axis.set_yticklabels(['Obama Only', 'Balanced', 'Trump Only'])

    right_axis.minorticks_off()
    plt.show()
    print('debug Stop')


def main():
    args = parser.parse_args()
    data = Path(args.rdd)
    img = Path(args.img)

    # Map a file or folder name to a plotting utility.
    NAME_2_FUNCTION = {
        # 'verbosity': verbosity_plots,
        # 'parties': party_plots,
        # 'Individuals': individuals,
        # 'Without': ablation_plots,
        # 'QuotationAggregation_RDD': basic_model_plots,
        # 'QuotationAggregationTrump_RDD': basic_model_plots,
        # 'QuotationAggregation_RDD_outliers': outlier_plots,
        # 'SpeakerAggregation_RDD': basic_model_plots,
        # 'SpeakerAggregation_RDD_outliers': outlier_plots,
        # 'AttributesAggregation_RDD': attribute_plots,
        # 'RDD_time_variation': RDD_kink_performance
    }

    basic_model = None

    for path in data.iterdir():
        if path.name.endswith('tex'):
            continue

        base_name = path.name.split('.')[0]
        if base_name not in NAME_2_FUNCTION:
            continue

        if base_name == 'QuotationAggregation_RDD':
            basic_model = pickle.load(open(path, 'rb'))

        print(base_name)
        base_folder = img.joinpath(base_name)
        plot = NAME_2_FUNCTION[base_name]
        plot(path, base_folder)

    if args.SI is not None:
        si_data = Path(args.SI)
        si_img = img.joinpath('SI')
        si_img.mkdir(exist_ok=True)
        quant = si_data.joinpath('quantitative_statisticts.csv')
        if quant.exists():
            plot_quantities(quant, si_img)

    # Build "Fig. 0"
    aggregates = data.parent.joinpath('aggregates')
    presidents = pd.read_csv(
        aggregates.joinpath('presidents.csv').open('r'))  # Contains Obamas and Trumps number of Quotes
    negemo = pd.read_csv(aggregates.joinpath('QuotationAggregation.csv').open('r'))[['liwc_Negemo', 'date']]
    negemo['liwc_Negemo'] = negemo['liwc_Negemo'] \
                            * pickle.load(aggregates.joinpath('std.pickle').open('rb'))['liwc_Negemo'] \
                            + pickle.load(aggregates.joinpath('mean.pickle').open('rb'))[
                                'liwc_Negemo']  # Restore the original liwc scores
    main_figure(presidents, negemo)


if __name__ == '__main__':
    main()
