import argparse
from copy import deepcopy
import collections
from datetime import datetime
import numpy as np
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from matplotlib.text import Text as mText
import pandas as pd
import pickle
from pathlib import Path
import re
import string
import sys
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from utils.plots import saveFigure, timeLinePlot, ONE_COL_FIGSIZE, TWO_COL_FIGSIZE, NARROW_TWO_COL_FIGSIZE, \
    NARROW_NARROW_TWO_COL_FIGSIZE, PRESIDENTIAL_ELECTIONS, LANDSCAPE_NARROW_FIGSIZE, LANDSCAPE_FIGSIZE
from analysis.RDD import RDD, KINK, TITLES, NAMES
from analysis.aggregate import MISSING_MONTHS

parser = argparse.ArgumentParser()
parser.add_argument('--rdd', help='Folder containing fitted RDDs', required=True)
parser.add_argument('--img', help='Folder to write images to.', required=True)
parser.add_argument('--SI', help='SI material', required=False, type=str)

FONTSIZE = 14
CORE_FEATURES = ['liwc_Negemo', 'liwc_Anger', 'liwc_Anx', 'liwc_Sad', 'liwc_Swear']
SI_FEATURES = ['liwc_Negemo', 'empath_negative_emotion', 'liwc_Anger', 'liwc_Anx', 'liwc_Sad', 'liwc_Swear', 'empath_swearing_terms']
LIWC_FEATURES = CORE_FEATURES + ['liwc_Posemo']
DROPPED_DATA = [datetime(int(m.split('-')[0]), int(m.split('-')[1]), 15) for m in MISSING_MONTHS]

# Colors, after https://mikemol.github.io/technique/colorblind/2018/02/11/color-safe-palette.html
VERMILLION = '#D55E00'
ORANGE = '#E69F00'
SKYBLUE = '#56B4E9'
REDDISH = '#CC79A7'
BLUE = '#0072B2'
BLUEGREEN = '#009E73'


def _default_style(): return {'color': 'tab:grey', 'linewidth': 2.5, 'scatter_color': 'tab:grey'}


STYLES = {
    'liwc_Negemo': {'color': VERMILLION, 'linewidth': 2.5, 'scatter_color': VERMILLION},
    'empath_negative_emotion': {'color': VERMILLION, 'linewidth': 2.5, 'scatter_color': VERMILLION},
    'liwc_Anger': {'color': ORANGE, 'linewidth': 2.5, 'scatter_color': ORANGE},
    'liwc_Anx': {'color': SKYBLUE, 'linewidth': 2.5, 'scatter_color': SKYBLUE},
    'liwc_Sad': {'color': REDDISH, 'linewidth': 2.5, 'scatter_color': REDDISH},
    'liwc_Swear': {'color': BLUE, 'linewidth': 2.5, 'scatter_color': BLUE},
    'empath_swearing_terms': {'color': BLUE, 'linewidth': 2.5, 'scatter_color': BLUE},
    'liwc_Posemo': {'color': BLUEGREEN, 'linewidth': 2.5, 'scatter_color': BLUEGREEN},
    'linreg': {'color': 'black', 'linewidth': 2.5, 'linestyle': '-.'}
}

PARTY_STYLES = {
    'democrats': {'color': 'tab:blue', 'linewidth': 2.5, 'scatter_color': 'tab:blue', 'label': 'Democrats'},
    'republicans': {'color': 'tab:red', 'linewidth': 2.5, 'scatter_color': 'tab:red', 'label': 'Republicans'}
}

POLITICIAN_IDS = {  # Make Individual Plots for these
    'Q76': 'Barack Obama',
    'Q6294': 'Hillary Clinton',
    'Q22686': 'Donald Trump',
    'Q6279': 'Joe Biden',
    'Q24313': 'Mike Pence',
    'Q4496': 'Mitt Romney',
}
# 'Q10390': 'John McCain'

PROMINENCE_IDS = {  # Annotate them in score-vs-verbosity plots
    'Q76': 'Barack Obama',
    'Q22686': 'Donald Trump',
    'Q4496': 'Mitt Romney',
    'Q6294': 'Hillary Clinton',
    'Q207': 'George W. Bush',
    'Q10390': 'John McCain',
    'Q6279': 'Joe Biden',
}
# 'Q170581': 'Nancy Pelosi',

PROMINENCE_SHORT = {
    'Barack Obama': 'O',
    'Donald Trump': 'T',
    'Mitt Romney': 'R',
    'Hillary Clinton': 'C',
    'George W. Bush': 'Bu',
    'John McCain': 'M',
    'Joe Biden': 'Bi',
}
#     'Nancy Pelosi': 'P',
#     'Bernie Sanders': 'S'

V_label = {
    0: 'Most prominent speaker quartile',
    1: '2nd most prominent speaker quartile',
    2: '3rd most prominent speaker quartile',
    3: 'Least prominent speaker quartile'
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


def _grid_annotate(ax: plt.axis, model: RDD, feature: str, **kwargs):
    txt = r"{\mathrm{adj}}"
    r2 = f'$R^2_{txt}$={model.rdd[feature].loc["r2_adj"]:.2f}'
    params = '\n'.join([
        ',  '.join([
            r'$\alpha_0$=' +
            model.get_table(asPandas=True)[r'$\alpha_0$'].loc[feature].split('(')[0],
            r'$\beta_0$=' +
            model.get_table(asPandas=True)[r'$\beta_0$'].loc[feature].split('(')[0]
        ]),
        ',  '.join([
            r'$\alpha$=' + model.get_table(asPandas=True)[r'$\alpha$'].loc[feature].split('(')[
                0],
            r'$\beta$=' + model.get_table(asPandas=True)[r'$\beta$'].loc[feature].split('(')[0]
        ])
    ])
    # Aligned version in comments:
    # params = '\\\\'.join([
    #     ', '.join([
    #         r'\alpha_0=' +
    #         model.get_table(asPandas=True)[r'$\alpha_0$'].loc[feature].split('(')[0],
    #         r'&\beta_0=' +
    #         model.get_table(asPandas=True)[r'$\beta_0$'].loc[feature].split('(')[0]
    #     ]),
    #     ', '.join([
    #         r'\alpha=' + model.get_table(asPandas=True)[r'$\alpha$'].loc[feature].split('(')[
    #             0],
    #         r'&\beta=' + model.get_table(asPandas=True)[r'$\beta$'].loc[feature].split('(')[0]
    #     ])
    # ])
    # params = r'\begin{align*}' + re.sub(r'(\*+)', r'\\text{\1}', params) + r'\end{align*}'
    # rc('text', usetex=True)
    # rc('text.latex', preamble=r'\usepackage{amsmath}')
    default_box_props = dict(boxstyle='round', facecolor='white', alpha=.75, ec='none')
    box_props = kwargs.get('box_props', default_box_props)
    ax.text(0.03, 0.05, r2, transform=ax.transAxes, fontsize=FONTSIZE-2, verticalalignment='bottom',
            horizontalalignment='left', bbox=box_props)
    ax.text(0.5, kwargs.get('param_y', .97), params, transform=ax.transAxes, fontsize=FONTSIZE-2, verticalalignment='top',
            horizontalalignment='center', bbox=box_props)
    # rc('text', usetex=False)


def grid(models: Dict[str, RDD], ncols: int, nrows: int, features: List[str], style: Dict, gridspec: bool = False,
         **kwargs):
    fontsize = kwargs.get('fontsize', FONTSIZE)
    fontweight = kwargs.get('fontweight', 'bold')
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
            ROW = i % nrows
            COL = i // nrows
            if isinstance(axs[0], np.ndarray):
                ax = axs[ROW][COL]
            else:
                ax = axs[i]
            ax.set_title(names[feature], fontsize=fontsize, fontweight=fontweight)
            ax.set_title('(' + kwargs.get('prefix', string.ascii_lowercase)[i] + ')', fontfamily='serif', loc='left',
                         fontsize=FONTSIZE+2, fontweight='bold')  # Subplot naming
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

            if COL > 0:
                ax.tick_params(axis='y', which='both', left=False, right=False)
            if kwargs.get('grid_annotate', False):
                if kwargs.get('y_param'):  # plug any number of other features in here if necessary
                    _grid_annotate(ax, model, feature, y_param=kwargs.get('y_param'))
                else:
                    _grid_annotate(ax, model, feature)
            if kwargs.get('mean_adapt', False):
                _, Y = model._get_rdd_plot_data(feature)
                ymin = min(ymin, min(Y))
                ymax = max(ymax, max(Y))
            else:
                ymin = min(ymin, min(model.data[feature]))
                ymax = max(ymax, max(model.data[feature]))
            if kwargs.get('ylabel', False) and (COL == 0):
                ax.set_ylabel(kwargs.get('ylabel'), fontsize=FONTSIZE)
            if kwargs.get('right_ylabel', False) and (COL == ncols - 1):
                ax.set_ylabel(kwargs.get('right_ylabel'), fontsize=FONTSIZE)
                ax.yaxis.set_label_position("right")

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

    plt.minorticks_off()
    return fig, axs


def scatter_grid(nrows, ncols, model, features, ylims=None, figsize=TWO_COL_FIGSIZE, axs=None, **kwargs):
    if axs is None:
        fig, axs = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows, sharex='all', sharey='all', squeeze=False)
    else:
        fig = plt.gcf()
        if nrows == 1:  # Repack
            axs = [axs]
        if ncols == 1:
            axs = [axs]
    fontsize = kwargs.get('fontsize', FONTSIZE)

    ymin, ymax = np.inf, -np.inf
    for i, feature in enumerate(features):
        ROW = i // ncols
        COL = i % ncols
        ax = axs[ROW][COL]
        _scatter_only(ax, feature, model, STYLES[feature]['color'])
        ax.set_title(NAMES[feature], fontweight='bold', fontsize=fontsize)
        ymin = min(ymin, min(model.data[feature]))
        ymax = max(ymax, max(model.data[feature]))
        if COL == 0:
            ax.set_ylabel('Pre-campaign z-scores', fontsize=fontsize)
        if ncols * nrows != 1:
            ax.set_title('(' + kwargs.get('prefix', string.ascii_lowercase)[i]+ ')', fontfamily='serif', loc='left', fontsize=fontsize + 2, fontweight='bold')

    ydiff = ymax - ymin
    for row in axs:
        for ax in row:
            ax.tick_params(axis='both', which='major', labelsize=fontsize)
            ax.set_ylim(ymin - 0.25 * ydiff, ymax + 0.25 * ydiff)
            if ylims is not None:
                ax.set_ylim(ylims[0], ylims[1])

    fig.autofmt_xdate(rotation=90, ha='left')
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

    # 5x grid
    fig, axs = grid({'agg': model}, 5, 1, CORE_FEATURES, STYLES, grid_annotate=True,
                    ylabel='Pre-campaign z-scores')
    saveFigure(fig, base.joinpath('negative_and_swear_grid'))

    # 2 x 3 Grid only scatter
    fig, axs = scatter_grid(2, 3, model, LIWC_FEATURES, ylims)
    saveFigure(fig, base.joinpath('2x3_scatter_grid'))
    plt.close()

    # Negative Grid only scatter
    fig, axs = scatter_grid(1, 5, model, CORE_FEATURES, ylims, figsize=NARROW_NARROW_TWO_COL_FIGSIZE)
    saveFigure(fig, base.joinpath('negative_scatter_grid'))
    plt.close()

    # Negative Grid only posemo
    fig, axs = scatter_grid(1, 1, model, ['liwc_Posemo'], ylims)
    saveFigure(fig, base.joinpath('posemo_scatter'))
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

    fig, axs = grid(models, 5, 1, CORE_FEATURES,
                    {'With Outliers': outlierStyle, 'Without Outliers': baseStyle}, mean_adapt=True, legend=True,
                    ylabel='Pre-campaign z-scores')

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


def verbosity_vs_parameter(folder: Path, base: Path, kind: str, alpha_CI: float = 0.05, **kwargs):
    """
    Plots RDD parameters (y-Axis) vs speaker verbosity (x-axis)
    Parameters
    ----------
    folder: RDD models folder
    base: Img storage folder
    kind: Either "individual" or "ablation" - changes style adjustments
    alpha_CI: Confidence Interval parameter
    """
    verbosity = folder.parent.parent.joinpath('speaker_counts.csv')
    assert verbosity.exists(), "To create the scatter plot influence / verbosity, there needs to be a speaker count file."
    base_model_path = folder.parent.joinpath('QuotationAggregation_RDD.pickle')
    assert base_model_path.exists(), "To create the scatter plot influence / verbosity, there needs to be a Quotation Aggregation file."
    base_model = pickle.load(base_model_path.open('rb'))

    def _get_qid(s: str) -> str:
        if 'outlier' in s:
            print('WARNING: Are outlier data supposed to be analysed for individuals?')
        return re.search('Q[0-9]+', s)[0]

    class TextHandlerB(HandlerBase):
        """
        Allows adding text as legend Handle.
        Source: https://stackoverflow.com/questions/27174425/how-to-add-a-string-as-the-artist-in-matplotlib-legend
        """

        def create_artists(self, legend, text, xdescent, ydescent, width, height, fontsize, trans):
            tx = mText(width / 2., height / 2, text, fontsize=fontsize, ha="center", va="center", fontweight="bold")
            return [tx]

    # Allow text annotations for the legend
    Legend.update_default_handler_map({str: TextHandlerB()})
    names = list(set([_get_qid(speaker.name) for speaker in folder.iterdir() if 'outlier' not in speaker.name]))

    verbosity_df = pd.read_csv(verbosity)
    verbosity_df = verbosity_df[verbosity_df.QID.isin(names)]
    plot_data = {
        feature: pd.DataFrame(columns=names,
                              index=['alpha', 'beta', 'verbosity', 'alpha_low', 'alpha_high', 'beta_low',
                                     'beta_high', 'p_alpha', 'p_beta'])
        for feature in CORE_FEATURES
    }

    for speaker in folder.iterdir():
        if not (speaker.name.endswith('.pickle')):
            continue
        qid = _get_qid(speaker.name)
        speaker_data = pickle.load(speaker.open('rb'))
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
            p_alpha = speaker_data.rdd_fit[feature].pvalues['C(threshold)[T.1]']
            p_beta = speaker_data.rdd_fit[feature].pvalues['C(threshold)[T.1]:time_delta']
            plot_data[feature].at['p_alpha', qid] = p_alpha
            plot_data[feature].at['p_beta', qid] = p_beta
            plot_data[feature].at['alpha', qid] = alpha
            plot_data[feature].at['alpha_low', qid] = alpha_low
            plot_data[feature].at['alpha_high', qid] = alpha_high
            plot_data[feature].at['beta', qid] = beta
            plot_data[feature].at['beta_low', qid] = beta_low
            plot_data[feature].at['beta_high', qid] = beta_high
            plot_data[feature].at['verbosity', qid] = verbosity_df[verbosity_df.QID == qid]['Unique Quotations'].values[0]

    for param in kwargs.get('params', ('alpha', 'beta')):
        if kwargs.get('axs', None) is None:
            if kind == 'individual':
                fig, all_axs = plt.subplots(figsize=[17.8, 9], ncols=5, nrows=2, sharey='row')
                axs = all_axs[0, :]
                axs_cum = all_axs[1, :]
            else:
                fig, axs = plt.subplots(figsize=NARROW_TWO_COL_FIGSIZE, ncols=5, nrows=1, sharex='all', sharey='all')
        else:
            assert kwargs.get('params', None) is not None
            axs = kwargs.get('axs')
            fig = plt.gcf()

        significant_count = 0
        significant_positive = 0
        cumulative_annotations = []
        cumulative_values = dict()
        for i, (feature, data) in enumerate(plot_data.items()):
            verbosity2sign = []
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
            for qid in data.columns:
                isTrump = int(qid == 'Q22686')
                CI_low, CI_high = data[qid].loc[param + '_low'], data[qid].loc[param + '_high']
                param_pval = data[qid].loc[f'p_{param}']
                # Highlight "significant" points where CIs share a sign
                clr = STYLES[feature]['color']
                if kind == 'individual':
                    color = clr if (CI_low * CI_high > 0) else 'grey'
                    significant_count += int(param_pval < 0.05)
                    significant_positive += int((param_pval < 0.05) and (data[qid].loc[param] > 0))
                else:
                    # color = clr if (base_low > CI_high) else 'grey'
                    color = clr if isTrump else 'grey'
                ax.plot((data[qid].loc['verbosity'], data[qid].loc['verbosity']), (CI_low, CI_high), '-', color=color,
                        linewidth=0.3 + 2 * isTrump)
                ax.scatter(x=data[qid].loc['verbosity'], y=data[qid].loc[param], c=color, s=7.5 * (1 + 3 * isTrump))
                verbosity2sign.append([data[qid].loc['verbosity'], data[qid].loc[param] > 0])
                if qid in PROMINENCE_IDS:
                    x_annot = int(data[qid].loc['verbosity'])
                    y_annot = CI_low
                    cumulative_annotations.append((qid, x_annot))
                    offset = 0.08 if param == 'alpha' else 0.002
                    if kind == 'individual':
                        offset *= 3
                    va = 'top'
                    if (param == 'alpha') and (isTrump and (feature in ['liwc_Anger', 'liwc_Anx']) or feature == 'liwc_Sad'):
                        y_annot = CI_high + 3 * offset  # Annotation would leave the axis else
                        va = 'bottom'
                    ax.annotate(PROMINENCE_SHORT[PROMINENCE_IDS[qid]], (x_annot, y_annot - offset), ha='center',
                                va=va, label=PROMINENCE_IDS[qid], fontweight='bold', fontsize=FONTSIZE - 4)

            if kind == 'individual':
                annot = rf'$\{param}>0: {(data.loc[param] > 0).sum()}/{len(data.columns)}$' + '\n' + \
                        rf'(${significant_positive}/{significant_count}$' + rf' with $p<0.05$)'
                box_props = dict(boxstyle='round', facecolor=None, alpha=0, ec='none')
                # Commented out, info is now contained in the cumulative plot
                # ax.text(0.97, 0.05, annot, transform=ax.transAxes, fontsize=FONTSIZE - 2, multialignment='center',
                #         verticalalignment='bottom', horizontalalignment='right', bbox=box_props)

            ax.set_title(NAMES[feature], fontsize=FONTSIZE, fontweight='bold')
            ax.tick_params(labelbottom=True)
            ax.set_title('(' + kwargs.get('prefix', string.ascii_lowercase)[i] + ')', fontfamily='serif', loc='left',
                         fontsize=FONTSIZE+2, fontweight='bold')
            if i > 0:
                ax.tick_params(axis='y', which='both', left=False, right=False)
            else:
                labelText = 'without individuals' if kind == 'ablation' else 'for individuals only'
                ax.set_ylabel(r'$\{}$ {}'.format(param, labelText), fontsize=FONTSIZE-2)
                if (param == 'alpha') and (kind == 'individual'):
                    ax.set_ylim(-20, 20)
                    ax.set_yticks([-10, 10])
                    ax.set_yticklabels(['-10', '10'])
                elif (param == 'alpha') and (kind == 'ablation'):
                    ax.set_ylim([-.5, 3])
                elif (param == 'beta') and (kind == 'ablation'):
                    ax.set_ylim([-0.01, 0.05])
            ax.set_xlabel('Number of quotes', fontsize=FONTSIZE-2)

            # Now make the cumulative plot
            if kind == 'individual':
                ax = axs_cum[i]
                ax.set_xscale('log')
                ax.set_xlabel('Number $n$ of quotes', fontsize=FONTSIZE-2)
                ax.set_title(NAMES[feature], fontsize=FONTSIZE, fontweight='bold')
                ax.set_title('(' + kwargs.get('prefix', string.ascii_lowercase)[5+i] + ')', fontfamily='serif',
                             loc='left', fontsize=FONTSIZE + 2, fontweight='bold')
                x_axis, is_positive = zip(*sorted(verbosity2sign, reverse=True))  # Verbose to "silent"
                y_axis = [sum(is_positive[:i]) / i for i in range(1, 1+len(x_axis))]  # Running mean from right to left
                ax.axhline(.5, linewidth=.5, linestyle='dashed', color='grey')
                ax.plot(x_axis, y_axis, color=STYLES[feature]['color'], linewidth=STYLES[feature]['linewidth'])
                cumulative_values[feature] = [x_axis, y_axis]
                if i > 0:
                    ax.tick_params(axis='y', which='both', left=False, right=False)
                else:
                    ax.set_ylabel('Fraction of speakers with $\\{} > 0$\namong those with $\geq n$ quotes'.format(param), fontsize=FONTSIZE-2)

                # Annotate famous speakers
                for qid, x_val in cumulative_annotations:
                    y_val = y_axis[x_axis.index(x_val)]

                    ax.annotate(PROMINENCE_SHORT[PROMINENCE_IDS[qid]], (x_val, y_val + 0.03), ha='center', va='bottom',
                                label=PROMINENCE_IDS[qid], fontweight='bold', fontsize=FONTSIZE-4)
                    ax.scatter(x_val, y_val, color=STYLES[feature]['color'])
                    if param == 'alpha':
                        ax.set_ylim(0.3, 1.1)

        labels = list(PROMINENCE_SHORT.keys())
        handles = [PROMINENCE_SHORT[k] for k in labels]
        y_offset = -.15 if kind == 'ablation' else -.1
        fig.legend(fontsize=FONTSIZE, ncol=4, loc='lower center', bbox_to_anchor=(.5, y_offset),
                   labels=labels, handles=handles)
        plt.tight_layout()

        if kind == 'individual':
            pickle.dump(cumulative_values, base.joinpath('cumulative_{}.pickle'.format(param)).open('wb'))

        if kwargs.get('axs', None) is None:
            saveFigure(fig, base.joinpath('verbosity_vs_{}'.format(param)))


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
    verbosity_vs_parameter(folder, base, 'ablation')
    # fig, axs = plt.subplots(nrows=2, ncols=5, figsize=TWO_COL_FIGSIZE, sharex='all', sharey='row')
    # verbosity_vs_parameter(folder, base, 'ablation', alpha_CI=0.17, params=['alpha'], axs=axs[0])
    # verbosity_vs_parameter(folder.parent.joinpath('Individuals'), base.parent.joinpath('Individuals'),
    #                        'individual', alpha_CI=0.17, params=['alpha'], axs=axs[1], prefix='fghijklmno')
    # saveFigure(fig, base.parent.joinpath('combinedIndividual').joinpath('alpha.pdf'))


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
        if 'without' in path.name:
            return path.name
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

        hrFeature = feature  # Human readable, as used in the tables
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
    fig, axs = grid(models, 5, 1, CORE_FEATURES, PARTY_STYLES, legend=True)
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
    features = SI_FEATURES

    SI_fig, SI_axs = plt.subplots(ncols=7, nrows=4, sharex='all', sharey='all', figsize=LANDSCAPE_FIGSIZE)
    SI_fig.subplots_adjust(wspace=.04, hspace=1, bottom=.3)

    lower, upper = (13926.0, 18578.0)  # Hard coded numeric Quotebank Date limits + margin
    for feature in features:
        fig, axs = plt.subplots(figsize=NARROW_TWO_COL_FIGSIZE, ncols=4, sharex='all', sharey='all')
        y_min = np.Inf
        y_max = - np.Inf

        for verbosity, model in models.items():
            ax = axs[verbosity]
            model.plot(feature, ax=ax, parameters=False, visuals=False, color=STYLES[feature]['color'], lin_reg=False,
                       annSize=FONTSIZE - 2, scatter_color=STYLES[feature]['color'])
            if feature in SI_FEATURES:
                model.plot(feature, ax=SI_axs[verbosity][SI_FEATURES.index(feature)], parameters=False, visuals=False,
                           color=STYLES[feature]['color'], lin_reg=False, scatter_color=STYLES[feature]['scatter_color'],
                           linewidth=2)
                _grid_annotate(SI_axs[verbosity][SI_FEATURES.index(feature)], model, feature, param_y=.95)
            # Adapt y-limits of the plot to mean
            _, Y = model._get_rdd_plot_data(feature)
            y_min = min(y_min, min(Y))
            y_max = max(y_max, max(Y))
            ax.set_title(NAMES[feature] + '\n' + V_label[verbosity], fontsize=FONTSIZE - 2, fontweight='bold')
            ax.set_title('(' + string.ascii_lowercase[verbosity] + ')\n', fontfamily='serif', loc='left',
                         fontsize=FONTSIZE+2, fontweight='bold')

            if verbosity == 0:
                ax.set_ylabel('Pre-campaign z-scores', fontsize=FONTSIZE)
            _grid_annotate(ax, model, feature)

        y_diff = y_max - y_min
        for ax in axs:
            ax.set_xlim(lower, upper)
            ax.set_ylim(y_min - y_diff, y_max + y_diff)

        fig.autofmt_xdate(rotation=90, ha='left')
        plt.minorticks_off()

        saveFigure(fig, base.joinpath(feature))

    for i, row in enumerate(SI_axs):
        for j, ax in enumerate(row):
            ax.set_ylim([-7, 7.5])
            ax.set_title('(' + ('I', 'II', 'III', 'IV')[i] + '.' + string.ascii_lowercase[j] + ')', fontfamily='serif',
                         loc='left', fontsize=FONTSIZE-2, fontweight='bold')
            ax.set_title(NAMES[SI_FEATURES[j]], fontsize=FONTSIZE-2, fontweight='bold')
            if j != 0:
                ax.tick_params(axis='y', which='both', left=False, right=False)
            if j == 3:
                ax.text(0.5, 1.33, V_label[i], fontsize=FONTSIZE, fontweight='bold', va='center', ha='center',
                        transform=ax.transAxes)
            ax.tick_params(labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=90, ha="left")

    plt.minorticks_off()

    saveFigure(SI_fig, base.joinpath('Verbosity_all'), excludeTightLayout=True)
    plt.close('all')


def attribute_plots(model_path: Path, base: Path):
    attributes = ['Intercept', 'C(threshold)[T.1]', 'time_delta', 'C(threshold)[T.1]:time_delta',
                  'party', 'governing_party', 'congress_member', 'gender']
    annotate = {
        'gender': ['Male', 'Female'],
        'party': ['Republicans', 'Democrats'],
        'congress_member': ['Others', 'Congress'],
        'governing_party': ['Opposition', 'Government']
    }
    model = pickle.load(model_path.open('rb'))

    styles_cpy = {NAMES[key]: val for key, val in STYLES.items()}

    ORDER = ['liwc_Negemo', 'liwc_Anger', 'liwc_Anx', 'liwc_Sad', 'liwc_Swear']
    titles = deepcopy(TITLES)
    for key, val in titles.items():
        titles[key] = re.sub(r'\$(.*)\$', r'$\\mathbf{\1}$', titles[key])

    fig, axs = plt.subplots(figsize=TWO_COL_FIGSIZE, ncols=4, nrows=2)
    for i, att in enumerate(attributes):
        ROW = i // 4
        COL = i % 4
        ax = axs[ROW][COL]
        df = pd.DataFrame(data=None, index=ORDER, columns=['mean', 'low', 'high'])

        for feat in ORDER:
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

        if 'time_delta' not in att:  # All but betas
            ax.set_xlim([-4.5, 4.5])
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_xlabel('Pre-campaign z-scores', fontsize=FONTSIZE - 2)
        else:
            ax.set_xlim([-0.075, 0.075])
            ax.set_xticks([-0.05, 0, 0.05])
            ax.set_xlabel('Pre-campaign z-scores per month', fontsize=FONTSIZE - 2)

        if att in annotate:
            left, right = annotate[att]
            ax.text(x=0.05, y=0.9, s=r'{}$\longleftarrow$'.format(left), fontsize=FONTSIZE - 2, ha='left',
                    va='bottom', transform=ax.transAxes)
            ax.text(x=0.95, y=0.9, s=r'$\longrightarrow${}'.format(right), fontsize=FONTSIZE - 2, ha='right',
                    va='bottom', transform=ax.transAxes)

        ax.tick_params(axis='both', labelsize=FONTSIZE - 2)
        ax.set_title(titles[att], fontsize=FONTSIZE - 2, fontweight='bold')
        ax.set_title('(' + string.ascii_lowercase[i] + ')', fontfamily='serif', loc='left', fontsize=FONTSIZE+2, fontweight='bold')
        ax.axvline(x=0, linestyle='dashed', color='black', linewidth=0.5)
        lower, upper = ax.get_ylim()
        ax.set_ylim(lower - .5, upper + .5)

        if COL in (1, 2):
            ax.set_yticklabels([])
            ax.tick_params(axis='y', which='both', left=False, right=False)

        if COL == 3:
            ax.yaxis.tick_right()

    plt.tight_layout()
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
    print('Monthly quotes and speakers')
    df = pd.read_csv(data_path)
    df['dt_month'] = df.apply(lambda r: datetime(int(r.year), int(r.month), 15), axis=1)
    fig, axs = plt.subplots(figsize=ONE_COL_FIGSIZE, nrows=2)
    for i, col, name in ((0, 'num_speaker', 'Monthly Speakers'), (1, 'num_quotes', 'Unique Monthly Quotations')):
        fig, ax = timeLinePlot(x=df.dt_month,
                               y=df[col],
                               kind='scatter',
                               snsargs={'color': 'black'},
                               ax=axs[i])
        drop_mask = df.dt_month.isin(DROPPED_DATA)
        fig, ax = timeLinePlot(x=df.dt_month[drop_mask],
                               y=df[col][drop_mask],
                               kind='scatter',
                               snsargs={'color': 'grey'},
                               ax=axs[i])
        ax.set_title(name, fontsize=FONTSIZE)
        ax.set_xlim(13926.0, 18578.0)

    axs[0].set_ylim([0, 11000])
    axs[0].set_yticks([5000, 10000])
    axs[1].set_ylim([0, 300000])
    axs[1].set_yticks([100000, 200000])

    saveFigure(fig, base.joinpath('quantities.pdf'))
    plt.close()


def RDD_kink_performance(data_path: Path, base: Path):
    """
    Plots the performance of the RDD, depending on the date of the discontinuity / kink
    """
    data = pickle.load(data_path.open('rb'))
    if 'YouGov' in str(data_path):
        base = base.joinpath('YouGov')
        base.mkdir(exist_ok=True)
    dates = list(data.keys())
    ylabeltext = r"$R^2_{\mathrm{adj}}$"
    fig, ax = plt.subplots(figsize=TWO_COL_FIGSIZE)
    rdd_style = deepcopy(STYLES)
    for f in rdd_style:
        try:
            del rdd_style[f]['scatter_color']
        except KeyError:
            pass

    for feature in CORE_FEATURES:
        timeLinePlot(
            x=dates,
            y=[data[dt][feature]['r2_adjust'] for dt in dates],
            snsargs=dict(label=NAMES[feature], **rdd_style[feature]),
            ax=ax,
            includeElections=False
        )
    ax.axvline(mdates.date2num(KINK), color='black', linestyle='--')
    plt.legend(fontsize=FONTSIZE, loc='upper right', framealpha=1, fancybox=False, ncol=3)
    ax.set_ylabel(ylabeltext, fontsize=FONTSIZE)
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
    ax.set_ylabel(ylabeltext, fontsize=FONTSIZE)
    saveFigure(fig, base.joinpath('r2_adj_posemo'))

    fig, ax = plt.subplots(figsize=TWO_COL_FIGSIZE)
    for feature in ['empath_negative_emotion', 'empath_swearing_terms']:
        timeLinePlot(
            x=dates,
            y=[data[dt][feature]['r2_adjust'] for dt in dates],
            snsargs=dict(label=re.sub('\n', ' ', NAMES[feature]), **rdd_style[feature]),
            ax=ax,
            includeElections=False
        )
    ax.axvline(mdates.date2num(KINK), color='black', linestyle='--')
    plt.legend(fontsize=FONTSIZE, loc='upper right', framealpha=1, fancybox=False, ncol=1)
    ax.set_ylabel(ylabeltext, fontsize=FONTSIZE)
    saveFigure(fig, base.joinpath('r2_empath'))

    plt.close('all')


def aggregation_overview(QuotationAggregation_RDD: RDD, SpeakerAggregation_RDD: RDD, democrats: RDD, republicans: RDD,
                         folder: Path):
    """
    Combines the plots for the three key aggregations (Quotations, Speaker, Party) in one figure.
    Parameters
    ----------
    QuotationAggregation_RDD: RDD model of the quotation aggregation
    SpeakerAggregation_RDD: RDD model of the speaker aggregation
    democrats: RDD model of the quotation aggregation for democrats
    republicans: RDD model of the quotation aggregation for republicans
    folder: Where to store the plot
    """
    print('Large Aggregation Grid')
    democratStyle = deepcopy(STYLES)
    republicanStyle = deepcopy(STYLES)
    for key in republicanStyle:
        republicanStyle[key]['color'] = 'grey'
        republicanStyle[key]['scatter_color'] = 'grey'
        republicanStyle[key]['label'] = 'Republicans'
        democratStyle[key]['label'] = 'Democrats'
    partyStyle = {'democrats': democratStyle, 'republicans': republicanStyle}

    fig, axs = plt.subplots(ncols=5, nrows=3, figsize=[NARROW_TWO_COL_FIGSIZE[0], 3 * NARROW_TWO_COL_FIGSIZE[1]], sharey='all')
    plt.subplots_adjust(wspace=.04, hspace=.6, bottom=.3)

    quot = axs[0, :]
    speak = axs[1, :]
    party = axs[2, :]

    grid({'agg': QuotationAggregation_RDD}, 5, 1, CORE_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2,
         ylabel='Pre-campaign z-scores', axs=quot)
    grid({'agg': SpeakerAggregation_RDD}, 5, 1, CORE_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2,
         ylabel='Pre-campaign z-scores', axs=speak, prefix=string.ascii_lowercase[5:])

    grid({'democrats': democrats, 'republicans': republicans}, 5, 1, CORE_FEATURES, partyStyle, fontsize=FONTSIZE-2,
         grid_annotate=False, ylabel='Pre-campaign z-scores', axs=party, prefix=string.ascii_lowercase[10:],
         legend=True)

    txt = ['Quote-level aggregation', 'Speaker-level aggregation', 'Quote-level aggregation by party']
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.set_ylim([-7, 7])
            if j == 2:
                ax.text(0.5, 1.2, txt[i], fontsize=FONTSIZE, fontweight='bold', va='center', ha='center',
                        transform=ax.transAxes)
            ax.tick_params(labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=90, ha="left")

    saveFigure(fig, folder.joinpath('Negativity.pdf'), excludeTightLayout=True)

    # Positive
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=NARROW_TWO_COL_FIGSIZE, sharex='all', sharey='all')
    _scatter_only(axs[0], 'liwc_Posemo', QuotationAggregation_RDD, STYLES['liwc_Posemo']['color'], s=40)
    axs[0].set_title(NAMES['liwc_Posemo'] + '\n' + txt[0], fontweight='bold', fontsize=FONTSIZE)
    axs[0].set_ylabel('Pre-campaign z-scores', fontsize=FONTSIZE)
    axs[0].set_title('(a)', fontfamily='serif', loc='left', fontsize=FONTSIZE+2, fontweight='bold')

    _scatter_only(axs[1], 'liwc_Posemo', SpeakerAggregation_RDD, STYLES['liwc_Posemo']['color'], s=40)
    axs[1].set_title(NAMES['liwc_Posemo'] + '\n' + txt[1], fontweight='bold', fontsize=FONTSIZE)
    axs[1].set_title('(b)', fontfamily='serif', loc='left', fontsize=FONTSIZE+2, fontweight='bold')

    timeLinePlot(democrats.data.date, democrats.data['liwc_Posemo'], ax=axs[2],
                 snsargs={'s': 40, 'color': STYLES['liwc_Posemo']['color'], 'label': 'Democrats'},
                 kind='scatter')
    timeLinePlot(republicans.data.date, republicans.data['liwc_Posemo'], ax=axs[2],
                 snsargs={'s': 40, 'color': 'grey', 'label': 'Republicans'},
                 kind='scatter')
    axs[2].set_title(NAMES['liwc_Posemo'] + '\n' + txt[2], fontweight='bold', fontsize=FONTSIZE)
    axs[2].set_title('(c)', fontfamily='serif', loc='left', fontsize=FONTSIZE+2, fontweight='bold')
    axs[2].legend(loc='lower center', ncol=2)

    fig.autofmt_xdate(rotation=90, ha='left')
    plt.minorticks_off()

    saveFigure(fig, folder.joinpath('Positive.pdf'))


def title_figure(president_verbosity: pd.DataFrame, raw_negemo: pd.DataFrame, quoteAggregationRDD: RDD, folder: Path):
    """
    Plots the raw liwc score of the negative emotion liwc category vs. the relative share of quotes
    uttered by Donald Trump vs Barack Obama
    """
    # Custom Legend Patches (multicolor) for the background
    # From: https://stackoverflow.com/questions/31908982/python-matplotlib-multi-color-legend-entry
    class MulticolorPatch:
        def __init__(self, colors):
            self.colors = colors

    class MulticolorPatchHandler:
        def legend_artist(self, legend, orig_handle, fontsize, handlebox):
            width, height = handlebox.width, handlebox.height
            patches = []
            for i, c in enumerate(orig_handle.colors):
                patches.append(
                    plt.Rectangle(
                        (float(width / len(orig_handle.colors) * i - handlebox.xdescent), float(-handlebox.ydescent)),
                        width / len(orig_handle.colors),
                        height,
                        facecolor=c,
                        edgecolor='none'
                    )
                )
            patch = PatchCollection(patches, match_original=True)

            handlebox.add_artist(patch)
            return patch
    print('Title Figure')
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

    fig, (left, right) = plt.subplots(figsize=TWO_COL_FIGSIZE, ncols=2)
    # fig, left_axis = plt.subplots(figsize=TWO_COL_FIGSIZE)
    presidents = left.twinx()

    leftStyle = {'label': 'Negative emotion (all politicians)', 'color': 'black', 'linewidth': 3}
    fig, left = timeLinePlot(x=dates, y=raw_negemo.liwc_Negemo,
                             snsargs=leftStyle,
                             kind='line',
                             ax=left,
                             zorder=1,
                             includeElections=False)
    fig, left = timeLinePlot(x=dates, y=raw_negemo.liwc_Negemo,
                             snsargs={'color': 'black', 's': 50, 'edgecolor': 'black'},
                             kind='scatter',
                             ax=left,
                             zorder=1,
                             includeElections=False)

    left.set_ylabel('Negative emotion score', fontsize=FONTSIZE)
    left.axvline(x=mdates.date2num([datetime(2015, 6, 15)])[0], color='tab:red', linewidth=2.5)
    left.text(0.59, 0.05, "June 2015: Beginning of\nTrump's primary campaign", transform=left.transAxes,
              color='tab:red', fontsize=FONTSIZE, fontweight='bold')
    left.set_xlim(*mdates.date2num([min(dates)]), *mdates.date2num([max(dates)]))
    left.set_facecolor('none')

    presidents.set_ylim([0, 1])
    presidents.set_yticklabels([])
    presidents.tick_params(axis='y', which='both', right=False)
    presidents.set_facecolor('none')

    colorTrump = (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 0.4)
    colorObama = (0.05, 0.48, 1, 0.75)
    presidents.fill_between(mdates.date2num(dates), np.zeros(len(dates)),
                            np.asarray(shares.trump / shares.total, dtype=float), color=colorTrump,
                            step='mid', zorder=0)
    presidents.fill_between(mdates.date2num(dates), np.asarray(shares.trump / shares.total, dtype=float),
                            np.ones(len(dates)), color=colorObama, step='mid', zorder=0)

    for election in PRESIDENTIAL_ELECTIONS:
        left.axvline(x=election, linewidth=2, c='black', linestyle='dotted', zorder=1, alpha=.3)
    electionLines = Line2D([0], [0], linewidth=2, c='black', linestyle='dotted', alpha=.3)
    backgroundColors = MulticolorPatch([colorTrump, colorObama])
    handles, labels = left.get_legend_handles_labels()
    left.get_legend().remove()
    left.legend(handles=handles + [electionLines, backgroundColors],
                labels=labels + ['Presidential elections', 'Trump/Obama quote share'],
                loc='upper left',
                handler_map={MulticolorPatch: MulticolorPatchHandler()})

    # Right plot
    settings = dict(ci=False, lin_reg=False, annotate=False, visuals=False,
                    ylabel='Negative emotion (pre-campaign z-scores)', color=STYLES['liwc_Negemo']['color'],
                    includeElections=False, s=50)
    quoteAggregationRDD.plot('liwc_Negemo', right, **settings)

    # Slope and Intercepts
    t_thresh = mdates.date2num(KINK)
    alpha_0_val = quoteAggregationRDD.rdd['liwc_Negemo']['Intercept']
    alpha_val = quoteAggregationRDD.rdd['liwc_Negemo']['C(threshold)[T.1]']
    beta_0_val = quoteAggregationRDD.rdd['liwc_Negemo']['time_delta']
    beta_val = quoteAggregationRDD.rdd['liwc_Negemo']['C(threshold)[T.1]:time_delta']
    alpha_0 = dict(x=t_thresh + 300, y=0, dx=0, dy=alpha_0_val, color='green')
    alpha_0_styling = dict(text=r'intercept $\mathbf{\alpha_0}$', fraction_dy=.5, x=t_thresh + 330, align='left')
    alpha = dict(x=t_thresh, y=alpha_0_val, dx=0, dy=alpha_val, color=(1, .65, 0))
    alpha_styling = dict(text=r'campaign offset $\mathbf{\alpha}$', fraction_dy=.9, x=t_thresh - 60, align='right')

    def reverseDirection(d):
        ret = deepcopy(d)
        ret['x'] = d['x'] + d['dx']
        ret['dx'] = - d['dx']
        ret['y'] = d['y'] + d['dy']
        ret['dy'] = - d['dy']
        d['linewidth'] = 0
        return ret

    arrowStyle = dict(length_includes_head=True, linewidth=3, head_width=75, head_length=0.2, joinstyle='bevel', zorder=2)
    right.hlines([0, alpha_0['dy']], [t_thresh, t_thresh], [t_thresh + 300, t_thresh + 300], colors='black', linewidth=2)
    for style, param in zip([alpha_0_styling, alpha_styling], [alpha_0, alpha]):
        name = style['text']
        right.arrow(**param, **arrowStyle)
        # right.arrow(**reverseDirection(param), **arrowStyle)
        right.text(style['x'], param['y'] + style['fraction_dy'] * param['dy'], name, color=param['color'],
                   ha=style['align'], va='center', fontsize=FONTSIZE + 2, fontweight='bold')

    def x_at(delta: int) -> int:
        deltaFromZero = delta + quoteAggregationRDD.rdd['liwc_Negemo']['split_0']
        return mdates.date2num(quoteAggregationRDD._get_approx_date(deltaFromZero))

    def y_at(delta: int) -> float:
        if delta < 0:
            return alpha_0_val + beta_0_val * delta
        else:
            return alpha_0_val + alpha_val + (beta_val + beta_0_val) * delta

    patchColor = (0.1, 0.4, 1, 0.4)
    beta_0 = dict(x1=-65, x2=-45, text=r'slope $\mathbf{\beta_0}$', y_offset=0.1)
    beta = dict(x2=20, x1=55, text=r'slope $\mathbf{\beta_0 + \beta}$', y_offset=0.2)
    patches = []
    for param in [beta_0, beta]:
        x1 = x_at(param['x1'])
        y1 = y_at(param['x1'])
        x2 = x_at(param['x2'])
        y2 = y_at(param['x2'])
        edges = np.array([[x1, y1], [x2, y1], [x2, y2]])
        x_text = x1 + 0.5 * (x2 - x1)
        y_text = max(y1, y2) + param['y_offset']
        right.text(x_text, y_text, param['text'], color=patchColor[:3], ha='center', va='bottom', fontsize=FONTSIZE + 2, fontweight='bold')
        patches.append(Polygon(xy=edges, fill=True, zorder=2))

    col = PatchCollection(patches, facecolors=patchColor)
    right.add_collection(col)

    equation = r"$y_t = \mathbf{\alpha_0} + \mathbf{\beta_0} \,t + \mathbf{\alpha} \,i_{t} + \mathbf{\beta} \,i_{t} \,t + \varepsilon_{t}$"
    box_props = dict(boxstyle='round', facecolor='white', alpha=.5, ec='none')
    right.text(0.5, 0.85, equation, transform=right.transAxes, fontsize=FONTSIZE + 2, verticalalignment='bottom', horizontalalignment='center', bbox=box_props)

    old_ylim = right.get_ylim()
    right.scatter(mdates.date2num(KINK), old_ylim[0], s=100, marker='X', color='tab:red', clip_on=False, zorder=100)
    right.set_ylim(old_ylim)  # Prevent that scatter above moves the limit around
    right.text(mdates.date2num(KINK) + 50, right.get_ylim()[0] + 0.15, '$t = 0$ (June 2015)', fontsize=FONTSIZE, color='tab:red')

    right.set_xlim(*mdates.date2num([min(dates)]), *mdates.date2num([max(dates)]))
    right.axhline(y=0, linestyle='--', color='black', linewidth=0.8, zorder=0)
    plt.tight_layout()

    # saveFigure(fig, folder.joinpath('Fig0.pdf'))
    # left_extend = left.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.1, 1.1)
    # saveFigure(fig, folder.joinpath('Title_figure_left.pdf'), bbox_inches=left_extend)
    # right_extend = right.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).expanded(1.1, 1.2)
    # saveFigure(fig, folder.joinpath('Title_figure_right.pdf'), bbox_inches=right_extend)
    plt.close()


def SI_additionals(data: Path, SI: Path):
    """Assumes the default file structure is kept"""
    # Aggregations with and without Outliers
    print('Aggregations with and without outliers')
    fig, axs = plt.subplots(ncols=7, nrows=4, figsize=LANDSCAPE_FIGSIZE, sharey='all')
    plt.subplots_adjust(wspace=.04, hspace=.6, bottom=.3)
    QA = pickle.load(data.joinpath('QuotationAggregation_RDD.pickle').open('rb'))
    QA_out = pickle.load(data.joinpath('QuotationAggregation_RDD_outliers.pickle').open('rb'))
    SA = pickle.load(data.joinpath('SpeakerAggregation_RDD.pickle').open('rb'))
    SA_out = pickle.load(data.joinpath('SpeakerAggregation_RDD_outliers.pickle').open('rb'))

    grid({'': QA}, 7, 1, SI_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False,
         ylabel='Pre-campaign z-scores', axs=axs[0], prefix=string.ascii_uppercase)
    grid({'': QA_out}, 7, 1, SI_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False, ylabel='Pre-campaign z-scores', axs=axs[1],
         prefix=string.ascii_lowercase)
    grid({'': SA}, 7, 1, SI_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False, ylabel='Pre-campaign z-scores', axs=axs[2],
         prefix=string.ascii_uppercase[7:])
    grid({'': SA_out}, 7, 1, SI_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False, ylabel='Pre-campaign z-scores', axs=axs[3],
         prefix=string.ascii_lowercase[7:])

    txt = ['Quote-level aggregation', 'Quote-level aggregation, outliers removed', 'Speaker-level aggregation', 'Speaker-level aggregation, outliers removed']
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.set_ylim([-5, 6])
            if j == 3:
                ax.text(0.5, 1.2, txt[i], fontsize=FONTSIZE, fontweight='bold', va='center', ha='center',
                        transform=ax.transAxes)
            ax.tick_params(labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=90, ha="left")

    saveFigure(fig, SI.joinpath('ALL_AGG.pdf'), excludeTightLayout=True)

    # Scatters Only
    print('Scatter only including empath')
    fig, axs = plt.subplots(ncols=7, nrows=2, figsize=LANDSCAPE_NARROW_FIGSIZE, sharey='all')
    plt.subplots_adjust(wspace=.04, hspace=.6, bottom=.3)

    scatter_grid(1, 7, QA, SI_FEATURES, axs=axs[0], fontsize=FONTSIZE-2)
    scatter_grid(1, 7, SA, SI_FEATURES, axs=axs[1], prefix=string.ascii_lowercase[7:], fontsize=FONTSIZE-2)

    txt = ['Quote-level aggregation', 'Speaker-level aggregation']
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            # ax.set_ylim([-7, 7])
            if j == 3:
                ax.text(0.5, 1.2, txt[i], fontsize=FONTSIZE, fontweight='bold', va='center', ha='center',
                        transform=ax.transAxes)
            ax.tick_params(labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=90, ha="left")

    saveFigure(fig, SI.joinpath('Negativity_Scatter.pdf'), excludeTightLayout=True)

    # All Individuals
    print('Individuals Grid SI')
    fig, axs = plt.subplots(nrows=5, ncols=6, figsize=[LANDSCAPE_FIGSIZE[0], LANDSCAPE_FIGSIZE[1] * 1.1], sharey='all')
    name2qid = {val: key for key, val in POLITICIAN_IDS.items()}
    plt.subplots_adjust(wspace=.04, hspace=.55, bottom=.3)
    Obama = pickle.load(data.joinpath('Individuals').joinpath(name2qid['Barack Obama'] + '_RDD.pickle').open('rb'))
    Trump = pickle.load(data.joinpath('Individuals').joinpath(name2qid['Donald Trump'] + '_RDD.pickle').open('rb'))
    Biden = pickle.load(data.joinpath('Individuals').joinpath(name2qid['Joe Biden'] + '_RDD.pickle').open('rb'))
    Romney = pickle.load(data.joinpath('Individuals').joinpath(name2qid['Mitt Romney'] + '_RDD.pickle').open('rb'))
    Pence = pickle.load(data.joinpath('Individuals').joinpath(name2qid['Mike Pence'] + '_RDD.pickle').open('rb'))
    Clinton = pickle.load(data.joinpath('Individuals').joinpath(name2qid['Hillary Clinton'] + '_RDD.pickle').open('rb'))

    grid({'': Biden}, 1, 5, CORE_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False,
         ylabel='Pre-campaign z-scores', axs=axs[:, 0], prefix=string.ascii_lowercase, y_param=0.95)
    grid({'': Clinton}, 1, 5, CORE_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False,
         ylabel='Pre-campaign z-scores', axs=axs[:, 1], prefix=string.ascii_lowercase[5:], y_param=0.95)
    grid({'': Obama}, 1, 5, CORE_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False,
         ylabel='Pre-campaign z-scores', axs=axs[:, 2], prefix=string.ascii_lowercase[10:], y_param=0.95)
    grid({'': Pence}, 1, 5, CORE_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False,
         ylabel='Pre-campaign z-scores', axs=axs[:, 3], prefix=string.ascii_lowercase[15:], y_param=0.95)
    grid({'': Romney}, 1, 5, CORE_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False,
         ylabel='Pre-campaign z-scores', axs=axs[:, 4], prefix=string.ascii_lowercase[20:], y_param=0.95)
    grid({'': Trump}, 1, 5, CORE_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False,
         ylabel='Pre-campaign z-scores', axs=axs[:, 5], prefix=string.ascii_lowercase[25:] + string.ascii_uppercase, y_param=0.95)

    txt = ['Joe Biden', 'Hillary Clinton', 'Barack Obama', 'Mike Pence', 'Mitt Romney', 'Donald Trump']
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.set_ylim([-15, 35])
            if i == 0:
                ax.text(0.5, 1.24, txt[j], fontsize=FONTSIZE, fontweight='bold', va='center', ha='center',
                        transform=ax.transAxes)
            if j != 0:
                ax.tick_params(axis='y', which='both', left=False, right=False)
            ax.tick_params(labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=90, ha="left")

    saveFigure(fig, SI.joinpath('Individuals_Grid.pdf'), excludeTightLayout=True)
    plt.close('all')

    # Party Empath controls + Republicans Without Trump
    print('Party, but with empath')
    democrats = pickle.load(data.joinpath('parties').joinpath('PartyAggregation_RDD_democrats.pickle').open('rb'))
    republicans = pickle.load(data.joinpath('parties').joinpath('PartyAggregation_RDD_republicans.pickle').open('rb'))
    wot = pickle.load(data.joinpath('PartyAggregationWithoutTrump_RDD_republicans.pickle').open('rb'))

    fig, axs = plt.subplots(ncols=7, nrows=3, figsize=[LANDSCAPE_FIGSIZE[0], LANDSCAPE_FIGSIZE[1] * .5], sharey='all')
    plt.subplots_adjust(wspace=.04, hspace=.55, bottom=.3)
    grid({'': democrats}, 7, 1, SI_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False,
         ylabel='Pre-campaign z-scores', axs=axs[0], prefix=string.ascii_uppercase)
    grid({'': republicans}, 7, 1, SI_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False,
         ylabel='Pre-campaign z-scores', axs=axs[1], prefix=string.ascii_lowercase[7:])
    grid({'': wot}, 7, 1, SI_FEATURES, STYLES, grid_annotate=True, fontsize=FONTSIZE-2, lin_reg=False,
         ylabel='Pre-campaign z-scores', axs=axs[2], prefix=string.ascii_lowercase[14:])

    txt = ['Democrats', 'Republicans', 'Republicans without Donald Trump']
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            # ax.set_ylim([-5, 6])
            if j == 3:
                ax.text(0.5, 1.2, txt[i], fontsize=FONTSIZE, fontweight='bold', va='center', ha='center',
                        transform=ax.transAxes)
            ax.tick_params(labelbottom=True)
            plt.setp(ax.get_xticklabels(), rotation=90, ha="left")

    saveFigure(fig, SI.joinpath('Party_all'))


def main():
    args = parser.parse_args()
    data = Path(args.rdd)
    img = Path(args.img)

    # Map a file or folder name to a plotting utility.
    NAME_2_FUNCTION = {
        'verbosity': verbosity_plots,
        'parties': party_plots,
        'yougov_outlets': basic_model_plots,
        'Individuals': individuals,
        'Without': ablation_plots,
        'QuotationAggregation_RDD': basic_model_plots,
        'QuotationAggregation_RDD_outliers': outlier_plots,
        'SpeakerAggregation_RDD': basic_model_plots,
        'SpeakerAggregationSanity_RDD': basic_model_plots,
        'SpeakerAggregation_RDD_outliers': outlier_plots,
        'AttributesAggregation_RDD': attribute_plots,
        'AttributesAggregationSpeakerLevel_RDD': attribute_plots,
        'RDD_time_variation': RDD_kink_performance,
        'RDD_time_variation_YouGov': RDD_kink_performance,
        'PartyAggregationWithoutTrump': basic_model_plots,
        'PartyAggregationWithoutTrump_RDD_democrats': basic_model_plots,
        'PartyAggregationWithoutTrump_RDD_republicans': basic_model_plots
    }

    paths = {
        'QuotationAggregation_RDD': None,
        'SpeakerAggregation_RDD': None,
        'democrats': None,
        'republicans': None
    }
    for path in data.iterdir():
        if path.name.endswith('tex'):
            continue

        base_name = path.name.split('.')[0]
        if base_name in paths:
            paths[base_name] = path
        elif base_name == 'parties':
            paths['democrats'] = path.joinpath('PartyAggregation_RDD_democrats.pickle')
            paths['republicans'] = path.joinpath('PartyAggregation_RDD_republicans.pickle')
        if base_name not in NAME_2_FUNCTION:
            continue

        print(base_name)
        base_folder = img.joinpath(base_name)
        plot = NAME_2_FUNCTION[base_name]
        plot(path, base_folder)

    if args.SI is not None:
        print('SI plots')
        si_data = Path(args.SI)
        si_img = img.joinpath('SI')
        si_img.mkdir(exist_ok=True)
        quant = si_data.joinpath('quantitative_statisticts.csv')
        plot_quantities(quant, si_img)
        SI_additionals(data, si_img)

    if all(p is not None for p in paths.values()):
        models = {key: pickle.load(paths[key].open('rb')) for key in paths}
        aggregation_overview(**models, folder=img)

    # Build "Fig. 0"
    if 'QuotationAggregation_RDD' in paths:
        aggregates = data.parent.joinpath('aggregates')
        presidents = pd.read_csv(
            aggregates.joinpath('presidents.csv').open('r'))  # Contains Obamas and Trumps number of Quotes
        negemo = pd.read_csv(aggregates.joinpath('QuotationAggregation.csv').open('r'))[['liwc_Negemo', 'date']]
        negemo['liwc_Negemo'] = negemo['liwc_Negemo'] \
                                * pickle.load(aggregates.joinpath('std.pickle').open('rb'))['liwc_Negemo'] \
                                + pickle.load(aggregates.joinpath('mean.pickle').open('rb'))[
                                    'liwc_Negemo']  # Restore the original liwc scores
        title_figure(presidents, negemo, pickle.load(paths['QuotationAggregation_RDD'].open('rb')), img)


if __name__ == '__main__':
    main()
