import argparse
from copy import deepcopy
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from pathlib import Path
import re
import seaborn as sns
import sys
from typing import Dict, Tuple

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from utils.plots import saveFigure, timeLinePlot, ONE_COL_FIGSIZE, TWO_COL_FIGSIZE, NARROW_TWO_COL_FIGSIZE
from analysis.RDD import RDD, aicc

parser = argparse.ArgumentParser()
parser.add_argument('--rdd', help='Folder containing fitted RDDs', required=True)
parser.add_argument('--img', help='Folder to write images to.', required=True)

FONTSIZE = 14
CORE_FEATURES = ['liwc_Negemo', 'liwc_Posemo', 'liwc_Anger', 'liwc_Sad', 'liwc_Anx', 'liwc_Swear']
NAMES = {
    'liwc_Negemo': 'Negative Emotions',
    'liwc_Anx': 'Anxiety',
    'liwc_Anger': 'Anger',
    'liwc_Sad': 'Sadness',
    'liwc_Swear': 'Swearing Terms',
    'liwc_Posemo': 'Positive Emotions',
    'linreg': 'Linear Regression',
    'liwc_Certain': 'Certainty',
    'liwc_Tentat': 'Tentativeness',
    'empath_negative_emotion': 'Neg. Emotion (empath)',
    'empath_positive_emotion': 'Pos. Emotion (empath)',
    'empath_swearing_terms': 'Swearing (empath)'
}
STYLES = {
    'liwc_Negemo': {'color': 'tab:red', 'linewidth': 3},
    'liwc_Anx': {'color': 'darkorange', 'linewidth': 2.5},
    'liwc_Anger': {'color': 'darkorange', 'linewidth': 2.5},
    'liwc_Sad': {'color': 'darkorange', 'linewidth': 2.5},
    'liwc_Swear': {'color': 'black', 'linewidth': 2.5},
    'liwc_Posemo': {'color': 'tab:green', 'linewidth': 3},
    'linreg': {'color': 'black', 'linewidth': 1.5, 'linestyle': '-.'}
}
PARTY_STYLES = {
    'democrats': {'color': 'tab:blue', 'linewidth': 2.5, 'scatter_color': 'tab:blue', 'label': 'Deomcrats'},
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


def _scatter_only(ax: plt.axis, feature: str, model: RDD, color: str):
    timeLinePlot(model.data.date, model.data[feature], ax=ax, snsargs={'s': 25, 'color': color}, kind='scatter')


def _grid_annotate(ax: plt.axis, model: RDD, feature: str, title: str):
    ax.set_title(title, fontsize=FONTSIZE, fontweight='bold')

    aicc_rdd = aicc(model.rdd[feature].aic, 4, len(model.data))
    aicc_lin = aicc(model.lin_reg[feature].aic, 2, len(model.data))

    fit = ',  '.join([
        f'$R^2_{"{adj}"}$={model.rdd[feature].loc["r2_adj"]:.2f}',
        # f'$r^2_{"{lin, adj}"}$={model.lin_reg[feature].loc["r2_adjust"]:.2f}',
        # r'$\Delta_{AICc}$: ' + f'{aicc_lin - aicc_rdd:.2f}'
    ])
    params = ',  '.join([
        r'$\alpha$=' + model.get_table(asPandas=True)[r'$\alpha$'].loc[' '.join(feature.split('_')[1:])],
        r'$\beta$=' + model.get_table(asPandas=True)[r'$\beta$'].loc[' '.join(feature.split('_')[1:])]
    ])
    box_props = dict(boxstyle='round', facecolor='white', alpha=1)

    ax.text(0.025, 0.05, fit, transform=ax.transAxes, fontsize=FONTSIZE, verticalalignment='bottom',
            horizontalalignment='left', bbox=box_props)
    ax.text(0.975, 0.95, params, transform=ax.transAxes, fontsize=FONTSIZE, verticalalignment='top',
            horizontalalignment='right', bbox=box_props)


def basic_model_plots(model_file: Path, base: Path, mean_adapt: bool = False, ylims: Tuple[float, float] = None):
    """
    Creates a single plot for every feature the RDD was fitted on.
    Parameters
    ----------
    model_file: Path to an RDD Model
    base: Base folder to store results in
    mean_adapt: If true, will adapt the y limits of the plots to the mean of the data instead of extreme values
    ylims: Another way to control the y axis limits: They can be directly provided here.
    -------
    """
    model = pickle.load(model_file.open('rb'))
    features = [col for col in model.data.columns if ('empath' in col) or ('liwc' in col)]

    # Single-Feature Plots
    for f in sorted(features):
        fig, ax = model.plot(f, parameters=True)
        if ylims is not None:
            ax.set_ylim(ylims[0], ylims[1])
        saveFigure(fig, base.joinpath(f))
        plt.close()

    # 2 x 3 Grid
    fig, axs = plt.subplots(figsize=TWO_COL_FIGSIZE, ncols=3, nrows=2, sharex='all', sharey='all')
    ymin, ymax = np.inf, -np.inf
    for i, feature in enumerate(CORE_FEATURES):
        COL = i % 2
        ROW = i // 2
        ax = axs[COL][ROW]
        model.plot(feature, ax=ax, annotate=False, visuals=False, **STYLES[feature], lin_reg=False)
        ax.set_xlim(13926.0, 18578.0)
        _grid_annotate(ax, model, feature, NAMES[feature])
        if mean_adapt:
            _, Y = model._get_rdd_plot_data(feature)
            ymin = min(ymin, min(Y))
            ymax = max(ymax, max(Y))
        else:
            ymin = min(ymin, min(model.data[feature]))
            ymax = max(ymax, max(model.data[feature]))

    ydiff = ymax - ymin
    for row in axs:
        for ax in row:
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
            ax.get_legend().remove()
            if mean_adapt:
                ax.set_ylim(ymin - ydiff, ymax + ydiff)
            else:
                ax.set_ylim(ymin - 0.25 * ydiff, ymax + 0.25 * ydiff)
            if ylims is not None:
                ax.set_ylim(ylims[0], ylims[1])

    fig.autofmt_xdate(rotation=75)
    plt.minorticks_off()
    saveFigure(fig, base.joinpath('grid'))
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
    BASE_STYLE = {
        'color': 'tab:red',
        'scatter_color': 'tab:red',
        'annotate': False,
        'parameters': False,
        'lin_reg': False,
        'label': 'Original RDD'
    }

    outliers = pickle.load(model_file.open('rb'))
    base_model = pickle.load(model_file.parent.joinpath(re.sub('_outliers', '', model_file.name)).open('rb'))
    features = [col for col in base_model.data.columns if ('empath' in col) or ('liwc' in col)]

    # Single-Feature Plots
    for f in sorted(features):
        fig, ax = base_model.plot(f, **BASE_STYLE)
        lower, upper = ax.get_ylim()
        fig, ax = outliers.plot(f, ax=ax, label='Outliers Removed')
        ax.set_ylim(lower, upper)  # Reset to include outliers
        ax.legend(fontsize=FONTSIZE - 2, loc='lower left', framealpha=1, fancybox=False, ncol=3)
        saveFigure(fig, store.joinpath(f))
        plt.close()


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


def verbosity_vs_parameter(folder: Path, base: Path, kind: str):
    """
    Plots RDD parameters (y-Axis) vs speaker verbosity (x-axis)
    Parameters
    ----------
    folder: RDD models folder
    base: Img storage folder
    kind: Either "individual" or "ablation" - changes style adjustments

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
                              index=['alpha', 'beta', 'verbosity', 'alpha_low', 'alpha_high', 'beta_low', 'beta_high'])
        for feature in CORE_FEATURES
    }

    for speaker in folder.iterdir():
        if not (speaker.name.endswith('.pickle')):
            continue
        qid = _get_qid(speaker.name)
        speaker_data = pickle.load(speaker.open('rb'))
        # params = speaker_data.rdd
        for feature in plot_data:
            summary = pd.read_html(speaker_data.rdd_fit[feature].summary().tables[1].as_html(), header=0, index_col=0)[
                0]
            alpha_low, alpha_high = summary['[0.025'].loc['C(threshold)[T.1]'], summary['0.975]'].loc[
                'C(threshold)[T.1]']
            beta_low, beta_high = summary['[0.025'].loc['C(threshold)[T.1]:time_delta'], summary['0.975]'].loc[
                'C(threshold)[T.1]:time_delta']
            alpha = summary['coef'].loc['C(threshold)[T.1]']
            beta = summary['coef'].loc['C(threshold)[T.1]:time_delta']
            plot_data[feature].at['alpha', qid] = alpha
            plot_data[feature].at['alpha_low', qid] = alpha_low
            plot_data[feature].at['alpha_high', qid] = alpha_high
            plot_data[feature].at['beta', qid] = beta
            plot_data[feature].at['beta_low', qid] = beta_low
            plot_data[feature].at['beta_high', qid] = beta_high
            plot_data[feature].at['verbosity', qid] = verbosity_df[verbosity_df.QID == qid]['Unique Quotations'].values[0]

    for param in ('alpha', 'beta'):
        fig, axs = plt.subplots(figsize=TWO_COL_FIGSIZE, ncols=3, nrows=2, sharex='all', sharey='all')
        for i, (feature, data) in enumerate(plot_data.items()):
            ROW = i % 2
            COL = i // 2
            ax = axs[ROW][COL]
            ax.set_xscale('log')
            if kind == 'individual':
                ax.axhline(y=0, linestyle='--', color='black', linewidth=0.8)
            else:
                key = 'C(threshold)[T.1]' if param == 'alpha' else 'C(threshold)[T.1]:time_delta'
                baseline = base_model.rdd[feature][key]
                summary = pd.read_html(base_model.rdd_fit[feature].summary().tables[1].as_html(), header=0, index_col=0)[0]
                base_low, base_high = summary['[0.025'].loc[key], summary['0.975]'].loc[key]
                ax.axhline(y=baseline, linestyle='--', color='black')
                ax.fill_between((min(verbosity_df['Unique Quotations']), max(verbosity_df['Unique Quotations'])),
                                base_low, base_high, color='grey', alpha=0.3)
            for qid in data.columns:
                CI_low, CI_high = data[qid].loc[param + '_low'], data[qid].loc[param + '_high']
                # Highlight "significant" points where CIs share a sign
                if kind == 'individual':
                    color = 'tab:red' if (CI_low * CI_high > 0) else 'grey'
                else:
                    color = 'tab:red' if ((baseline - CI_low) * (baseline - CI_high) > 0) else 'grey'
                ax.plot((data[qid].loc['verbosity'], data[qid].loc['verbosity']), (CI_low, CI_high), '-', color=color,
                        linewidth=0.3)
                ax.scatter(x=data[qid].loc['verbosity'], y=data[qid].loc[param], c=color, s=7.5)
            if kind == 'individual':
                annot = r'$\left|\{' + rf'\{param}>0' + r'\}' + rf'\right|={(data.loc[param] > 0).sum()}$' + \
                        r', $\left|\{' + rf'\{param}<0' + r'\}' + rf'\right|={(data.loc[param] < 0).sum()}$'
                box_props = dict(boxstyle='round', facecolor='white', alpha=1)
                ax.text(0.975, 0.05, annot, transform=ax.transAxes, fontsize=FONTSIZE, multialignment='center',
                        verticalalignment='bottom', horizontalalignment='right', bbox=box_props)

            ax.set_title(NAMES[feature], fontsize=FONTSIZE, fontweight='bold')
            if COL > 0:
                # ax.set_yticklabels([])
                ax.tick_params(axis='y', which='both', left=False, right=False)
            else:
                ax.set_ylabel(r'$\{}$'.format(param))
                if (param == 'alpha') and (kind == 'individual'):
                    ax.set_ylim(-20, 20)
                    ax.set_yticks([-10, 10])
                    ax.set_yticklabels(['-10', '10'])
            if ROW == 0:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', bottom=False)
            else:
                ax.set_xlabel('Uttered Quotations')
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
    # individual_plots(folder, base)
    verbosity_vs_parameter(folder, base, 'ablation')


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
            _scatter_only(ax, feature, model, PARTY_STYLES[party]['color'])
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

        box = '\n'.join([
            ',  '.join([
                r'$\alpha_{DEM}$: ' + str(models["democrats"].get_table(asPandas=True).loc['Negemo'][r'$\alpha$']),
                r'$\beta_{DEM}$: ' + str(models["democrats"].get_table(asPandas=True).loc['Negemo'][r'$\beta$'])
            ]), ',  '.join([
                r'$\alpha_{REP}$: ' + str(
                    models["republicans"].get_table(asPandas=True).loc['Negemo'][r'$\alpha$']),
                r'$\beta_{REP}$: ' + str(models["republicans"].get_table(asPandas=True).loc['Negemo'][r'$\beta$'])
            ])
        ])
        box_props = dict(boxstyle='round', facecolor='white', alpha=1)
        ax.text(0.975, 0.95, box, transform=ax.transAxes, fontsize=FONTSIZE, multialignment='center',
                verticalalignment='top', horizontalalignment='right', bbox=box_props)

        fig.autofmt_xdate(rotation=75)
        plt.minorticks_off()

        saveFigure(fig, base.joinpath(feature))
        plt.close()

    # Grid
    fig, axs = plt.subplots(figsize=TWO_COL_FIGSIZE, ncols=3, nrows=2, sharex='all', sharey='all')
    ymin = np.inf
    ymax = - np.inf
    for party, model in models.items():
        for i, feature in enumerate(CORE_FEATURES):
            COL = i % 2
            ROW = i // 2
            ax = axs[COL][ROW]
            ax.set_title(feature, fontsize=FONTSIZE, fontweight='bold')
            model.plot(feature, ax=ax, annotate=False, lin_reg=False, visuals=False, **PARTY_STYLES[party])
            ymin = min(ymin, min(model.data[feature]))
            ymax = max(ymax, max(model.data[feature]))

    ydiff = ymax - ymin
    for row in axs:
        for ax in row:
            ax.legend(fontsize=FONTSIZE, loc='lower left', framealpha=1, fancybox=False, ncol=2)
            ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
            ax.set_ylim(ymin - 0.25 * ydiff, ymax + 0.25 * ydiff)
            ax.set_xlim(13926.0, 18578.0)

    fig.autofmt_xdate(rotation=75)
    plt.minorticks_off()
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
        0: {'color': plasma(0.), 'linewidth': 3, 'label': 'Most verbose quartile'},
        1: {'color': plasma(.25), 'linewidth': 3, 'label': '2nd most verbose quartile'},
        2: {'color': plasma(.5), 'linewidth': 3, 'label': '3rd most verbose quartile'},
        3: {'color': plasma(.75), 'linewidth': 3, 'label': 'Least most verbose quartile'}
    }

    lower, upper = (13926.0, 18578.0)  # Hard coded numeric Quotebank Date limits + margin
    for feature in features:
        fig, axs = plt.subplots(figsize=NARROW_TWO_COL_FIGSIZE, ncols=4)
        y_min = np.Inf
        y_max = - np.Inf

        for verbosity, model in models.items():
            ax = axs[verbosity]
            model.plot(feature, ax=ax, annotate=False, visuals=False)
            # Adapt y-limits of the plot to mean
            _, Y = model._get_rdd_plot_data(feature)
            y_min = min(y_min, min(Y))
            y_max = max(y_max, max(Y))
            ax.get_legend().remove()
            ax.set_title(V_style[verbosity]['label'], fontsize=FONTSIZE, fontweight='bold')

        y_diff = y_max - y_min
        for ax in axs:
            ax.set_xlim(lower, upper)
            ax.set_ylim(y_min - y_diff, y_max + y_diff)

        fig.autofmt_xdate(rotation=75)
        plt.minorticks_off()

        saveFigure(fig, base.joinpath(feature))
        plt.close()


def attribute_plots(model_path: Path, base: Path):
    attributes = ['party', 'governing_party','gender', 'congress_member',
                  'C(threshold)[T.1]', 'C(threshold)[T.1]:time_delta']
    model = pickle.load(model_path.open('rb'))

    ticks = {
        'gender': ['Male', 'Female'],
        'party': ['Republican', 'Democratic'],
        'congress_member': ['Others', 'Congress'],
        'governing_party': ['Opposition', 'Government']
    }
    titles = {
        'gender': 'Gender',
        'party': 'Party',
        'congress_member': 'Congress Affiliation',
        'governing_party': 'Government Affiliation',
        'C(threshold)[T.1]': r'$\mathbf{\alpha}$',
        'C(threshold)[T.1]:time_delta': r'$\mathbf{\beta}$'
    }

    styles_cpy = {NAMES[key]: val for key, val in STYLES.items()}

    ORDER = ['liwc_Posemo', 'liwc_Negemo', 'liwc_Anger', 'liwc_Anx', 'liwc_Sad', 'liwc_Swear']

    def _sort_features(idx: pd.Index) -> pd.Index:
        srt = sorted(idx.values, key=lambda x: ORDER.index(x))
        return pd.Index(srt)

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

        if 'threshold' not in att:
            ax.set_xlim([-2, 4.5])
            ax.set_xticks([-1, 0, 1, 2, 3, 4])
            ax.text(x=0.05, y=-0.2, s=r'{}$\longleftarrow$'.format(ticks[att][0]), fontsize=FONTSIZE, ha='left', va='bottom', transform=ax.transAxes)
            ax.text(x=0.95, y=-0.2, s=r'$\longrightarrow${}'.format(ticks[att][1]), fontsize=FONTSIZE, ha='right', va='bottom', transform=ax.transAxes)
            ax.text(x=0.5, y=-0.2, s='$\mathbf{\sigma}$', fontsize=FONTSIZE, ha='center', va='bottom', transform=ax.transAxes)

        ax.tick_params(axis='both', labelsize=FONTSIZE)
        ax.set_title(titles[att], fontsize=FONTSIZE, fontweight='bold')
        ax.set_title('abcdef'[i] + ')', fontfamily='serif', loc='left', fontsize=FONTSIZE)
        if att != 'C(threshold)[T.1]':
            ax.axvline(x=0, linestyle='dashed', color='black', linewidth=0.5)
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


def main():
    args = parser.parse_args()
    data = Path(args.rdd)
    img = Path(args.img)

    # Map a file or folder name to a plotting utility.
    NAME_2_FUNCTION = {
        # 'verbosity': verbosity_plots,
        # 'parties': party_plots,
        'Individuals': individuals,
        'Without': ablation_plots,
        # 'QuotationAggregation_RDD': basic_model_plots,
        # 'QuotationAggregationTrump_RDD': basic_model_plots,
        # 'QuotationAggregation_RDD_outliers': outlier_plots,
        # 'SpeakerAggregation_RDD': basic_model_plots,
        # 'SpeakerAggregation_RDD_outliers': outlier_plots,
        # 'AttributesAggregation_RDD': attribute_plots
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
