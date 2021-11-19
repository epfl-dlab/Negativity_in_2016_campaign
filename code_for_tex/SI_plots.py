from collections import OrderedDict
from pathlib import Path
import re
import sys

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from analysis.figures import NAMES

SECTIONS = OrderedDict()
SECTIONS['Quotation Aggregation'] = '',
SECTIONS['Quotation Aggregation, Excluding Outliers'] = ''
SECTIONS['Speaker Aggregation'] = '',
SECTIONS['Speaker Aggregation, Excluding Outliers'] = ''
SECTIONS['Verbosity Quartiles, Speaker Aggregation'] = ''
SECTIONS['Quotation Aggregation, Split by Parties'] = ''
SECTIONS['Quotation Aggregation Scores for Individual Politicians'] = ''
SECTIONS['Quotation Aggregation Scores for all but one Speaker'] = ''
SECTIONS['Influence of the Placement of the Discontinuity on RDD performance'] = ''

KIND_CATPIONS = {
    'qa': 'Quotation Aggregation',
    'sa': 'Speaker Aggregation',
    'parties': 'Quotation Aggregation, Seperated for Parties',
    'verbosity': 'Speaker Aggregation with Speakers Divided in Verbosity Quartiles',
    'individuals': 'Quotation Aggregation Scores for Individual Politicians',
    'ablation': 'Quotation Aggregation Scores for all but one Speaker',
    'rddtv': 'Influence of the Placement of the Discontinuity on RDD performance'
}


def _get_kind(name: str) -> str:
    if name.lower() in ('parties', 'verbosity', 'individuals', 'without'):
        return name.lower()
    elif name.lower().startswith('attributes'):
        return 'attributes'
    elif name.lower().startswith('quotationaggregation'):
        return 'qa'
    elif name.lower().startswith('speakeraggregation'):
        return 'sa'
    elif name.lower() == 'custom_ci':
        return 'ablation'
    elif name.lower().startswith('rdd_time'):
        return 'rddtv'
    else:
        return 'UNK'


def _get_section(kind: str, is_outlier: bool) -> str:
    MAP = {
        'qa': 'Quotation Aggregation',
        'sa': 'Speaker Aggregation',
        'parties': 'Quotation Aggregation, Split by Parties',
        'verbosity': 'Verbosity Quartiles, Speaker Aggregation',
        'individuals': 'Quotation Aggregation Scores for Individual Politicians',
        'ablation': 'Quotation Aggregation Scores for all but one Speaker',
        'rddtv': 'Influence of the Placement of the Discontinuity on RDD performance'
    }
    section = MAP[kind]
    if is_outlier:
        section += ', Excluding Outliers'
    return section


def get_fig_caption(name: str, kind: str, is_outlier: bool) -> str:
    caption = f"{name}: {KIND_CATPIONS[kind]}"
    if is_outlier:
        caption += ' (Outliers removed)'
    return caption


def get_feature_name(name: str):
    try:
        return NAMES[re.sub('.pdf', '', name)]
    except KeyError:
        return None


def make_figure(path: Path, caption: str, label: str) -> str:
    label = ''.join(letter for letter in label if (letter not in r'$\\'))
    txt = r'\begin{figure}[h]\centering' + '\n\t'
    txt += r'\includegraphics[width =\textwidth]{' + str(path) + '}' + '\n\t'
    txt += r'\caption{' + caption + '}\n\t'
    txt += r'\label{fig: ' + label + '}' + '\n'
    txt += r'\end{figure}' + '\n\n'
    return txt


def plots(plot_dir: Path):
    """Creates a tex file with labeled plot references."""

    for folder in plot_dir.iterdir():
        kind = _get_kind(folder.name)
        if kind in ['attributes', 'without']:
            continue  # Not gonna be used

        is_outlier = 'outliers' in folder.name
        txt = ''

        for fig in sorted(folder.iterdir()):

            if not fig.name.endswith('.pdf'):
                continue

            if kind in ['individuals', 'ablation']:
                print('ALARM!')

            feature = get_feature_name(fig.name)
            if feature is None:
                print(fig.name)
                continue
            caption = get_fig_caption(feature, kind, is_outlier)
            txt += make_figure(fig, caption, kind + '_' + feature)

        txt += r'\clearpage' + '\n'
        txt += r'\pagebreak' + '\n\n'
        SECTIONS[_get_section(kind, is_outlier)] += txt


def main():
    """Note that this works only if you kept the original project structure."""
    SI = Path(__file__).parent.parent.joinpath('SI')
    SI.mkdir(exist_ok=True)
    data = SI.parent.joinpath('data')

    for section_name in SECTIONS.keys():
        SECTIONS[section_name] = r'\subsection{' + section_name + '}\n\n'

    plots(SI.parent.joinpath('img'))

    with SI.joinpath('plots.tex').open('w') as tex_file:
        tex_file.write(''.join(val for val in SECTIONS.values()))


if __name__ == '__main__':
    main()
