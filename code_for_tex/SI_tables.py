from pathlib import Path
import pandas as pd
import pickle
import re
import sys
from typing import List

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from analysis.figures import CORE_FEATURES, NAMES, KINK

AGG_FOLDER = Path(__file__).parent.parent.joinpath('data').joinpath('aggregates')
RDD = AGG_FOLDER.parent.joinpath('RDD')
SAVE = Path(__file__).parent.parent.joinpath('data').joinpath('tables')
SENTIMENT = SAVE.joinpath('sentiment_words')
SI = SAVE.parent.parent.joinpath('SI')
HEADERS = [r'$\mu$', r'$\sigma$', 'Every Nth', r'$\sigma / \mu$']

LABEL_COUNTS = dict()
DISC = KINK.strftime('%d %B %Y')


def mean_std_table(mean: pd.Series, std: pd.Series, features: List[str] = None) -> pd.DataFrame:
    if features is not None:
        mean = mean[features]
        std = std[features]
    every_nth_word = 1 / mean
    std_increase = 100 * (std / mean)
    summary = pd.DataFrame(data=[mean, std, every_nth_word, std_increase], index=HEADERS).rename(columns=NAMES).T
    summary[summary.columns[0]] = summary[summary.columns[0]].apply(lambda x: f'{x:.5f}')
    summary[summary.columns[1]] = summary[summary.columns[1]].apply(lambda x: f'{x:.5f}')
    summary[summary.columns[2]] = summary[summary.columns[2]].apply(lambda x: f'{x:.1f}')
    summary[summary.columns[3]] = summary[summary.columns[3]].apply(lambda x: f'{x:.1f}\\%')
    summary.columns.name = '$y_{cat}$'
    return summary


def make_table(file: Path, caption: str, label: str) -> str:
    file_content = file.open('r').read()

    adjustbox = False if caption in ['Key Metrics for all Sentiment Attributes',
                                     f'Kullback-Leibler Divergence for the word distribution within the Categories, before and after {DISC}.'] else True

    label = ''.join(letter for letter in label if (letter not in r'$\\'))
    txt = r'\begin{table}[h]\centering' + '\n'
    if adjustbox:
        txt += r'\begin{adjustbox}{width=\linewidth, center}' + '\n'

    for line in file_content.split('\n'):
        txt += '\t' + line + '\n'

    if adjustbox:
        txt += r'\end{adjustbox}' + '\n\t'
    txt += r'\caption{' + caption + '}\n\t'
    txt += r'\label{fig: ' + label + '}' + '\n'
    txt += r'\end{table}' + '\n\n'
    return txt


def _get_rdd_caption(fname: str) -> str:
    caption = 'RDD parameters for'

    def sw(x): return fname.lower().startswith(x)
    if sw('attributes'):
        caption += ' aggregation on distinct speaker groups'
    elif sw('quotation'):
        caption += ' quotation aggregation'
    elif sw('speaker'):
        caption += ' speaker aggregation'
    elif sw('party'):
        caption += ' aggregation over different parties'
    elif sw('verbosity'):
        caption += ' aggregation on verbosity quartiles'

    if 'outliers' in fname:
        caption += ', excluding outliers'

    return caption + '.'


def _get_augmented_caption(pname: str, fname: str) -> str:
    base = _get_rdd_caption(fname)[:-1] + ', ' # Remove the "."
    verbosity_map = {'0': 'most', '1': '2nd most', '2': '3rd most', '3': 'least'}
    if pname == 'verbosity':
        grp = re.search('[0-9]', fname)[0]
        caption = base + verbosity_map[grp] + ' verbose quartile of politicians.'
    elif pname == 'parties':
        party = 'democrats' if 'democrats' in fname else 'republicans'
        caption = base + party + ' only.'
    else:
        raise NotImplementedError("Don't know how to create caption for folder {}.".format(pname))
    return caption


def _get_sentiment_caption(fname: str) -> str:
    if 'after' in fname:
        t = 'after ' + DISC
    elif 'before' in fname:
        t = 'before ' + DISC
    else:
        t = None

    if re.search('top_[0-9]+', fname.lower()):
        n = re.search('[0-9]+', fname)[0]
        return f'The most common {n} words for each sentiment category, ' + t + '. Bold values deviate from the other' \
                                                                                ' period by more than one percentage point.'
    elif 'kullbackleibler' in fname.lower():
        return f'Kullback-Leibler Divergence for the word distribution within the Categories, before and after {DISC}.'

    print(fname.lower())
    raise NotImplementedError


def _get_label(fname: str) -> str:
    """Returns the first word for most naming conventions"""
    label = re.match('[A-Z]?[a-z]+', fname)[0]
    LABEL_COUNTS[label] = LABEL_COUNTS.get(label, 0) + 1
    return label + '_' + str(LABEL_COUNTS[label])


def get_rdd_tables() -> str:
    txt = ''
    # First all .tex files, then go through folders. That's inefficient, but that doesn't actually matter here
    for path in sorted(RDD.iterdir()):
        if not path.name.endswith('.tex'):
            continue

        label = _get_label(path.name)
        caption = _get_rdd_caption(path.name)
        txt += make_table(path, caption, label)

    for path in sorted(RDD.iterdir()):
        if path.name in ('parties', 'verbosity'):
            for file in sorted(path.iterdir()):
                if not file.name.endswith('.tex'):
                    continue
                label = _get_label(file.name)
                caption = _get_augmented_caption(path.name, file.name)
                txt += make_table(file, caption, label)

    return txt


def get_sentiment_tables():
    txt = ''
    # First all .tex files, then go through folders. That's inefficient, but that doesn't actually matter here
    for path in sorted(SENTIMENT.iterdir(), reverse=True):
        if not path.name.endswith('.tex'):
            continue

        label = _get_label(path.name)
        caption = _get_sentiment_caption(path.name)
        txt += make_table(path, caption, label)

    return txt


def main():
    SAVE.mkdir(exist_ok=True)

    # Make Tables with Mean and STD
    mean = pickle.load(AGG_FOLDER.joinpath('mean.pickle').open('rb'))  # Pandas Series
    std = pickle.load(AGG_FOLDER.joinpath('std.pickle').open('rb'))  # Pandas Series
    features = [c for c in mean.index if ('empath' in c) or ('liwc' in c)]
    for name, feat in zip(('all', 'core'), (features, CORE_FEATURES)):
        summary = mean_std_table(mean, std, feat)
        with SAVE.joinpath('mean_std_' + name + '_metrics.tex').open('w') as dump:
            dump.write(summary.to_latex(sparsify=True, escape=False))

    txt = make_table(SAVE.joinpath('mean_std_all_metrics.tex'), 'Key Metrics for all Sentiment Attributes', 'mean_std')

    # Get the latest RDD parameter tables
    txt += get_rdd_tables()

    txt += get_sentiment_tables()

    with SI.joinpath('tables.tex').open('w') as tex_file:
        tex_file.write(txt)


if __name__ == '__main__':
    main()
