from pathlib import Path
import pandas as pd
import pickle
import re
import sys
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from analysis.aggregate import DEMOCRATIC_PARTY, REPUBLICAN_PARTY
from analysis.figures import CORE_FEATURES, KINK, V_label, SI_FEATURES
from analysis.RDD import NAMES

AGG_FOLDER = Path(__file__).parent.parent.joinpath('data').joinpath('aggregates')
RDD = AGG_FOLDER.parent.joinpath('RDD')
SAVE = Path(__file__).parent.parent.joinpath('data').joinpath('tables')
SENTIMENT = SAVE.joinpath('sentiment_words')
SI = SAVE.parent.parent.joinpath('SI')
HEADERS = [r'$\mu$', r'$\sigma$', r'$1/\mu$', r'$\sigma / \mu$']
NAMES = {key: re.sub('\n', ' ', val).lower() for key, val in NAMES.items()}

LABEL_COUNTS = dict()
DISC = KINK.strftime('%d %B %Y')

ORDER = [
    'MostFrequentWords',
    'QuotationAggregation',
    'SpeakerAggregation',
    'PartyAggregationDemocrat',
    'PartyAggregationRepublican',
    'PartyAggregationWithoutTrump',
    'VerbosityAggregation',
    'AttributesAggregation',
]

DESCRIPTIONS = {
    'metrics': r"Means $\mu$ and standard deviations $\sigma$ were calculated over monthly quote-level aggregates from the pre-campaign period (September 2008 through May 2015). "
        "The value $n=1/\mu$ in the fourth column implies that, in an average quote, on average every $n$-th word belongs to the respective category. "
        "The coefficients of variation, $\sigma/\mu$, shown in the fifth column, allow to easily translate pre-campaign standard deviations "
        "(as shown on the y-axes of time series plots) into fractions of pre-campaign means. The most frequent words per category are listed "
        "in SI Tables S2 and S3.",
    'QuotationAggregation': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'QuotationAggregation_out': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'SpeakerAggregation': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'SpeakerAggregation_out': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'AttributesAggregation': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'AttributesAggregationSpeakerLevel': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'PartyAggregationDemocrat': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'PartyAggregationRepublican': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'PartyAggregationWithoutTrump': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'VerbosityAggregation_0': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'VerbosityAggregation_1': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'VerbosityAggregation_2': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'VerbosityAggregation_3': 'SEs of coefficients are in parantheses. ***$p < 0.001$, **$p < 0.01$ and *$p < 0.05$',
    'TopSpeaker': None
}


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
    summary.columns.name = 'Word category'
    keep = [NAMES[f] for f in SI_FEATURES]
    keep = [f for f in keep if f in summary.index]
    return summary.loc[keep]


def make_table_from_disk(file: Path, caption: str, label: str, desription: str) -> str:
    file_content = file.open('r').read()
    return make_table(file_content, caption, label, desription)


def make_table(tabular: str, caption: str, label: str, description: str) -> str:

    adjustbox = False if caption in ['(Extended version of Table 1 in the main text) Key metrics for all sentiment attributes, including results for empath word categories',
                                     'Most frequently quoted politicians'] else True

    label = ''.join(letter for letter in label if (letter not in r'$\\'))
    txt = r'\begin{table}[h]\centering' + '\n'
    txt += r'\caption{\textbf{' + caption + '}. ' + description + '}\n\t'
    txt += r'\label{fig: ' + label + '}' + '\n'
    if adjustbox:
        txt += r'\begin{adjustbox}{width=\linewidth, center}' + '\n'

    for line in tabular.split('\n'):
        txt += '\t' + line + '\n'

    if adjustbox:
        txt += r'\end{adjustbox}' + '\n\t'
    txt += r'\end{table}' + '\n\n'
    return txt


def _get_label(fname: str) -> str:
    """Returns the first word for most naming conventions"""
    label = re.match('[A-Z]?[a-z]+', fname)[0]
    LABEL_COUNTS[label] = LABEL_COUNTS.get(label, 0) + 1
    return label + '_' + str(LABEL_COUNTS[label])


def _get_sentiment_caption(fname: str) -> Tuple[str, str]:
    if 'after' in fname:
        t = 'after ' + DISC
        otherPeriod = 'first'
        other = 'before ' + DISC
    elif 'before' in fname:
        t = 'before ' + DISC
        otherPeriod = 'second'
        other = 'after ' + DISC
    else:
        t = None

    if re.search('top_[0-9]+', fname.lower()):
        n = re.search('[0-9]+', fname)[0]
        caption = f'The most common {n} words for each word category, ' + t
        descr = f'Bold values deviate from the {otherPeriod} period ({other}) by more than one percentage point'
        return caption, descr
    else:
        raise NotImplementedError


def _get_rdd_caption(fname: str) -> Tuple[str, str]:
    """Creates a caption that will appear in the SI"""

    hasOutliers = 'outliers' in fname.lower()
    name = re.search('(.*?)_tabular', fname)[1]
    caption = 'OLS regression parameters for '
    agg = None

    if name == 'AttributesAggregation':
        caption += "quote-level aggregation on speaker groups based on party affiliation, party's federal role, Congress membership, and gender"
    elif name == 'AttributesAggregationSpeakerLevel':
        caption += "speaker-level aggregation on speaker groups based on party affiliation, party's federal role, Congress membership, and gender"
    elif name == 'QuotationAggregation':
        caption += ' quote-level aggregation'
        if not hasOutliers:
            caption += ' shown in Figure 2 in the main text'
    elif name == 'SpeakerAggregation':
        caption += ' speaker-level aggregation'
        if not hasOutliers:
            caption += ' shown in Figure 2 in the main text'
    elif name == 'PartyAggregation':
        if 'democrats' in fname:
            agg = 'PartyAggregationDemocrat'
            caption += 'quote-level aggregation for Democrats'
        else:
            agg = 'PartyAggregationRepublican'
            caption += 'quote-level aggregation for Republicans'
        if not hasOutliers:
            caption += ' shown in Figure 2 in the main text'
    elif name == 'VerbosityAggregation':
        group = int(re.search('[0-9]', fname)[0])
        caption += 'aggregation on the {}'.format(V_label[group].lower())
        agg = name + '_{}'.format(group)
    elif name == 'PartyAggregationWithoutTrump' and 'republicans' in fname:
        caption += 'quote-level aggregation for Republicans, but excluding Donald Trump'
    else:
        return None, None

    if hasOutliers:
        if name in ['QuotationAggregation', 'SpeakerAggregation']:
            caption += ', excluding outliers'
        else:
            return None, None

    if agg is None:  # Not assigned yet
        agg = name

    if hasOutliers:
        appendix = '_out'
    else:
        appendix = ''

    return caption, agg + appendix


def get_rdd_tables() -> Dict[str, str]:
    txt = {}
    # First all .tex files, then go through folders. That's inefficient, but that doesn't actually matter here
    for path in sorted(RDD.iterdir()):
        if not path.name.endswith('.tex'):
            continue

        label = _get_label(path.name)
        caption, agg = _get_rdd_caption(path.name)
        if caption is None:
            print('exclude:', path)
            continue

        assert agg not in txt, "Double Tables?"
        txt[agg] = make_table_from_disk(path, caption, label, DESCRIPTIONS[agg])

    for path in sorted(RDD.iterdir()):
        if path.name in ('parties', 'verbosity'):
            for file in sorted(path.iterdir()):
                if not file.name.endswith('.tex'):
                    continue
                label = _get_label(file.name)
                caption, agg = _get_rdd_caption(file.name)
                if caption is None:
                    print('exclude:', file)
                    continue

                assert agg not in txt, "Double Tables?"
                txt[agg] = make_table_from_disk(file, caption, label, DESCRIPTIONS[agg])

    return txt


def get_sentiment_tables():
    txt = ''
    # First all .tex files, then go through folders. That's inefficient, but that doesn't actually matter here
    for path in sorted(SENTIMENT.iterdir(), reverse=True):
        if not path.name.endswith('.tex'):
            continue

        label = _get_label(path.name)
        caption, description = _get_sentiment_caption(path.name)
        txt += make_table_from_disk(path, caption, label, description)

    return txt


def get_top_speaker_tables():
    TOPS = AGG_FOLDER.parent.joinpath('speaker_counts_with_biographics.csv')
    speaker = pd.read_csv(TOPS)
    party_map = {0: 'Republican', 1: 'Democrat'}
    gender_map = {0: 'M', 1: 'F'}
    table = speaker[:30][['qid', 'name', 'num_quotes', 'party', 'gender']]
    table.party = table.party.map(party_map)
    table.gender = table.gender.map(gender_map)
    table = table.rename(columns={'qid': 'QID', 'party': 'Party', 'name': 'Name', 'gender': 'Gender', 'num_quotes': 'Number of quotes'})
    table.index = range(1, len(table) + 1)
    return make_table(table.to_latex(), 'Most frequently quoted politicians and the Wikidata identifiers (QID)', 'Top30', '')


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

    txt = make_table_from_disk(SAVE.joinpath('mean_std_all_metrics.tex'),
                               '(Extended version of Table 1 in the main text) Key metrics for all sentiment attributes, including results for empath word categories',
                               'mean_std', DESCRIPTIONS['metrics'])

    # Get the latest RDD parameter tables
    RDD = get_rdd_tables()

    for key in ORDER:
        candidates = [agg for agg in RDD.keys() if key in agg]
        for c in sorted(candidates):
            txt += RDD[c]
        if key == 'MostFrequentWords':
            txt += get_sentiment_tables()

    txt += get_top_speaker_tables()

    with SI.joinpath('tables.tex').open('w') as tex_file:
        tex_file.write(txt)


if __name__ == '__main__':
    main()
