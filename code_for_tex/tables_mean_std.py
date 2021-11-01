import argparse
from pathlib import Path
import pandas as pd
import pickle
import sys
from typing import List

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from analysis.figures import CORE_FEATURES, NAMES

parser = argparse.ArgumentParser()
parser.add_argument('--agg_folder', help='Aggregation folder as created by the aggregation script', required=True)
parser.add_argument('--save', help='Folder where to store tex tables.', required=True)


HEADERS = [r'$\mu$', r'$\sigma$', 'Every Nth', r'$\sigma / \mu$']


def make_table(mean: pd.Series, std: pd.Series, features: List[str] = None) -> pd.DataFrame:
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


def main():
    args = parser.parse_args()
    folder = Path(args.agg_folder)
    save = Path(args.save)
    save.mkdir(exist_ok=True)
    mean = pickle.load(folder.joinpath('mean.pickle').open('rb'))  # Pandas Series
    std = pickle.load(folder.joinpath('std.pickle').open('rb'))  # Pandas Series
    features = [c for c in mean.index if ('empath' in c) or ('liwc' in c)]
    for name, feat in zip(('all', 'core'), (features, CORE_FEATURES)):
        summary = make_table(mean, std, feat)
        with save.joinpath(name + '_metrics.tex').open('w') as dump:
            dump.write(summary.to_latex(sparsify=True, escape=False))


if __name__ == '__main__':
    main()
