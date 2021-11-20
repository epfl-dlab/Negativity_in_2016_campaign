import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pickle
import re
import sys
from typing import Dict
import warnings

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from analysis.figures import CORE_FEATURES, NAMES, ONE_COL_FIGSIZE, FONTSIZE


parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Pickle file made by the according analysis script', required=True)
parser.add_argument('--img', help='Image Folder', required=True)
parser.add_argument('--tables', help='Tables Folder', required=True)


def _manual_escape(s: str) -> str:
    """Escapes certain characters for LaTex."""
    escapeMe = '&%$#_{}~^\\'
    ret = ''
    for char in s:
        if char in escapeMe:
            ret += rf'\{char}'
        else:
            ret += char
    return ret


def _extract_numeric(s: str) -> float:
    """Finds the first number in a string and returns it as float"""
    try:
        return float(re.search(r'-?[0-9]*\.?[0-9]+', s)[0])
    except TypeError:
        raise ValueError('String {} does not contain numeric values.'.format(s))


def discreteKullbackLeibler(P: np.array, Q: np.array) -> np.array:
    """
    Takes two arrays with distributions over a set of discrete values and returns the KL divergence
    Parameters
    ----------
    P: Probabilities P
    Q: Probabilities Q
    Returns
    -------
    Pointwise Kullback-Leibler Divergence D(P|Q)
    """
    if (len(P) != len(Q)) or not np.isclose(sum(P), 1) or not np.isclose(sum(Q), 1):
        warnings.warn("The Probability Distributions for KL-Divergence seem to be invalid.")
    return (P * np.log(P/Q)).sum()


def evaluate_sentiment_patterns(counts: Dict, wordMap: Dict, categoryMap: Dict, savePlot: Path, saveTable: Path) -> None:
    """
    Performs several evaluation utilities on a group of sentiment word counts
    Parameters
    ----------
    counts: A mapping Name->Counts, where counts is an array of word/regex counts
    wordMap: Maps indices from the counts array to the respective word
    categoryMap: Maps indices from the counts array to the respective super category
    savePlot: Parent folder where to save plots to.
    saveTable: Parent folder where to save tables to.
    Returns
    -------
    Nothing
    """
    if ('after' not in counts) or ('before' not in counts):
        raise NotImplementedError("Evaluation only for available for a before and after category currently.")
    topTables = dict()
    TOP_N = 15
    for name, cnt in counts.items():
        #  Get the share of each word for each category
        tops = pd.DataFrame(index=[str(i+1) for i in range(TOP_N)] + ['Total nr Expressions', 'Total Matches'],
                            columns=sorted(set(categoryMap.values())))
        tops.index.name = 'Top N words'
        for cat in tops.columns:
            indices = np.asarray([idx for idx, ctgry in categoryMap.items() if ctgry == cat])
            matchingCounts = cnt[indices]
            matchingWords = [word for idx, word in wordMap.items() if idx in indices]
            sortIndices = np.argsort(matchingCounts)[::-1]  # Sorting large to small values
            totalMatches = sum(matchingCounts)
            topsInfo = [f"{matchingWords[i]}: {100 * matchingCounts[i] / totalMatches:.2f}%" for i in sortIndices[:TOP_N]]
            tops[cat] = topsInfo + [str(len(matchingWords)), str(int(totalMatches))]

        tops = tops.applymap(_manual_escape)
        topTables[name] = tops

    numericIndices = [str(i+1) for i in range(TOP_N)]
    diff = topTables['before'].loc[numericIndices].applymap(_extract_numeric) - topTables['after'].loc[numericIndices].applymap(_extract_numeric)
    bold = diff.applymap(abs) > 1

    def _maybeBold(s: str, isBold: bool):
        if isBold:
            s = r'\textbf{' + s + '}'
        return s

    for name in counts:
        for idx in numericIndices:
            for cat in topTables[name].columns:
                topTables[name].at[idx, cat] = _maybeBold(topTables[name].at[idx, cat], bold.at[idx, cat])

    for name, tops in topTables.items():
        saveTable.joinpath(f'{name}_top_{TOP_N}.tex').open('w').write(tops.to_latex(sparsify=True, escape=False))

    # Kullback-Leibler Divergence
    kl = pd.Series(index=tops.columns, dtype=float)
    kl.name = 'KL Divergence'
    for cat in tops.columns:
        p = dict()
        for name, cnt in counts.items():
            indices = np.asarray([idx for idx, ctgry in categoryMap.items() if ctgry == cat])
            matchingCounts = cnt[indices]
            totalMatches = sum(matchingCounts)
            p[name] = matchingCounts / totalMatches
        keep = np.logical_and(p['before'] != 0, p['after'] != 0)
        kl[cat] = discreteKullbackLeibler(p['before'][keep], p['after'][keep])

    saveTable.joinpath('KullbackLeiblerDivergenceSentiment.tex').open('w').write(kl.to_latex(escape=True))


def main():
    args = parser.parse_args()
    data = pickle.load(open(args.data, 'rb'))
    before = np.asarray(data['counts_before'][0])  # Unpacking the DenseVector
    after = np.asarray(data['counts_after'][0])  # Unpacking the DenseVector

    plots = Path(args.img)
    tables = Path(args.tables)

    def _extract_pattern(p: str) -> str:
        """Takes an annotated and spark-ready regular expression and returns a human-readable version"""
        return re.search(r'\\b(.*)\\b', p)[1]

    idx2regex = {i: _extract_pattern(ptrn) for i, ptrn in data['mapping'].items()}
    idx2cat = {i: NAMES['_'.join(ptrn.split('_')[:2])] for i, ptrn in data['mapping'].items()}  # Human-readable naming

    evaluate_sentiment_patterns({'before': before, 'after': after}, idx2regex, idx2cat, plots, tables)


if __name__ == '__main__':
    main()
