#!/usr/bin/env python3
# Author: Jonathan Külz
import argparse
from datetime import datetime
from pathlib import Path
import pickle
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from analysis.RDD import RDD_statsmodels

parser = argparse.ArgumentParser()
parser.add_argument('--aggregates', type=str, required=True, help='Path to aggregate file')
parser.add_argument('--save_as', type=str, required=True, help='Path to save file (without extension)')

def main():
    """
    This script takes monthly aggregated word scores and fits a linear regression model with a discontinuity to it (RDD)
    It alternates over all possible RDD fits and stores the scores for fit quality over time as pickle file
    """
    args = parser.parse_args()
    df = pd.read_csv(args.aggregates)
    dates = [datetime.strptime(dt, '%Y-%m-%d') for dt in df.date]

    scores = dict()

    for dt in dates:
        results = RDD_statsmodels(df, kink=dt)
        for kw in results:
            if 'summary' in results[kw]:
                del results[kw]['summary']
        scores[dt] = results

    with open(args.save_as + '.pickle', 'wb') as f:
        pickle.dump(scores, f)

if __name__ == '__main__':
    main()