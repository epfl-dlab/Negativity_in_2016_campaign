#!/usr/bin/env python3
# Author: Jonathan Külz
import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from analysis.RDD import RDD_statsmodels

parser = argparse.ArgumentParser()
parser.add_argument('--aggregates', type=str, required=True, help='Path to aggregate file')

def main():
    """
    This script takes monthly aggregated word scores and fits a linear regression model with a discontinuity to it (RDD)
    It alternates over all possible RDD fits and stores the scores for fit quality over time as pickle file
    """
    # args = parser.parse_args()
    # df = pd.read_csv(args.aggregates)
    df = pd.read_csv('/dlabdata1/kuelz/Negativity_in_2016_campaign/data/aggregates/YouGov_sources.csv')

    dates = [datetime.strptime(dt, '%Y-%m-%d') for dt in df.date]

    scores = dict()

    for dt in dates:
        results = RDD_statsmodels(df, kink=dt)
        for kw in results:
            if 'summary' in results[kw]:
                del results[kw]['summary']

    return results


if __name__ == '__main__':
    main()