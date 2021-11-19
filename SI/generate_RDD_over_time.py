import argparse
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
import analysis.RDD as RDD

START = datetime(2008, 9, 15)
END = datetime(2020, 4, 15)


parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Quotation aggregation csv file', required=True)
parser.add_argument('--save', help='Folder to save RDD fits to',  required=True)


def main():
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    storage_folder = Path(args.save)
    storage_folder.mkdir(exist_ok=True, parents=True)

    date = START
    ret = dict()
    while date < END:
        RDD.KINK = date
        try:
            rdd = RDD.RDD_statsmodels(data)
            cpy = rdd.copy()
            for feature in rdd:
                rdd[feature] = {key: cpy[feature][key] for key in ['f_score', 'r2', 'r2_adjust', 'RMSE', 'SSR']}
            ret[date] = rdd
        except IndexError:  # No data at the given month. Splitting here would yield to an already observed result
            pass
        date = date + relativedelta(months=1)

    pickle.dump(ret, storage_folder.joinpath('RDD_time_variation.pickle').open('wb'))


if __name__ == '__main__':
    main()
