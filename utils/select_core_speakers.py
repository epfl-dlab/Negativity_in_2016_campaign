from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

parser = ArgumentParser()
parser.add_argument('--monthly_threshold', help='Minimum number of quotations to select a month as "active".', type=int,
                    default=50)
parser.add_argument('--months_covered', help='Minimum number of "active" months to be in the core speaker set.',
                    type=int, default=120)
parser.add_argument('--quotations', help='Path to quotations parquet file. Needs a single QID column', required=True)
parser.add_argument('--save', help='Path to save data at (csv)', required=True)


def main():
    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    save = Path(args.save)
    save.parent.mkdir(parents=True, exist_ok=True)

    df = spark.read.parquet(args.quotations)
    speaker = df \
        .groupby(['qid', 'year', 'month']) \
        .agg(f.count('*').alias('numQuotations')) \
        .filter(f.col('numQuotations') >= args.monthly_threshold) \
        .groupby(['qid']) \
        .agg(f.count('*').alias('active_months'), f.sum('numQuotations').alias('numQuotations')) \
        .filter(f.col('active_months') >= args.months_covered) \
        .sort(f.desc('numQuotations')) \
        .rdd \
        .map(lambda x: (x.qid, x.active_months, x.numQuotations)) \
        .collect()

    pd_df = pd.DataFrame(data=speaker, columns=['QID', 'active_months', 'num_quotations'])
    pd_df.to_csv(save, index=True)


if __name__ == '__main__':
    main()
