import argparse
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

parser = argparse.ArgumentParser()
parser.add_argument('--quotations', help='Dataframe (.parquet) containing quotations and a QID column', required=True)
parser.add_argument('--save', help='Folder to store the counts in. (.csv)', required=True)


def main():
    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    df = spark.read.parquet(args.quotations)
    counts = df \
        .groupby('qid') \
        .agg(f.countDistinct('quoteID').alias('cnt')) \
        .rdd \
        .map(lambda r: (r.qid, r.cnt)) \
        .collect()

    sorted_counts = list(zip(range(len(counts)), sorted(counts, key=lambda x: x[1], reverse=True)))
    df = pd.DataFrame(data=None, index=range(len(sorted_counts)), columns=['QID', 'Rank', 'Unique Quotations'])
    for rank, (qid, uq) in sorted_counts:
        df.iloc[rank] = [qid, rank + 1, uq]  # Start rank at 1
    total = df['Unique Quotations'].sum()
    print('TOTAL:', total)
    df['% of Total'] = (100 * df['Unique Quotations'] / total).map(lambda r: f'{r:.5f}')
    with open(args.save, 'w') as dump:
        df.to_csv(dump, index=False, header=True)


if __name__ == '__main__':
    main()
