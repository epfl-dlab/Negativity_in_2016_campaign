import argparse
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

parser = argparse.ArgumentParser()
parser.add_argument('--quotations', help='Dataframe (.parquet) containing quotations and a QID column', required=True)
parser.add_argument('--save_count', help='File to store the counts in. (.csv)', required=True)
parser.add_argument('--save_attribution', help='File to store the attribution counts in. (.txt)', required=True)


def main():
    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    df = spark.read.parquet(args.quotations)
    counts = df \
        .withColumn('year-month', f.concat(df.year, df.month)) \
        .groupby('qid') \
        .agg(f.countDistinct('quoteID').alias('cnt'),
             f.countDistinct('year-month').alias('appears_in_months')) \
        .rdd \
        .map(lambda r: (r.qid, r.cnt, r.appears_in_months)) \
        .collect()

    # Also, give som information about multiply-assigned quotations
    mutliples = df \
        .groupby('quoteID') \
        .agg(f.collect_set('qid').alias('speaker')) \
        .withColumn('numAttributed', f.size('speaker')) \
        .groupby('numAttributed') \
        .agg(f.count('*').alias('total')) \
        .sort('total') \
        .rdd.map(lambda r: (r.numAttributed, r.total)).collect()

    sorted_counts = list(zip(range(len(counts)), sorted(counts, key=lambda x: x[1], reverse=True)))
    df = pd.DataFrame(data=None, index=range(len(sorted_counts)),
                      columns=['QID', 'Rank', 'Unique Quotations', 'Appears in Months'])
    for rank, (qid, uq, aim) in sorted_counts:
        df.iloc[rank] = [qid, rank + 1, uq, aim]  # Start rank at 1
    total = df['Unique Quotations'].sum()
    print('TOTAL:', total)
    df['% of Total'] = (100 * df['Unique Quotations'] / total).map(lambda r: f'{r:.5f}')
    with open(args.save_count, 'w') as dump:
        df.to_csv(dump, index=False, header=True)

    attribution_txt = 'Number Attributed: Total'
    for (na, tot) in mutliples:
        attribution_txt += '{}: {}\n'.format(na, tot)
    with open(args.save_attribution, 'w') as dump:
        dump.write(attribution_txt)


if __name__ == '__main__':
    main()
