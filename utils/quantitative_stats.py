from argparse import ArgumentParser
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

parser = ArgumentParser()
parser.add_argument('--quotations', help='Path to quotations DF', required=True)
parser.add_argument('--save', help='Path where to save Dataframe (csv)', required=True)


def main():
    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    df = spark.read.parquet(args.quotations)
    stats = df \
        .withColumn('numDomains', f.size('domains')) \
        .groupby(['year', 'month']) \
        .agg(f.countDistinct('qid').alias('num_speaker'),
             f.countDistinct('quoteID').alias('num_quotes'),
             f.sum('numDomains').alias('total_domains'),
             ) \
        .sort(['year', 'month']) \
        .toPandas()

    stats.to_csv(open(args.save, 'w'))


if __name__ == '__main__':
    main()
