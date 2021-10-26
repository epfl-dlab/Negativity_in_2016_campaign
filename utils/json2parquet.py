from argparse import ArgumentParser
from pyspark.sql import SparkSession
from pyspark.sql.functions import date_format

parser = ArgumentParser()
parser.add_argument('--json', help='JSON file location.', required=True)
parser.add_argument('--parquet', help='Parquet file location.', required=True)


def main():
    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    from_file = args.json
    to_file = args.parquet

    df = spark.read.json(from_file)
    columns = df.columns
    if 'date' in columns:
        # Add a Year, Month and standardized date column
        all_but_date = [c for c in columns if c != 'date']
        df = df \
            .select(*all_but_date,
                    date_format('date', 'yyyy-MM-dd').alias('date'),
                    date_format('date', 'yyyy').alias('year'),
                    date_format('date', 'MM').alias('month')
                    )

        df.repartition('year', 'month').write.mode('overwrite').partitionBy('year', 'month').parquet(to_file)
    else:
        df.write.mode('overwrite').parquet(to_file)


if __name__ == '__main__':
    main()
