from argparse import ArgumentParser
from datetime import date
import json
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as f
import sys
import time

from utils import pyspark_udfs as udf_util


parser = ArgumentParser()
parser.add_argument('--quotations', help='Link to Quotebank in quotation centric format - stored as parquet.', required=True)
parser.add_argument('--occupations', help='JSON file, mapping Wikidata QIDs to (political) occupations', required=True)
parser.add_argument('--speaker', help='A Wikidata Dump containing speaker attributes - stored as parquet.', required=True)
parser.add_argument('--save', help='Save political data here.', required=True)
parser.add_argument('--log', help='Path to a logfile', required=False)


def _trimDuplicates(df: DataFrame, useArraySizeForOccurrences: bool = False) -> DataFrame:
    """
    A function to find and discard quotations matching the following criteria:
    - They have 10 or more total occurrences.
    - More than 50% of their occurrences belong to a single root domain.
    Parameters
    ----------
    df: The quotation data frame. It has to have a column called 'domains', containing root domains of the occurrences.
    useArraySizeForOccurrences: If set to true, the total number of occurrences will be set to the length of the domains
    array of the quotation. This SHOULD be the same, but it is not in ~1-2% of cases.
    Returns
    -------
    A list of the quotations that are "duplicated" and that should be dropped in a filtering step.
    """
    cv = CountVectorizer(inputCol='domains', outputCol='domainCount')
    model = cv.fit(df)
    df = model.transform(df)
    if useArraySizeForOccurrences:
        df = df.drop('numOccurrences').withColumn('numOccurrences', udf_util.arraySize('domains'))

    duplicates = df \
        .filter('numOccurrences > 10') \
        .select('quoteID', (udf_util.arrayMax('domainCount') / f.col('numOccurrences')).alias('maxFraction')) \
        .filter(f.col('maxFraction') > 0.5)

    return df.join(duplicates, on='quoteID', how='left_anti')


def preprocess(df: DataFrame) -> DataFrame:
    """
    Trims down URLs to root domains.
    Adds a 'year' and a 'month' column for convenience.
    """
    df = df.withColumn('domains', udf_util.URLs2domain(f.col('urls')))
    # TODO: Year and Month
    return df


def sanitize(df: DataFrame) -> DataFrame:
    """
    Removes spurious quotations from the Dataframe, such as:
    - Dates (Quotebank seems to contain a couple ones, covered as quotation)
    - Duplicates (Mostly from sited that are queried over and over again)
    """
    no_dates = df.filter(~udf_util.isDate(f.col('quotation')))
    no_duplicates = _trimDuplicates(df)


def getQuotesFromSpeakers(df: DataFrame, speakers: DataFrame) -> DataFrame:
    """
    Narrows down the quotation dataframe to quotations uttered by one of the speakers in the speaker Dataframe.
    To achieve this, first
    """
    # TODO
    return df


def main():
    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    t = time.time()
    political_occupations = json.load(open(args.occupations, 'r'))
    jobIDs = list(political_occupations.keys())
    df = spark.read.parquet(args.quotations)
    speaker = spark.read.parquet(args.speaker)

    df = preprocess(df)
    initial_count = df.count()
    df = sanitize(df)
    sanitized_count = df.count()
    t1 = time.time()
    print(f'Time for preprocessing and sanitizing: {t1-t:.2f}s.')
    print(f'Went from {initial_count} quotations down to {sanitized_count} quotations.')

    politicians = speaker.filter(f.col('occupation').isin(jobIDs))
    df = getQuotesFromSpeakers(df, politicians)

    df.repartition('year', 'month').write.mode('overwrite').parquet(args.save)
    if args.log is not None:
        unique_quotations = df.select('quoteID').distinct().count()
        attributed_quotations = df.count()
        num_politicians = df.select('qid').distinct().count()
        with open(args.log, 'w') as logfile:
            logfile.write('Quotation Sanitation and Extraction of Politicians.\n')
            logfile.write('Date: ' + str(date.today()) + '\n')
            logfile.write('Initial Number of Quotations: ' + str(initial_count) + '\n')
            logfile.write('Final Unique Number of Quotations: ' + str(unique_quotations) + '\n')
            logfile.write('Final Attributed Number of Quotations: ' + str(attributed_quotations) + '\n')
            logfile.write('Final Number of Politicians: ' + str(num_politicians) + '\n')
            logfile.write(f'Time For Sanitation: {t1-t:.2f}s\n')
            logfile.write(f'Time Total: {time.time()-t:.2f}s\n')
            logfile.write('Output written to: ' + args.save + '\n')
            logfile.write('Schema of output:\n\n')
            __stdout = sys.stdout
            try:
                sys.stdout = logfile
                df.printSchema()
            finally:
                sys.stdout = __stdout

    print(f'Over and Out. Time: {time.time() - t:.2f}s')


if __name__ == '__main__':
    main()
