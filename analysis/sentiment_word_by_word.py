import argparse
import re
from datetime import datetime
import json
import pickle
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as f
from pyspark.ml.feature import RegexTokenizer, Tokenizer
import time
from sentiment import count
import sys
from typing import Dict, List


parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to configuration file to select empath and LIWC categories. (.json)', required=True)
parser.add_argument('--quotations', help='Path To the quotation file.', required=True)
parser.add_argument('--liwc', help='Path to your local LIWC copy, as .pickle', required=True)
parser.add_argument('--save', help='Path where to store the resulting data.', required=True)
parser.add_argument('--log', help='Path where the write a logfile to.', required=False)
parser.add_argument('--content_column', help='The strings to analyse', default='quotation')


def loadLIWCregex(fpath: str, categories: List[str]) -> Dict[str, str]:
    """
    Uses the LIWC regex dump and maps each regex to itself.
    Parameters
    ----------
    fpath: Path to the LIWC dump (.pickle)
    categories: A selection of LIWC categories of interest.
    Returns
    -------
    A mapping of liwc_category (yes, including the liwc_ prefix) -> regex
    """
    with open(fpath, 'rb') as lw:
        d = pickle.load(lw)

    # Translate the patterns as given to a format that works with the spark RegexTokenizer
    def transform_liwc(pat: str) -> str:
        no_spaces = pat.strip()  # Removes sometimes occurring trailing spaces
        non_greedy = re.sub('\\.\\*', '.*?', no_spaces)  # Translates greedy to non-greedy match
        standalone = '\\b' + non_greedy + '\\b'  # \b matches word beginnings and/or ends
        return standalone
    ret = {'liwc_' + key + '_' + pat: pat
           for key in categories
           for pat in [transform_liwc(w) for w in d[key].pattern[2:-2].lower().split('|')]}
    return ret


def main():
    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel('WARN')
    t0 = time.time()

    config = json.load(open(args.config, 'r'))
    quotes = spark.read.parquet(args.quotations)
    quotes = quotes.withColumnRenamed(args.content_column, 'ANALYSIS_CONTENT').filter(~f.col('ANALYSIS_CONTENT').isNull())

    patterns = loadLIWCregex(args.liwc, config['liwc_categories'])
    keepColumns = quotes.columns
    assert ('year' in keepColumns) and ('month' in keepColumns), 'If year or month is missing, how to repartition?'

    # The line that matters:
    df = count(quotes.select(keepColumns), patterns)
    df = df.withColumnRenamed('ANALYSIS_CONTENT', args.content_column)
    df.repartition('year', 'month').write.mode('overwrite').partitionBy('year', 'month').parquet(args.save)

    if args.log is not None:
        with open(args.log, 'w') as log:
            log.write('Log for {}\n'.format(datetime.today()))
            log.write('Wrote to {}\n'.format(args.save))
            log.write('Extracted for {} Patterns. Took {:.2f}s.\n'.format(len(patterns), time.time() - t0))

            __stdout = sys.stdout
            try:
                sys.stdout = log
                df.printSchema()
                print('\n\n')
            finally:
                sys.stdout = __stdout
            for cat, words in patterns.items():
                log.write('Pattern for {}: '.format(cat))
                log.write(words + '\n\n')


if __name__ == '__main__':
    main()
