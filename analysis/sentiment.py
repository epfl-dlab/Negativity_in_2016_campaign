import argparse
import re
from datetime import datetime
from empath import Empath
import json
import pickle
from pyspark.sql import DataFrame, SparkSession
import pyspark.sql.functions as f
from pyspark.ml.feature import RegexTokenizer, Tokenizer
import time
import sys
from typing import Dict, List

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to configuration file to select empath and LIWC categories. (.json)', required=True)
parser.add_argument('--quotations', help='Path To the quotation file.', required=True)
parser.add_argument('--liwc', help='Path to your local LIWC copy, as .pickle', required=True)
parser.add_argument('--save', help='Path where to store the resulting data.', required=True)
parser.add_argument('--log', help='Path where the write a logfile to.', required=False)
parser.add_argument('--content_column', help='The strings to analyse', default='quotation')


def loadEmpathVocab(categories: List[str]) -> Dict[str, str]:
    """
    Uses a selection of empath categories and maps them to according regular expressions.
    Parameters
    ----------
    categories: A selection of empath categories of interest.
    Returns
    -------
    A mapping of empath_category (yes, including the empath_ prefix) -> regex
    """
    lexicon = Empath()
    vocabulary = {}
    patterns = {}

    for cat in categories:
        vocabulary[cat] = lexicon.cats[cat]

    for cat, words in vocabulary.items():
        # Regex pattern (multiple optional words joint by '|')
        patterns['empath_' + cat] = '\\b' + '\\b|\\b'.join(words) + '\\b'

    print('Words per empath category:')
    print('\n'.join([f'{cat}: {len(words.lower().split("|"))}' for cat, words in patterns.items()]))
    return patterns


def loadLIWCVocab(fpath: str, categories: List[str]) -> Dict[str, str]:
    """
    Uses the LIWC regex dump and maps the selected categories to regular expressions that work with pyspark.
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
    ret = {'liwc_' + key: '|'.join([transform_liwc(w) for w in d[key].pattern[2:-2].lower().split('|')]) + '\\b' for key in categories}
    print('Loaded LIWC patterns for categories', ', '.join(ret.keys()))
    return ret


def __mergeDictionaries(listOfDicts: List[Dict]) -> Dict:
    ret = {}
    for d in listOfDicts:
        for key, val in d.items():
            if key in ret:
                raise ValueError('Duplicate dictionary keys!')
            else:
                ret[key] = val
    return ret


def count(df: DataFrame, regexPattern: Dict[str, str]) -> DataFrame:
    """
    Takes a dataframe with strings to analyze (ANALYSIS_CONTENT) and counts the number of matches for each of the
    regexPattern keys. Casing is ignored.
    """
    originalColumns = df.columns
    newColumns = []

    for kw, pat in regexPattern.items():
        if kw in originalColumns:
            print('{} already present. Will not be counted again'.format(kw))
            continue
        print('Fitting tokenizer for {}'.format(kw))
        tokenizer = RegexTokenizer(inputCol='ANALYSIS_CONTENT', outputCol=kw + '_tokens', gaps=False, toLowercase=True,
                                   pattern=pat)
        df = tokenizer.transform(df)
        newColumns.append(kw)

    if 'numTokens' not in originalColumns:
        print('And also one tokenizer for the number of tokens.')
        tokenizer = Tokenizer(inputCol='ANALYSIS_CONTENT', outputCol='numTokens')
        df = tokenizer.transform(df)
        newColumns.append('numTokens')

    sizeFunctions = [f.size(col + '_tokens').alias(col) for col in newColumns]
    df = df.select(
        *originalColumns,
        *sizeFunctions
    )
    print('Finished preparations, starting writing and wrap-up.')
    return df


def main():
    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel('WARN')
    t0 = time.time()

    config = json.load(open(args.config, 'r'))
    quotes = spark.read.parquet(args.quotations)
    quotes = quotes.withColumnRenamed(args.content_column, 'ANALYSIS_CONTENT').filter(~f.col('ANALYSIS_CONTENT').isNull())

    liwcPatterns = loadLIWCVocab(args.liwc, config['liwc_categories'])
    empathPatterns = loadEmpathVocab(config['empath_categories'])
    allPatterns = __mergeDictionaries([liwcPatterns, empathPatterns])
    keepColumns = quotes.columns
    assert ('year' in keepColumns) and ('month' in keepColumns), 'If year or month is missing, how to repartition?'

    # The line that matters:
    df = count(quotes.select(keepColumns), allPatterns)
    df = df.withColumnRenamed('ANALYSIS_CONTENT', args.content_column)
    df.repartition('year', 'month').write.mode('overwrite').partitionBy('year', 'month').parquet(args.saveAs)

    if args.logfile is not None:
        with open(args.logfile, 'w') as log:
            log.write('Log for {}\n'.format(datetime.today()))
            log.write('Wrote to {}\n'.format(args.saveAs))
            log.write('Extracted for {} Patterns. Took {:.2f}s.\n'.format(len(allPatterns), time.time() - t0))

            __stdout = sys.stdout
            try:
                sys.stdout = log
                df.printSchema()
                print('\n\n')
            finally:
                sys.stdout = __stdout
            for cat, words in allPatterns.items():
                log.write('Pattern for {}: '.format(cat))
                log.write(words + '\n\n')


if __name__ == '__main__':
    main()
