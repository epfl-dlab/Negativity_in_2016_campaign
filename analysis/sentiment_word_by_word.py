import argparse
import re
from datetime import datetime
import json
from pathlib import Path
import pickle
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
import pyspark.sql.functions as f
from pyspark.ml.linalg import DenseVector, Vectors, VectorUDT
import sys
import time
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from analysis.RDD import KINK
from analysis.aggregate import _prep_people

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to configuration file to select empath and LIWC categories. (.json)', required=True)
parser.add_argument('--quotations', help='Path To the quotation file.', required=True)
parser.add_argument('--people', help='Path to a parquet file containing people background info', required=True)
parser.add_argument('--liwc', help='Path to your local LIWC copy, as .pickle', required=True)
parser.add_argument('--save', help='Path where to store the resulting data.', required=True)
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


def make_liwc_count_udf(liwc: Dict) -> callable:
    words, patterns = zip(*liwc.items())
    indices = list(range(len(words)))

    @f.udf(VectorUDT())
    def count(s: StringType()) -> DenseVector:
        return Vectors.dense([len(re.findall(ptrn, s)) for ptrn in patterns])

    return count, dict(zip(indices, words))


def main():
    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel('WARN')
    t0 = time.time()

    config = json.load(open(args.config, 'r'))
    quotes = spark.read.parquet(args.quotations)
    quotes = quotes.withColumnRenamed(args.content_column, 'ANALYSIS_CONTENT').filter(~f.col('ANALYSIS_CONTENT').isNull())
    people = _prep_people(spark.read.parquet(args.people)).cache()
    quotes = quotes.join(people, on='qid')  # Necessary to drop quotes from people that are not analyzed (e.g. ambiguous party affiliation)

    patterns = loadLIWCregex(args.liwc, config['liwc_categories'])
    keepColumns = quotes.columns
    assert ('year' in keepColumns) and ('month' in keepColumns), 'If year or month is missing, how to repartition?'

    # The line that matters:
    udf, mapping = make_liwc_count_udf(patterns)
    df = quotes \
        .withColumn('counts', udf(f.col('ANALYSIS_CONTENT'))) \
        .withColumn('after_threshold', f.col('date') > KINK)
    before = df.filter(~f.col('after_threshold')).select('counts').rdd.fold([0] * len(mapping), lambda a, b: [x + y for x, y in zip(a, b)])
    after = df.filter(f.col('after_threshold')).select('counts').rdd.fold([0] * len(mapping), lambda a, b: [x + y for x, y in zip(a, b)])
    with open(args.save, 'wb') as savefile:
        ret = {
            'counts_before': before,
            'counts_after': after,
            'mapping': mapping
        }
        pickle.dump(ret, savefile)


if __name__ == '__main__':
    main()
