import argparse
import pickle
from datetime import datetime
import itertools
import math
import numpy as np
import pandas as pd
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession, Window
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType
import sys
from typing import Dict, List, Union
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from preparations.getPolitics import DEMOCRATIC_PARTY, MALE, FEMALE, REPUBLICAN_PARTY

MISSING_MONTHS = ['2010-5', '2010-6', '2016-1', '2016-3', '2016-6', '2016-10', '2016-11', '2017-1']
OBAMA_ELECTION = datetime(2008, 11, 4)
TRUMP_ELECTION = datetime(2016, 11, 8)

parser = argparse.ArgumentParser()
parser.add_argument('--sentiment', help='Quotations including sentiment counts',  required=True)
parser.add_argument('--save', help='FOLDER to save data to.',  required=True)
parser.add_argument('--people', help='Path to people / politicians dataframe.', required=True)


def _prep_people(df: DataFrame) -> DataFrame:
    """
    Preprocessing of the speaker / people Dataframe. Maps categories to binary variables.
    Edge Case: People who are listed having 2 genders or being members of 2 parties will be chosen for both.
    """
    party_map = {REPUBLICAN_PARTY: 0, DEMOCRATIC_PARTY: 1}
    gender_map = {MALE: 0, FEMALE: 1}
    __map_party = f.udf(lambda x: party_map.get(x, None), IntegerType())
    __map_gender = f.udf(lambda x: gender_map.get(x, None), IntegerType())

    return df \
        .withColumn('congress_member', (f.size('CIDs') > 0).cast('integer')) \
        .select('qid', 'congress_member', 'genders', f.explode('parties').alias('tmp_party')) \
        .select('*', f.explode('genders').alias('tmp_gender')) \
        .select('qid', 'congress_member', __map_party('tmp_party').alias('party'), __map_gender('tmp_gender').alias('gender')) \
        .dropna(how='any', subset=['gender', 'party'])


def _add_governing_column(df: pd.DataFrame) -> pd.DataFrame:
    """Uses the date and the party column to add a 'governing_party' indicator column."""
    df['governing_party'] = df.apply(lambda r: int(
        (r.date <= OBAMA_ELECTION) and (r.party == 0) or
        ((r.date > OBAMA_ELECTION) and (r.date <= TRUMP_ELECTION)) and (r.party == 1) or
        (r.date > TRUMP_ELECTION) and (r.party == 0)
    ), axis=1)
    return df


def _df_postprocessing(df: pd.DataFrame, features: List[str], MEAN: pd.DataFrame, STD: pd.DataFrame) -> pd.DataFrame:
    """Standardizes the dataframe and adds utility columns."""
    df = df.sort_index().reset_index().rename(columns={'index': 'date'})
    start = min(df.date)
    df['time_delta'] = df.apply(lambda r: int((r.date.year - start.year) * 12 + r.date.month - start.month), axis=1)

    df[features] = (df[features] - MEAN) / STD
    return df


def _make_date(s: str) -> datetime:
    """
    Takes a string as used for grouped score keys yyyy-mm-att1-att2-... and returns the datetime for year and month.
    """
    year, month = s.split('-')[:2]
    return datetime(int(year), int(month), 15)


def _score_dict_to_pandas(d: Dict, keys: List[Union[datetime, str]], columns: List[str]) -> pd.DataFrame:
    for_pandas = {dt: [d[dt][c] for c in columns] for dt in keys}
    return pd.DataFrame.from_dict(for_pandas, orient='index', columns=columns)


def getScores(df: DataFrame) -> pd.DataFrame:
    """
    Groups scores by year and month and returns quotation-average scores for sentiment features.
    Parameters
    ----------
    df: Spark Dataframe containing sentiment counts.
    """
    scores = {}
    columns = [c for c in df.columns if '_' in c]
    iterbar = tqdm(columns)

    for col in iterbar:
        iterbar.set_description(col)
        iterbar.update()
        counts = df.groupby(['year', 'month', col, 'numTokens']) \
            .agg(f.count('*').alias('cnt')) \
            .withColumn('weighted_itf', f.col('cnt') * f.col(col) / f.col('numTokens')) \
            .withColumn('yyyy-mm', f.concat(f.col('year'), f.lit('-'), f.col('month'))) \
            .filter(~f.col('yyyy-mm').isin(MISSING_MONTHS)) \
            .drop('yyyy-mm') \
            .groupby(['year', 'month']) \
            .agg(f.sum('cnt').alias('total_cnt'), f.sum('weighted_score').alias('summed_weighted_score')) \
            .withColumn('score', f.col('summed_weighted_score') / f.col('total_cnt')) \
            .rdd \
            .map(lambda r: (r.year, r.month, r.score)).collect()

        for year, month, score in itertools.chain(counts):
            date = datetime(year, month, 15)
            if date not in scores:
                scores[date] = {}

            scores[date][col] = score

    return _score_dict_to_pandas(scores, list(scores.keys()), columns)


def getScoresByGroups(df: DataFrame, groupby: List[str]) -> pd.DataFrame:
    """
    Aggregates by the year, month and given groups and returns a dataframe per binary group.
    Parameters
    ----------
    df: Spark dataframe containing counts
    groupby: Binary Variables to group by.
    """
    scores = {}
    columns = [c for c in df.columns if '_' in c]
    iterbar = tqdm(columns)
    groupby = ['year', 'month'] + groupby

    for col in iterbar:
        iterbar.set_description(col)
        iterbar.update()
        counts = df.groupby([*groupby, col, 'numTokens']) \
            .agg(f.count('*').alias('cnt')) \
            .withColumn('weighted_score', f.col('cnt') * f.col(col) / f.col('numTokens')) \
            .withColumn('yyyy-mm', f.concat(f.col('year'), f.lit('-'), f.col('month'))) \
            .filter(~f.col('yyyy-mm').isin(MISSING_MONTHS)) \
            .drop('yyyy-mm') \
            .groupby(list(groupby)) \
            .agg(f.sum('cnt').alias('total_cnt'), f.sum('weighted_score').alias('summed_weighted_scores')) \
            .rdd \
            .map(lambda r: (*[r[g] for g in groupby], r['summed_weighted_scores'], r['total_cnt'])).collect()

        for elements in itertools.chain(counts):
            elements = list(elements)
            cnt = elements.pop(-1)
            sws = elements.pop(-1)
            key = '-'.join(map(str, elements))
            if key not in scores:
                scores[key] = {}

            scores[key][col] = sws / cnt

    keys = list(scores.keys())
    # combinations = ['-'.join(comb) for comb in list(itertools.product(map(str, range(2)), repeat=len(groupby)))]
    df = _score_dict_to_pandas(scores, keys, columns)
    df[groupby] = df.index.map(lambda x: list(map(int, x.split('-')))).to_list()
    df.index = df.index.map(_make_date)

    return df


def getScoresBySpeaker(df: DataFrame) -> pd.DataFrame:
    """
    Lexicographic feature scores, macro-average over speakers.
    Parameters
    ----------
    df: Spark dataframe containing counts
    """
    scores = {}
    columns = [c for c in df.columns if '_' in c]
    iterbar = tqdm(columns)

    for col in iterbar:
        iterbar.set_description(col)
        iterbar.update()
        counts = df.groupby(['year', 'month', 'qid', col, 'numTokens']) \
            .agg(f.count('*').alias('cnt')) \
            .withColumn('weighted_score', f.col('cnt') * f.col(col) / f.col('numTokens')) \
            .withColumn('yyyy-mm', f.concat(f.col('year'), f.lit('-'), f.col('month'))) \
            .filter(~f.col('yyyy-mm').isin(MISSING_MONTHS)) \
            .drop('yyyy-mm') \
            .groupby(['year', 'month', 'qid']) \
            .agg(f.sum('cnt').alias('total_cnt'), f.sum('weighted_score').alias('summed_weighted_score')) \
            .rdd \
            .map(lambda r: (r['year'], r['month'], r['qid'], r['summed_weighted_score'], r['total_cnt'])).collect()

        for year, month, qid, sws, cnt in itertools.chain(counts):
            date = datetime(year, month, 15)
            if date not in scores:
                scores[date] = {}

            if col not in scores[date]:
                scores[date][col] = []

            scores[date][col].append(sws / cnt)

    for date in scores:
        for col in scores[date]:
            scores[date][col] = np.mean(scores[date][col])

    return _score_dict_to_pandas(scores, list(scores.keys()), columns)


def getScoresByVerbosity(df, splits: int = 4) -> pd.DataFrame:
    """
    Speaker-Averaged Scores, but aggregated by verbosity
    Parameters
    ----------
    df: Spark dataframe containing counts
    splits: ...
    """
    # Warnings about the window moving all things to a single partition can be ignored
    # The performance is not that bad actually
    w = Window().orderBy(f.desc('numQuotes'))
    num_speakers = df.select('qid').distinct().count()
    interval_size = math.ceil(num_speakers / splits)
    # Floor division row number by interval size gives 0, 1, ..., numIntervals
    speaker_map = df \
        .groupby('qid') \
        .agg(f.count('qid').alias('numQuotes')) \
        .sort(f.desc('numQuotes')) \
        .withColumn('verbosity', f.floor(f.row_number().over(w) / f.lit(interval_size)))

    df = df.join(speaker_map, on='qid')

    scores = {}
    columns = [c for c in df.columns if '_' in c]
    iterbar = tqdm(columns)

    for col in iterbar:
        iterbar.set_description(col)
        iterbar.update()
        counts = df.groupby(['year', 'month', 'qid', col, 'numTokens']) \
            .agg(f.count('*').alias('cnt'), f.first('verbosity').alias('verbosity')) \
            .withColumn('weighted_score', f.col('cnt') * f.col(col) / f.col('numTokens')) \
            .withColumn('yyyy-mm', f.concat(f.col('year'), f.lit('-'), f.col('month'))) \
            .filter(~f.col('yyyy-mm').isin(MISSING_MONTHS)) \
            .drop('yyyy-mm') \
            .groupby('year', 'month', 'qid') \
            .agg((f.sum('weighted_score') / f.sum('cnt')).alias('speaker_score'),
                 f.sum('cnt').alias('speaker_cnt'),
                 f.first('verbosity').alias('verbosity')) \
            .groupby('year', 'month', 'verbosity') \
            .agg((f.sum('speaker_score') / f.count('qid')).alias('group_score'),
                 f.sum('speaker_cnt').alias('quotes_in_group')) \
            .rdd \
            .map(lambda r: (r['year'], r['month'], r['verbosity'], r['group_score'], r['quotes_in_group'])).collect()

        sample_sizes = dict()

        for elements in itertools.chain(counts):
            elements = list(elements)
            cnt = elements.pop(-1)
            score = elements.pop(-1)
            key = '-'.join(map(str, elements))
            if key not in scores:
                scores[key] = {}
                sample_sizes[key] = {}

            scores[key][col] = score

    df = _score_dict_to_pandas(scores, list(scores.keys()), columns)
    df[['year', 'month', 'verbosity']] = df.index.map(lambda x: list(map(int, x.split('-')))).to_list()
    df.index = df.index.map(_make_date)

    return df


def main():
    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    base = Path(args.save)
    base.mkdir(exist_ok=True)

    df = spark.read.parquet(args.sentiment)
    features = [c for c in df.columns if '_' in c]
    people = _prep_people(spark.read.parquet(args.people)).cache()
    df = df.join(people, on='qid')

    def save(_df: pd.DataFrame, name: str):
        print(name)
        print(df.head())
        _df.to_csv(base.joinpath(name + '.csv').open('w'))

    # Basic Quotation Aggregation
    agg = getScoresByGroups(df, [])
    # The same standardization values will be used for all
    MEAN = agg[features].mean()
    STD = agg[features].std()
    pickle.dump(MEAN, base.joinpath('mean.parquet').open('wb'))
    pickle.dump(STD, base.joinpath('std.parquet').open('wb'))
    save(_df_postprocessing(agg, features, MEAN, STD), 'QuotationAggregation')

    agg = getScoresByGroups(df, ['gender', 'party', 'congress_member'])
    agg = _add_governing_column(agg)
    save(_df_postprocessing(agg, features, MEAN, STD), 'AttributesAggregation')

    agg = getScoresByGroups(df, ['party'])
    save(_df_postprocessing(agg, features, MEAN, STD), 'PartyAggregation')

    agg = getScoresBySpeaker(df)
    save(_df_postprocessing(agg, features, MEAN, STD), 'SpeakerAggregation')

    spark.sparkContext.setLogLevel('ERROR')  # Window Warning is Expected and can be ignored
    save(_df_postprocessing(agg, features, MEAN, STD), 'VerbosityAggregation')


if __name__ == '__main__':
    main()
