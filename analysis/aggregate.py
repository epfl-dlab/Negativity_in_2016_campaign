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
from pyspark.sql.types import ArrayType, IntegerType, StringType
import sys
from typing import Dict, List, Union
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from preparations.getPolitics import DEMOCRATIC_PARTY, MALE, FEMALE, REPUBLICAN_PARTY
from analysis.RDD import KINK

MISSING_MONTHS = ['2010-5', '2010-6', '2016-1', '2016-3', '2016-6', '2016-10', '2016-11', '2017-1']
OBAMA_ELECTION = datetime(2008, 11, 4)
TRUMP_ELECTION = datetime(2016, 11, 8)

parser = argparse.ArgumentParser()
parser.add_argument('--sentiment', help='Quotations including sentiment counts',  required=True)
parser.add_argument('--save', help='FOLDER to save data to.',  required=True)
parser.add_argument('--people', help='Path to people / politicians dataframe.', required=True)
parser.add_argument('--individuals', help='If provided, will get aggregates for these individuals as well.'
                                          'Takes any number of QIDs', nargs='+', default=[])
parser.add_argument('--exclude_top_n', help='If given, will perform an aggregation for each of the top n speakers'
                                            'excluding that speaker from the data. This enables to analyse the influence'
                                            'of the indivdual on the score', type=int, required=False)
parser.add_argument('--extract_top_n', help='Same as "individuals" argument, but takes the n most verbose individuals'
                                            'instead of specifying every one.', type=int)
parser.add_argument('--top_n_file', help='Required for exclude_top_n: CSV file including rank and qid.')
parser.add_argument('--standardize', default=True)


MANUAL_PARTY_MEMBERSHIP = {
    'Q22686': REPUBLICAN_PARTY,  # Donald Trump, was republican president
    'Q6294': DEMOCRATIC_PARTY,  # Hillary Clinton, was democratic presidential candidate
    'Q215057': REPUBLICAN_PARTY,  # Rick Perry, Dem->Rep in 1989
    'Q607': DEMOCRATIC_PARTY,  # Michael Bloomberg, Rep->Dem in 2007
    'Q160582': REPUBLICAN_PARTY,  # Michele Bachmann, left democrats 1978
    'Q244631': DEMOCRATIC_PARTY,  # Leon Panetta, high political democratic offices from 1994 on
    'Q981167': REPUBLICAN_PARTY,  # Chris Smith, Dem->Rep 1978
    'Q816459': REPUBLICAN_PARTY,  # Ben Carson, left democrats 1981, joined republicans 2014, candydacy for primaries 2015
    # Dropping Charlie Christ, who changed party affiliation 2012
    'Q6250211': REPUBLICAN_PARTY,  # John Neely Kennedy, Dem->Rep 2007
    'Q6834862': DEMOCRATIC_PARTY,  # Michael Thompson, Found no source for his republican membership, but was candidate as Democrat for many elections
    'Q1680235': REPUBLICAN_PARTY,  # James D. Martin, left Dem 1962
    'Q6174997': DEMOCRATIC_PARTY,  # Jeff Smith, Wikipedia lists him only as Democrat - was candidate for house of representatives as democrat in 2004
    'Q179732':  REPUBLICAN_PARTY,  # Nathan Deal, Dem -> Rep 1995
    'Q525362': REPUBLICAN_PARTY,  # Sonny Perdue, Dem -> Rep 1997
    'Q1683881': REPUBLICAN_PARTY,  # Jason Chaffetz, Dem -> Rep 1990
    # Dropping Wilbur Ross (Q8000233) who changed party affiliation in 2016
    # Drop Patrick Murphy (Q3182011), Rep -> Dem 2011 [but held every noteworthy position he had as a Democrat (from 2012 on)]
    # Drop Tom Smith (Q7817616), Dem -> Rep 2011
    # Drop Arlen Specter (Q363055), Rep -> Dem 2009, died 2012
    'Q256334': REPUBLICAN_PARTY,  # Susana Martinez, Dem -> Rep 1995
    'Q472254': REPUBLICAN_PARTY,  # Richard Shelby, Dem -> Rep 1994
}


def _prep_people(df: DataFrame) -> DataFrame:
    """
    Preprocessing of the speaker / people Dataframe. Maps categories to binary variables.
    Edge Case: People who are listed having 2 genders or being members of 2 parties will be chosen for both.
    Edge Case: People who are listed having 2 genders or being members of 2 parties will be chosen for both.
    """
    party_map = {REPUBLICAN_PARTY: 0, DEMOCRATIC_PARTY: 1}
    gender_map = {MALE: 0, FEMALE: 1}
    __map_party = f.udf(lambda x: party_map.get(x, None), IntegerType())
    __map_gender = f.udf(lambda x: gender_map.get(x, None), IntegerType())
    __manual_party = f.udf(lambda x: MANUAL_PARTY_MEMBERSHIP[x], StringType())

    allPeople = df.withColumn('congress_member', (f.size('CIDs') > 0).cast('integer'))
    manualAssigned = list(MANUAL_PARTY_MEMBERSHIP.keys())
    manual = allPeople \
        .filter((f.size('parties') > 1) & (f.col('qid').isin(manualAssigned))) \
        .withColumn('tmp_party', __manual_party(f.col('qid'))) \
        .drop('parties')

    ret = allPeople \
        .filter((f.size('parties') == 1) & (f.size('genders') == 1)) \
        .withColumn('tmp_party', f.explode('parties')) \
        .drop('parties') \
        .union(manual) \
        .select('qid', 'congress_member', 'genders', 'tmp_party') \
        .select('*', f.explode('genders').alias('tmp_gender')) \
        .select('qid', 'congress_member', __map_party('tmp_party').alias('party'), __map_gender('tmp_gender').alias('gender')) \
        .dropna(how='any', subset=['gender', 'party'])

    return ret


def _add_governing_column(df: pd.DataFrame) -> pd.DataFrame:
    """Uses the date and the party column to add a 'governing_party' indicator column."""
    if 'date' not in df.columns:
        tmp = df.copy(deep=True)
        tmp['date'] = df.index
        df['governing_party'] = tmp.apply(lambda r: int(
            (r.date <= OBAMA_ELECTION) and (r.party == 0) or
            ((r.date > OBAMA_ELECTION) and (r.date <= TRUMP_ELECTION)) and (r.party == 1) or
            (r.date > TRUMP_ELECTION) and (r.party == 0)
        ), axis=1)
    else:
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

    # Limit what will be returned, excludes additional information like e.g. leftovers from speaker names or URLs
    keep = features + ['date', 'time_delta', 'gender', 'party', 'governing_party', 'congress_member', 'year', 'month', 'verbosity']
    keep = [col for col in keep if col in df.columns]

    return df[keep]


def _make_date(s: str) -> datetime:
    """
    Takes a string as used for grouped score keys yyyy-mm-att1-att2-... and returns the datetime for year and month.
    """
    year, month = s.split('-')[:2]
    return datetime(int(year), int(month), 15)


def _score_dict_to_pandas(d: Dict, keys: List[Union[datetime, str]], columns: List[str]) -> pd.DataFrame:
    for_pandas = {dt: [d[dt][c] for c in columns] for dt in keys}
    return pd.DataFrame.from_dict(for_pandas, orient='index', columns=columns)


def getScoresByGroups(df: DataFrame, groupby: List[str]) -> pd.DataFrame:
    """
    Aggregates by the year, month and given groups and returns a dataframe per binary group.
    Parameters
    ----------
    df: Spark dataframe containing counts
    groupby: Binary Variables to group by. Takes an empty list in case no grouping is needed.
    """
    scores = {}
    columns = [c for c in df.columns if ('liwc' in c) or ('empath' in c)]
    iterbar = tqdm(columns)
    assert(df.drop_duplicates(['quoteID', 'qid']).count() == df.count())
    groupby = ['year', 'month'] + groupby

    for col in iterbar:
        iterbar.set_description(col)
        iterbar.update(n=0)
        counts = df \
            .groupby([*groupby, col, 'numTokens']) \
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
    columns = [c for c in df.columns if ('liwc' in c) or ('empath' in c)]
    iterbar = tqdm(columns)

    for col in iterbar:
        iterbar.set_description(col)
        iterbar.update(n=0)
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


def getScoresBySpeakerGroup(df: DataFrame, groupby: List[str] = None) -> pd.DataFrame:
    """
    Lexicographic feature scores, macro-average over speakers.
    Parameters
    ----------
    df: Spark dataframe containing counts
    groupby: Grouping parameters
    """
    scores = {}
    columns = [c for c in df.columns if ('liwc' in c) or ('empath' in c)]
    iterbar = tqdm(columns)
    if groupby is None:
        groupby = ['year', 'month']
    else:
        groupby = ['year', 'month'] + groupby

    for col in iterbar:
        iterbar.set_description(col)
        iterbar.update(n=0)
        counts = df \
            .groupby([*groupby, 'qid', col, 'numTokens']) \
            .agg(f.count('*').alias('cnt')) \
            .withColumn('weighted_score', f.col('cnt') * f.col(col) / f.col('numTokens')) \
            .withColumn('yyyy-mm', f.concat(f.col('year'), f.lit('-'), f.col('month'))) \
            .filter(~f.col('yyyy-mm').isin(MISSING_MONTHS)) \
            .drop('yyyy-mm') \
            .groupby(list(groupby) + ['qid']) \
            .agg(f.sum('cnt').alias('total_cnt'), f.sum('weighted_score').alias('summed_weighted_score')) \
            .rdd \
            .map(lambda r: (*[r[g] for g in groupby], r['qid'], r['summed_weighted_score'], r['total_cnt'])).collect()

        for elements in itertools.chain(counts):
            elements = list(elements)
            cnt = elements.pop(-1)
            sws = elements.pop(-1)
            qid = elements.pop(-1)
            year = elements.pop(0)
            month = elements.pop(0)
            date = datetime(year, month, 15)

            if date not in scores:
                scores[date] = {}

            key = None if len(elements) == 0 else '-'.join(map(str, elements))
            if col not in scores[date]:
                if key is None:
                    scores[date][col] = []
                else:
                    scores[date][col] = {}
            if key is None:
                scores[date][col].append(sws / cnt)
            else:
                if key not in scores[date][col]:
                    scores[date][col][key] = []
                scores[date][col][key].append(sws / cnt)

    for date in scores:
        for col in scores[date]:
            if isinstance(scores[date][col], list):
                scores[date][col] = np.mean(scores[date][col])
            else:
                for key in scores[date][col]:
                    scores[date][col][key] = np.mean(scores[date][col][key])

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
    columns = [c for c in df.columns if ('liwc' in c) or ('empath' in c)]
    iterbar = tqdm(columns)

    for col in iterbar:
        iterbar.set_description(col)
        iterbar.update(n=0)
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

    if ((args.exclude_top_n is not None) or (args.extract_top_n is not None)) and (args.top_n_file is None):
        raise argparse.ArgumentError("If exclude_top_n / extract_top_n is given, top_n_file must be given, too.")

    base = Path(args.save)
    base.mkdir(exist_ok=True)

    df = spark.read.parquet(args.sentiment)
    features = [c for c in df.columns if ('liwc' in c) or ('empath' in c)]
    people = _prep_people(spark.read.parquet(args.people)).cache()
    df = df.join(people, on='qid')

    def save(_df: pd.DataFrame, name: str):
        savefile = base.joinpath(name + '.csv')
        savefile.parent.mkdir(parents=True, exist_ok=True)
        _df.to_csv(savefile.open('w'))

    # Basic Quotation Aggregation
    agg = getScoresByGroups(df, [])
    # The same standardization values will be used for all - take mean and std "pre-treatment" (before the primaries)
    if args.standardize:
        MEAN = agg[features][agg.index < KINK].mean()
        STD = agg[features][agg.index < KINK].std()
        pickle.dump(MEAN, base.joinpath('mean.pickle').open('wb'))
        pickle.dump(STD, base.joinpath('std.pickle').open('wb'))
    else:
        MEAN = 0
        STD = 1
    save(_df_postprocessing(agg, features, MEAN, STD), 'QuotationAggregation')

    agg = getScoresByGroups(df, ['gender', 'party', 'congress_member'])
    agg = _add_governing_column(agg)
    save(_df_postprocessing(agg, features, MEAN, STD), 'AttributesAggregation')

    agg = getScoresByGroups(df, ['party'])
    save(_df_postprocessing(agg, features, MEAN, STD), 'PartyAggregation')

    agg = getScoresByGroups(df.filter(f.col('qid') != 'Q22686'), ['party'])
    save(_df_postprocessing(agg, features, MEAN, STD), 'PartyAggregationWithoutTrump')

    agg = getScoresBySpeaker(df)
    save(_df_postprocessing(agg, features, MEAN, STD), 'SpeakerAggregation')

    agg = getScoresBySpeakerGroup(df)
    save(_df_postprocessing(agg, features, MEAN, STD), 'SpeakerAggregationSanity')

    agg = getScoresBySpeakerGroup(df, ['gender', 'party', 'congress_member'])
    agg = _add_governing_column(agg)
    save(_df_postprocessing(agg, features, MEAN, STD), 'AttributesAggregationSpeakerLevel')

    spark.sparkContext.setLogLevel('ERROR')  # Window Warning is Expected and can be ignored
    agg = getScoresByVerbosity(df)
    save(_df_postprocessing(agg, features, MEAN, STD), 'VerbosityAggregation')

    spark.sparkContext.setLogLevel('WARN')
    individuals = [] if args.individuals is None else args.individuals
    if args.extract_top_n is not None:
        rank_file = pd.read_csv(args.top_n_file)
        individuals = individuals + rank_file['QID'][rank_file['Rank'] <= args.extract_top_n].tolist()
    if len(individuals) > 0:
        base.joinpath('Individuals').mkdir(exist_ok=True)
    for qid in individuals:
        print('\n\nAggregation for qid:', qid)
        uttered = df.filter(f.col('qid') == qid)
        agg = getScoresByGroups(uttered, [])
        save(_df_postprocessing(agg, features, MEAN, STD), 'Individuals/{}'.format(qid))

    if args.exclude_top_n is not None:
        base.joinpath('Without').mkdir(exist_ok=True)
        rank_file = pd.read_csv(args.top_n_file)
        for i in range(args.exclude_top_n):
            rank = i + 1
            try:
                qid = rank_file['QID'][rank_file['Rank'] == rank].values[0]
            except IndexError:
                print('Rank {} is not available in the given speaker file.'.format(rank))
                continue  # Might be okay, e.g. if the given rank is very large, it can just serve as an "include all"
            print('Collecting for quotations for all but rank {}'.format(rank))
            tmp = df.filter(f.col('QID') != qid)
            agg = getScoresByGroups(tmp, [])
            save(_df_postprocessing(agg, features, MEAN, STD), 'Without/{}'.format(qid))


if __name__ == '__main__':
    main()
