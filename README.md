# How to Start

1. You need a local version of LIWC for the sentiment analysis. The code expects it to be a pickle file for a python dictionary, mapping {LIWC category -> Regular Expression}
2. Make sure you have Quotebank in the quotation centric format available.
3. Prepare a python environment with pyspark - for the paper, we used pyspark 2.4.0, but newer versions should work as well.
4. If you did not do so already, transform the Quotebank data to the parquet format. There is a script for that purpose in the utils folder. If you do this by yourself, make sure the new dataframe contains the columns "year" and "month" - they are convenient for filtering and parquet partitioning. Also, the date column should be transformed to "yyyy-MM-dd" format.



## Order to run things

- Start with some preprocessing: Transform Quotebank to Parquet, download the Wikidata dump and extract the politicians from it.

## Expected Runtime

All times refer to a Setting using 22 CPUs and 26GB of Memory. When adapting to the tasks, faster results are possible.

__Preparations__

- Transforming Quotebank in the article-centric format from json to parquet: 8h [you don't need to do that]
- Transforming Quotebank in the quotation-centric format from json to parquet: 1.5h

- Reading relevant speaker attributes from a Wikidata dump (single core): 5h
- Transforming them to parquet: xxx
- Get all politicians and their quotes: 
