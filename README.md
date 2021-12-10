# United States Politicians' Tone Became More Negative\\with 2016 Primary Campaigns

There is a widespread belief that the tone of US political language has become more negative recently, in particular when Donald Trump entered politics.
At the same time, there is disagreement as to whether Trump changed or continued previous trends. To date, data-driven evidence regarding these questions is scarce, partly due to the difficulty of obtaining a comprehensive, longitudinal record of politicians' utterances. Here we apply psycholinguistic tools to a novel, comprehensive corpus of 24.3 million quotes from online news attributed to 18,954 US politicians in order to analyze how the tone of US politicians' language evolved between 2008 and 2020. We show that, whereas negative emotion words had decreased continuously during Obama's tenure, they suddenly and lastingly increased with the 2016 primary campaigns, by 2.2 pre-campaign standard deviations, or 11\% of the pre-campaign mean, in a pattern that emerges across parties. The effect size drops to one-half when omitting Trump's quotes, and to one-third when averaging over speakers rather than quotes, implying that prominent speakers, and Trump in particular, have disproportionately, though not exclusively, contributed to the rise in negative language. This work provides the first large-scale data-driven evidence of a drastic shift toward a more negative political tone following Trump's campaign start as a catalyst, with important implications for the debate about the state of US politics.

# Code Repository

This repository contains the code and aggregated data for the publication "United States Politicians' Tone Became More Negative\\with 2016 Primary Campaigns".

# MISC

What needs to be inside the README?

# How to Start

1. You need a local version of LIWC for the sentiment analysis. The code expects it to be a pickle file for a python dictionary, mapping {LIWC category -> Regular Expression}
2. Make sure you have Quotebank in the quotation centric format available.
3. Prepare a python environment with pyspark - for the paper, we used pyspark 2.4.0, but newer versions should work as well.
4. If you did not do so already, transform the Quotebank data to the parquet format. There is a script for that purpose in the utils folder. If you do this by yourself, make sure the new dataframe contains the columns "year" and "month" - they are convenient for filtering and parquet partitioning. Also, the date column should be transformed to "yyyy-MM-dd" format.
