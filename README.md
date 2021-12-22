# United States Politicians' Tone Became More Negative with 2016 Primary Campaigns

There is a widespread belief that the tone of US political language has become more negative recently, in particular when Donald Trump entered politics.
At the same time, there is disagreement as to whether Trump changed or merely continued previous trends.
To date, data-driven evidence regarding these questions is scarce, partly due to the difficulty of obtaining a comprehensive, longitudinal record of politicians' utterances.
Here we apply psycholinguistic tools to a novel, comprehensive corpus of 24 million quotes from online news attributed to 18,627 US politicians in order to analyze how the tone of US politicians' language evolved between 2008 and 2020.
We show that, whereas the frequency of negative emotion words had decreased continuously during Obama's tenure, it suddenly and lastingly increased with the 2016 primary campaigns, by 1.6 pre-campaign standard deviations, or 8% of the pre-campaign mean, in a pattern that emerges across parties.
The effect size drops by 40% when omitting Trump's quotes, and by 50% when averaging over speakers rather than quotes, implying that prominent speakers, and Trump in particular, have disproportionately, though not exclusively, contributed to the rise in negative language.
This work provides the first large-scale data-driven evidence of a drastic shift toward a more negative political tone following Trump's campaign start as a catalyst, with important implications for the debate about the state of US politics.

# Code Repository

This repository contains the code and aggregated data for the publication "United States Politicians' Tone Became More Negative with 2016 Primary Campaigns".


# How to Start

1. You need a local version of LIWC for the sentiment analysis. The code expects it to be a pickle file for a python dictionary, mapping {LIWC category -> Regular Expression}
2. Make sure you have Quotebank in the quotation centric format available.
3. Prepare a python environment according to requirements.txt with pyspark - for the paper, we used pyspark 2.4.0, but newer versions should work as well.
4. If you did not do so already, transform the Quotebank data from the json format it is provided in to the parquet format. There is a script for that purpose in the utils folder. If you do this by yourself, make sure the new dataframe contains the columns "year" and "month" - they are convenient for filtering and parquet partitioning. Also, the date column should be transformed to "yyyy-MM-dd" format.

# Folder Overview

- analysis: Contains the heart of this project: This includes the code to aggregate the Quotebank data, to fit ordinary linear regression models, to extract liwc scores from the quotations and to produce the figures as they appear in the paper.
- code_for_tex: Here you find python utilties to auto-generate the Latex-files for plots and tables in the supplementary material.
- config: Configuration files.
- data: Data aggregates, fitted regression models (RDD) and more.
- preparations: Utilities to filter and prepare Quotebank before applying any analysis functions on it. This includes extracting quotes from politicians and removing unwanted quotes.
- SI: The supplementary material
- utils: Most importantly, a plot wrapper function used for most figures in the paper. Moreover: Useful shortcuts and frequently needed spark commands.
