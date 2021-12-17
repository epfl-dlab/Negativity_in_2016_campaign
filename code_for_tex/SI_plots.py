from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
# Relative path, Caption
FIGURES = [
    ('SI/ALL_AGG', 'Quote-level and speaker-level aggregation, with and without outliers (+ empath control categories)'),
    ('SI/Negativity_Scatter', 'Standardized raw data for quote-level and speaker-level aggregates'),
    ('verbosity/Verbosity_all', 'Speaker-level aggregation by prominence quartiles (extended version of Fig. 3 in the main text)'),  # Was 4
    ('SI/Party_all', 'Quote-level aggregation split by party'),  # Was 5
    ('RDD_time_variation/r2_adj', r'Adjusted R2 score of the OLS regression as a function of the discontinuity placement'),
    ('AttributesAggregationSpeakerLevel_RDD/selected_attributes', 'Biographic correlates of negative language based on speaker-level aggregation'),
    ('Individuals/verbosity_vs_beta', r'Regression parameter $\beta$ resulting from an ordinary least squares regressions fitted separately to the time series of each of the 200 most quoted speakers. For detailed description of plot format, see caption of Fig. 5 in main text.'),
    ('SI/Individuals_Grid', 'Quote-level aggregation for prominent politicians.'),  # Was 3
    ('Without/verbosity_vs_beta', r'Regression parameter $\beta$ obtained by removing all quotes by one target speaker, using the 50 most quoted speakers as target speakers'),
    ('RDD_time_variation/r2_adj_posemo', r'Adjusted R2 score of the OLS regression as a function of the discontinuity placement for positive emotions'),
    ('RDD_time_variation/r2_empath', r'Adjusted R2 score of the OLS regression as a function of the discontinuity placement for empath-based scores'),
    ('SI/quantities', r'Monthly number of speakers and unique quotations in Quotebank (restricted to the 18,627 US politicians considered in the analysis). Months that were removed due missing data (cf. Materials and Methods in main text) are plotted in gray.'),
]


def make_figure(path: Path, caption: str, label: str) -> str:
    label = ''.join(letter for letter in label if (letter not in r'$\\'))
    txt = r'\begin{figure}[h]\centering' + '\n\t'
    txt += r'\includegraphics[width =\linewidth]{' + str(path.absolute()) + '}' + '\n\t'
    txt += r'\caption{' + caption + '}\n\t'
    txt += r'\label{fig: ' + label + '}' + '\n'
    txt += r'\end{figure}' + '\n\n'
    return txt


def main():
    """Note that this works only if you kept the original project structure."""
    SI = Path(__file__).parent.parent.joinpath('SI')
    SI.mkdir(exist_ok=True)
    img = SI.parent.joinpath('img')

    txt = ''
    for i, (relativePath, caption) in enumerate(FIGURES):
        path = img.joinpath(relativePath + '.pdf')
        txt += make_figure(path, caption, 'SI_' + str(i+1))
        txt += '\n\n'

    txt += r'\clearpage' + '\n'
    txt += r'\pagebreak' + '\n'

    with SI.joinpath('plots.tex').open('w') as tex_file:
        tex_file.write(txt)


if __name__ == '__main__':
    main()
