from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import pandas as pd
import seaborn as sns
import warnings


def _election_tuesday(year: int) -> datetime:
    """Elections are held on the tuesday following the first monday of November every year."""
    seventh = datetime(year, 11, 7)
    first_monday = seventh - timedelta(seventh.weekday())
    return first_monday + timedelta(1)


ELECTION_TUESDAYS = [_election_tuesday(y) for y in range(2008, 2020)]
PRESIDENTIAL_ELECTIONS = ELECTION_TUESDAYS[::4]

NANO = 1e-9
ONE_COL_FIGSIZE = [8.7, 5]
TWO_COL_FIGSIZE = [17.8, 8]
NARROW_TWO_COL_FIGSIZE = [17.8, 4.5]
NARROW_NARROW_TWO_COL_FIGSIZE = [17.8, 3]
LANDSCAPE_FIGSIZE = [25, 17.8]
LANDSCAPE_NARROW_FIGSIZE = [25, 8]


def timeLinePlot(x, y,
                 timeDelta: str = '1y',
                 snsargs: dict = None,
                 kind: str = 'line',
                 **kwargs):
    """
    A flexible function to plot y-Values versus points in time on the x-Axis.
    Library used for plotting is seaborn, key word arguments for seaborn and key word arguments for the matplotlib
    backend can be provided separately.
    Both x and y need to be mappable to np arrays.
    Parameters
    ----------
    x: Dates, either as datetime or numpy datetime.
    y: Numeric values
    timeDelta: Handles the distance of x-ticks and accepts a combination of integer between 0-9 and unit (d=day, m=month, y=year)
    snsargs: Additional seaborn key word arguments.
    kind: Scatter or Line
    kwargs: Key word arguments for matplotlib. These can be any of:
        "ax": Use a pre-existing axis
        "figsize": If ax not given, will create one axis in a figure of given figsize
        "includeElections": If given, will plot vertical lines at election dates.
        Anything that will work as argument for ax.update(), such as title or labels
    Returns
    -------
    A (fig, ax) tuple.
    """
    x, y = map(np.squeeze, map(np.asarray, (x, y)))

    figsize = kwargs.get('figsize', TWO_COL_FIGSIZE)
    timeDelta = [split for split in re.split('([0-9]+)', timeDelta) if split != '']

    if len(timeDelta) == 1:
        timeDeltaValue = 1
        timeDeltaUnit = timeDelta[0]
    else:
        timeDeltaValue = int(timeDelta[0])
        timeDeltaUnit = timeDelta[1]

    # Maps different kinds of datetime types to python datetime.
    try:
        x = [parse_date(d) for d in x]
    except TypeError:
        try:
            x = [pd.to_datetime(d) for d in x]
        except TypeError:
            pass

    if 'ax' not in kwargs:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = kwargs['ax']
        fig = plt.gcf()

    # Transform the date to numeric values for plotting
    try:
        ordinalDate = mdates.date2num(x)
        x = ordinalDate
    except ValueError as exe:
        print("X must be a date.")
        print(exe)
        return

    # -------------------------------- Build the Plot --------------------------------
    if snsargs is None:
        snsargs = {}

    if kind == 'line':
        plot = sns.lineplot
    elif kind == 'scatter':
        plot = sns.scatterplot
    elif kind == 'regplot':
        plot = sns.regplot
    else:
        raise NotImplementedError('Unknown plot kind {}', format(kind))

    plot(x=x, y=y, ax=ax, **snsargs)

    # ----------------------------------- Visuals -----------------------------------

    # Include a little space left and right for nicer visuals
    xOffset = (ordinalDate.max() - ordinalDate.min()) // 20
    ax.set_xlim(ordinalDate.min() - xOffset, ordinalDate.max() + xOffset)

    xaxis = ax.xaxis
    if timeDeltaUnit.lower() == 'd':
        loc = mdates.DayLocator(interval=timeDeltaValue)
        loc_minor = mdates.DayLocator(interval=1)
        xaxis.set_minor_locator(loc_minor)
        xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    elif timeDeltaUnit.lower() == 'm':
        loc = mdates.MonthLocator(interval=timeDeltaValue)
        loc_minor = mdates.MonthLocator(interval=1)
        xaxis.set_minor_locator(loc_minor)
        xaxis.set_major_formatter(mdates.AutoDateFormatter(loc))
    elif timeDeltaUnit.lower() == 'y':
        loc = mdates.YearLocator()
        xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        raise ValueError("Invalid Timedelta, don't understand the time delta Unit: {}".format(timeDeltaUnit))
    xaxis.set_major_locator(loc)

    # includeElection can be anything that is boolean true for presidential elections. If it is 'all',
    # all yearly November election tuesdays are plotted
    if bool(kwargs.get('includeElections', True)):
        for electionDay in ELECTION_TUESDAYS:
            if electionDay in PRESIDENTIAL_ELECTIONS:
                ax.axvline(x=electionDay, linewidth=2, c='grey', linestyle='dotted', alpha=0.5)
            else:
                if kwargs.get('includeElections') == 'all':
                    ax.axvline(x=electionDay, linewidth=1, c='grey', linestyle='dotted', alpha=0.5)

    # User defined inputs
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for kw in kwargs:
            try:
                ax.update({kw: kwargs[kw]})
            except AttributeError:
                pass

    fig.autofmt_xdate(rotation=90, ha='left')
    return fig, ax


def saveFigure(fig: plt.Figure, path: Path, dpi: int = 300, excludeTightLayout: bool = False):
    """
    A wrapper to save Figures as PDF
    Parameters
    ----------
    fig: Figure to save
    path: Path to save figure at
    dpi: Image resolution
    excludeTightLayout: Per default, tight layout will be applied. If False, will no do that.
    """
    if '.' not in path.name:
        path = Path(str(path) + '.pdf')

    if not excludeTightLayout:
        fig.tight_layout(pad=1.1)

    base = path.parent
    base.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
