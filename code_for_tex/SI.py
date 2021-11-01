from pathlib import Path


def plots(plot_dir: Path, store_in: Path):
    """Creates a tex file with labeled plot references."""
    for folder in plot_dir.iterdir():
        pass # TODO: Define order and reference in order.


def main():
    """Note that this works only if you kept the original project structure."""
    SI = Path(__file__).parent.joinpath('SI')
    SI.mkdir(exist_ok=True)
    data = SI.parent.joinpath('data')
    plots(SI.parent.joinpath('img'), SI.joinpath('plots.tex'))


if __name__ == '__main__':
    main()