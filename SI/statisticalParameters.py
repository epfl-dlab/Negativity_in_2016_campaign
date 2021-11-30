from pathlib import Path
import pandas as pd
import pickle
import sys
from tqdm import tqdm
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))  # Only works when keeping the original repo structure
from analysis.RDD import RDD

MODEL_PATH = Path(__file__).parent.parent.joinpath('data').joinpath('RDD')
SAVE = Path(__file__).parent.joinpath('statisticalParameters.pickle')
PARAMETER_MAP = {''
                 'Intercept': r'$\alpha_0$',
                 'C(threshold)[T.1]': r'$\alpha$',
                 'time_delta': r'$\beta_0$',
                 'C(threshold)[T.1]:time_delta': r'$\beta$'}


def recursiveIterdir(path: Path, criterion: callable, maxDepth: int) -> List[Path]:
    """Takes a path and returns all files matching a given criterion recursively with a maximum recursion depth."""
    if maxDepth == 0:
        return []
    files = [f for f in path.iterdir() if (f.is_file()) and criterion(f)]
    for child in path.iterdir():
        if child.is_dir():
            files += recursiveIterdir(child, criterion, maxDepth - 1)

    return files


def extractParameters(model: RDD) -> Dict:
    ret = dict()
    try:
        features = model.rdd.columns
    except AttributeError:
        return None
    for f in features:
        p = model.rdd_fit[f].pvalues.rename(PARAMETER_MAP)
        cov = model.rdd_fit[f].cov_params().rename(columns=PARAMETER_MAP, index=PARAMETER_MAP)
        param = model.rdd_fit[f].params.rename(PARAMETER_MAP)
        ret[f] = {'p-Values': p, 'covariance': cov, 'parameters': param}
    return ret


def main():
    def __filter(x: Path) -> bool: return x.name.endswith('.pickle')

    rddFiles = recursiveIterdir(MODEL_PATH, __filter, 3)

    parameters = dict()
    for model in tqdm(rddFiles):
        parameters[model.name.split('.')[0]] = extractParameters(pickle.load(model.open('rb')))

    pickle.dump(parameters, SAVE.open('wb'))


if __name__ == '__main__':
    main()
