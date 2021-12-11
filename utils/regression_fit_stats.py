import numpy as np
import scipy


def regression_fit_stats(target, prediction_model, dof_model=4, dof_comparison=None, prediction_comparison=None):
    """Basic f-value and p-value calculation in default setting. (Null hypothesis: Model better than mean)"""
    N = len(target)

    if prediction_comparison is None:
        prediction_comparison = np.mean(target)
        dof_comparison = 1

    residues = target - prediction_model
    residues_comp = target - prediction_comparison

    SSR_comp = np.sum(np.power(residues_comp, 2))
    SSR_model = np.sum(np.power(residues, 2))
    DIFF = SSR_comp - SSR_model

    f = (DIFF / (dof_model - dof_comparison)) / (SSR_model / (N - dof_model))
    p = 1 - scipy.stats.f.cdf(f, dof_model - dof_comparison, N - dof_model)
    print('yo')

    return f, p
