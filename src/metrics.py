import numpy as np
from abc import ABC
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

from .static import TASKS


class Metric(ABC):
    name = "Metric"

    def __call__(self, y_pred: np.array, y_true: np.array) -> float:
        raise NotImplementedError


class Accuracy(Metric):
    name = 'Accuracy'

    def __call__(self, y_pred: np.array, y_true: np.array) -> float:
        y_pred = y_pred.argmax(axis=1)
        y_true = y_true.flatten()
        return np.mean(y_true == y_pred)


class Correlation(Metric):
    name = 'Correlation'

    def __call__(self, y_pred: np.array, y_true: np.array) -> float:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        pearson_corr = pearsonr(y_pred, y_true)[0]
        spearman_corr = spearmanr(y_pred, y_true)[0]
        return (pearson_corr + spearman_corr) / 2


class MatthewsCorrelation(Metric):
    name = 'Matthews corr. coeff.'

    def __call__(self, y_pred: np.array, y_true: np.array) -> float:
        y_pred = y_pred.argmax(axis=1)
        y_true = y_true.flatten()
        return matthews_corrcoef(y_true, y_pred)


def get_metric_for(task_name: str) -> Metric:

    if task_name == 'cola':
        return MatthewsCorrelation()
    elif task_name == 'sts-b':
        return Correlation()
    elif task_name in ['mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst-2', 'wnli'] + TASKS['vis']:
        return Accuracy()
    else:
        raise ValueError(f"Unknown task name: {task_name}")