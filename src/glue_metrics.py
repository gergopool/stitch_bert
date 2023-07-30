import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from scipy.special import softmax


class Accuracy:
    name = 'Accuracy'

    def __call__(self, y_pred, y_true):
        y_pred = y_pred.argmax(axis=1)
        y_true = y_true.flatten()
        return np.mean(y_true == y_pred)


class Correlation:
    name = 'Correlation'

    def __call__(self, y_pred, y_true):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        pearson_corr = pearsonr(y_pred, y_true)[0]
        spearman_corr = spearmanr(y_pred, y_true)[0]
        return (pearson_corr + spearman_corr) / 2


class MatthewsCorrelation:
    name = 'Matthews corr. coeff.'

    def __call__(self, y_pred, y_true):
        y_pred = y_pred.argmax(axis=1)
        y_true = y_true.flatten()
        return matthews_corrcoef(y_true, y_pred)


def get_metric_for(task_name):

    if task_name == 'cola':
        return MatthewsCorrelation()
    elif task_name == 'sts-b':
        return Correlation()
    elif task_name in ['mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst-2', 'wnli']:
        return Accuracy()
    else:
        return NameError(f"Unknown task name: {task_name}")