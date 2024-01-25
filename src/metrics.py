import numpy as np
from abc import ABC
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
import torch.nn as nn
import torch

from .static import TASKS


from .static import TASKS


class Metric(ABC):
    name = "Metric"

    @property
    def value(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def accumulate(self,  y_pred: np.array, y_true: np.array):
        raise NotImplementedError
    
    def __call__(self, y_pred: np.array, y_true: np.array) -> float:
        self.accumulate(y_pred, y_true)
        return self.value


class Accuracy(Metric):
    name = 'Accuracy'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corrects = []

    @property
    def value(self):
        corrects = np.concatenate(self.corrects)
        return np.mean(corrects)

    def accumulate(self, y_pred: np.array, y_true: np.array) -> float:
        y_pred = y_pred.argmax(axis=1)
        y_true = y_true.flatten()
        self.corrects.append(y_pred == y_true)

    def reset(self):
        self.corrects = []


class Correlation(Metric):
    name = 'Correlation'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = []
        self.labels = []

    @property
    def value(self):
        preds = np.concatenate(self.preds)
        labels = np.concatenate(self.labels)
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return (pearson_corr + spearman_corr) / 2
    
    def accumulate(self, y_pred: np.array, y_true: np.array):
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()
        self.preds.append(y_pred)
        self.labels.append(y_true)

    def reset(self):
        self.preds = []
        self.labels = []


class MatthewsCorrelation(Metric):
    name = 'Matthews corr. coeff.'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preds = []
        self.labels = []

    @property
    def value(self):
        self.labels = np.concatenate(self.labels)
        self.preds = np.concatenate(self.preds)
        return matthews_corrcoef(self.labels, self.preds)
    
    def accumulate(self, y_pred: np.array, y_true: np.array):
        y_pred = y_pred.argmax(axis=1)
        y_true = y_true.flatten()
        self.preds.append(y_pred)
        self.labels.append(y_true)

    def reset(self):
        self.preds = []
        self.labels = []


class Perplexity:
    name = 'Perplexity'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy = nn.CrossEntropyLoss(reduction='none')
        self.losses = []

    @property
    def value(self):
        mean_loss = torch.mean(torch.cat(self.losses))
        return -torch.exp(mean_loss).item()

    def accumulate(self, y_pred: np.array, y_true: np.array) -> float:
        batch_y_pred = torch.tensor(y_pred).permute(0, 2, 1).float()
        batch_y_true = torch.tensor(y_true).long()
        eval_loss = self.entropy(batch_y_pred, batch_y_true)
        self.losses.append(eval_loss)

    def reset(self):
        self.losses = []

    

def get_metric_for(task_name: str) -> Metric:

    if task_name == 'cola':
        return MatthewsCorrelation()
    elif task_name == 'sts-b':
        return Correlation()
    elif task_name in ['mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst-2', 'wnli'] + TASKS['vis']:
        return Accuracy()
    elif task_name == 'mlm':
        return Perplexity()
    else:
        raise ValueError(f"Unknown task name: {task_name}")