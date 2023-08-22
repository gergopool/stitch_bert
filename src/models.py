from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, glue_processors
from transformers import ViTForImageClassification

from .static import TASKS

def build_pretrained_transformer(task_name: str, dataset: Dataset):
    if task_name in TASKS['nlp']:
        # NLP task
        n_classes = len(glue_processors[task_name]().get_labels())
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                   num_labels=n_classes)
    else:
        # Vision task
        if task_name == 'flowers':
            n_classes = max(dataset._labels) + 1
        else:
            n_classes = len(dataset.classes)
        model = ViTForImageClassification.from_pretrained("facebook/dino-vitb16",
                                                          num_labels=n_classes)

    return model