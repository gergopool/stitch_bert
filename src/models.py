import os
import torch
from torch.utils.data import Dataset
from torch import nn
from transformers import AutoModelForSequenceClassification, glue_processors
from transformers import ViTForImageClassification

from .static import TASKS, N_CLASSES, Logger


def build_pretrained_transformer(task_name: str):
    if task_name in TASKS['nlp']:
        # NLP task
        n_classes = len(glue_processors[task_name]().get_labels())
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                   num_labels=n_classes)
    else:
        n_classes = N_CLASSES[task_name]
        model = ViTForImageClassification.from_pretrained("facebook/dino-vitb16",
                                                          num_labels=n_classes)

    return model


def load_model(model_root: str, task: str, seed: int, device: torch.device) -> nn.Module:
    model_path = os.path.join(model_root, f"{task}_{seed}.pt")
    model = build_pretrained_transformer(task)
    model.load_state_dict(torch.load(model_path, map_location=device))
    Logger.info(f"Model loaded from file {model_path}.")
    return model