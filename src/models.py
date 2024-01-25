import os
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoConfig,
    glue_processors,
    ViTForImageClassification
)

from .static import TASKS, N_CLASSES, Logger

def build_pretrained_transformer(task_name: str, random_init: bool = False) -> nn.Module:
    """Builds a pretrained transformer model based on the task name.

    Args:
        task_name: The name of the task.

    Returns:
        Pretrained model for the given task.
    """
    if task_name == 'mlm':
        return AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    elif task_name in TASKS['nlp']:
        return _build_nlp_model(task_name, random_init)
    else:
        return _build_vision_model(task_name, random_init)


def _build_nlp_model(task_name: str, random_init: bool = False) -> nn.Module:
    """Builds a pretrained transformer model for NLP tasks."""
    n_classes = len(glue_processors[task_name]().get_labels())
    if random_init:
        config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=n_classes)
        return AutoModelForSequenceClassification.from_config(config)
    else:
        return AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                   num_labels=n_classes)


def _build_vision_model(task_name: str, random_init: bool = False) -> nn.Module:
    """Builds a pretrained transformer model for vision tasks."""
    n_classes = N_CLASSES[task_name]
    if random_init:
        config = AutoConfig.from_pretrained("facebook/dino-vitb16", num_labels=n_classes)
        return ViTForImageClassification(config)
    else:
        return ViTForImageClassification.from_pretrained("facebook/dino-vitb16",
                                                         num_labels=n_classes)


def load_model(model_root: str,
               task: str,
               seed: int,
               device: torch.device,
               random_weights: bool = False) -> nn.Module:
    """Loads a pretrained model for a given task.

    Args:
        model_root: The path to the model directory.
        task: The name of the task.
        seed: The seed value.
        device: The device to load the model on.
        random_weights: Whether to load a randomly initialized model.

    Returns:
        Pretrained model for the given task.
    """
    if random_weights:
        return load_random_model(task, device)
    else:
        model_path = os.path.join(model_root, f"{task}_{seed}.pt")
        model = build_pretrained_transformer(task)
        model.load_state_dict(torch.load(model_path, map_location=device))
        Logger.info(f"Model loaded from file {model_path}.")
        return model

def load_random_model(task: str, device: torch.device) -> nn.Module:
    """Loads a randomly initialized model for a given task.

    Args:
        task: The name of the task.
        device: The device to load the model on.

    Returns:
        Randomly initialized model for the given task.
    """
    model = build_pretrained_transformer(task, True).to(device)
    Logger.info(f"Randomly initialized model loaded.")
    return model
