import os
import torch
from torch.utils.data import TensorDataset
from transformers import glue_processors, glue_output_modes, glue_convert_examples_to_features
from transformers import AutoTokenizer

from . import Logger


def load_glue_data_from_args(args, tokenizer, dev=False):
    """
    Load and cache examples from GLUE dataset given command line arguments.

    Parameters:
    - args: Argparse arguments.
    - tokenizer: AutoTokenizer instance.
    - dev: Whether to use evaluation data. If False, training data will be used.
    """
    if hasattr(args, 'task'):
        task = args.task
    elif hasattr(args, 'task2'):
        task = args.task2
    else:
        raise ValueError("args must have either 'task' or 'task2' attribute.")
    return load_glue_data(data_dir=args.data_dir,
                          task=task,
                          tokenizer=tokenizer,
                          dev=dev,
                          model_type=args.model_type,
                          overwrite_cache=args.overwrite_cache,
                          max_seq_length=args.max_seq_length)


def load_glue_data(data_dir: str,
                   task: str,
                   tokenizer: AutoTokenizer,
                   dev: bool = False,
                   model_type: str = 'bert',
                   overwrite_cache: bool = False,
                   max_seq_length: int = 128):
    """
    Load and cache examples from GLUE dataset.

    Parameters:
    - data_dir: Directory where the data is located.
    - task: Name of the task (dataset).
    - tokenizer: AutoTokenizer instance.
    - dev: Whether to use evaluation data. If False, training data will be used.
    - model_type: Type of the model. E.g. roberta, bert
    - overwrite_cache: If True, overwrite the cached data.
    - max_seq_length: Maximum sequence length for the tokenized data.

    Returns:
    - TensorDataset instance containing the cached data.
    """

    # Determine if to load from cache or from dataset file
    mode = "dev" if dev else "train"

    # Select task directory
    data_dir = os.path.join(data_dir, glue_dir_name(task))

    cached_features_file = os.path.join(data_dir, f"cached_{mode}_{max_seq_length}_{task}.pt")

    # Load from cache if possible, otherwise load from dataset
    features = load_from_cache(cached_features_file,
                               overwrite_cache,
                               data_dir,
                               task,
                               model_type,
                               tokenizer,
                               max_seq_length,
                               dev)

    # Convert to Tensors and build dataset
    return build_dataset(features, task)


def load_from_cache(cached_features_file: str,
                    overwrite_cache: bool,
                    data_dir: str,
                    task: str,
                    model_type: str,
                    tokenizer: AutoTokenizer,
                    max_seq_length: int,
                    dev: bool):
    """
    Load features from cache file or dataset.

    Parameters:
    - cached_features_file: Path to the cache file.
    - overwrite_cache: If True, overwrite the cached data.
    - data_dir: Directory where the data is located.
    - task: Name of the task (dataset).
    - model_type: Type of the model.
    - tokenizer: AutoTokenizer instance.
    - max_seq_length: Maximum sequence length for the tokenized data.
    - dev: Whether to use evaluation data. If False, training data will be used.

    Returns:
    - features: A list of InputFeatures.
    """
    if os.path.exists(cached_features_file) and not overwrite_cache:
        Logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        Logger.info("Creating features from dataset file at %s", data_dir)
        features = create_features_from_dataset(data_dir,
                                                task,
                                                model_type,
                                                tokenizer,
                                                max_seq_length,
                                                dev)
        Logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    return features


def create_features_from_dataset(data_dir: str,
                                 task: str,
                                 model_type: str,
                                 tokenizer: AutoTokenizer,
                                 max_seq_length: int,
                                 dev: bool):
    """
    Create features from dataset.

    Parameters:
    - data_dir: Directory where the data is located.
    - task: Name of the task (dataset).
    - model_type: Type of the model.
    - tokenizer: AutoTokenizer instance.
    - max_seq_length: Maximum sequence length for the tokenized data.
    - dev: Whether to use evaluation data. If False, training data will be used.

    Returns:
    - features: A list of InputFeatures.
    """

    # Prepare the processor and output mode
    processor = glue_processors[task]()
    output_mode = glue_output_modes[task]

    label_list = processor.get_labels()
    if task in ["mnli", "mnli-mm"] and model_type in ["roberta", "xlmroberta"]:
        # label indices are swapped in RoBERTa pretrained model
        label_list[1], label_list[2] = label_list[2], label_list[1]
    examples = (processor.get_dev_examples(data_dir)
                if dev else processor.get_train_examples(data_dir))

    features = glue_convert_examples_to_features(examples,
                                                 tokenizer,
                                                 label_list=label_list,
                                                 max_length=max_seq_length,
                                                 output_mode=output_mode)

    return features


def build_dataset(features, task: str):
    """
    Build a TensorDataset from features.

    Parameters:
    - features: A list of InputFeatures.
    - output_mode: Output mode for the task.

    Returns:
    - TensorDataset instance.
    """
    output_mode = glue_output_modes[task]
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def glue_dir_name(task: str):
    """Get name of task directory"""
    return task.upper().replace('COLA', 'CoLA')