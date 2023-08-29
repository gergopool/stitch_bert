import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, BatchSampler, RandomSampler, DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import glue_processors, glue_output_modes, glue_convert_examples_to_features
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from typing import Any, Iterator, List
from datasets import load_dataset

from . import Logger
from .static import TASKS, GlobalState

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Settings for those datasets that has no train-dev-test split
SPLIT_MAX_TEST_SIZE = 20000
SPLIT_MAX_TRAIN_SIZE = 200000
TEST_SPLIT_RATIO = 0.1


def load_data_from_args(args, n_iters: int = None, n_points: int = None, dev=False) -> DataLoader:
    """
    Load data from argparse arguments.

    Parameters:
    - args: Argparse arguments.
    - n_iters: Number of iterations to run the training loop.
    - n_points: Limiting the dataset to n_points.
    - dev: Whether to use evaluation data. If False, training data will be used.

    Returns:
    - DataLoader instance with exactly n_iters batches.
    """
    if hasattr(args, 'task'):
        task = args.task
    elif hasattr(args, 'task2'):
        task = args.task2
    else:
        raise ValueError("args must have either 'task' or 'task2' attribute.")

    return load_data(task=task,
                     data_dir=args.data_dir,
                     n_iters=n_iters,
                     n_points=n_points,
                     batch_size=args.batch_size,
                     overwrite_cache=args.overwrite_cache,
                     max_seq_length=args.max_seq_length,
                     dev=dev)


def load_data(task: str,
              data_dir: str,
              n_iters: int = None,
              n_points: int = None,
              batch_size: int = 32,
              overwrite_cache: bool = False,
              max_seq_length: int = 128,
              dev: bool = False) -> DataLoader:
    """
    Load data loader for the given task.

    Parameters:
    - task: Name of the task (e.g. cola, mnist, etc).
    - data_dir: Directory where the data is located.
    - n_iters: Number of iterations to run the training loop.
    - n_points: Limiting the dataset to n_points.
    - batch_size: Batch size.
    - overwrite_cache: If True, overwrite the cached data.
    - max_seq_length: Maximum sequence length for the tokenized data.
    - dev: Whether to use evaluation data. If False, training data will be used.

    Returns:
    - DataLoader instance with exactly n_iters batches.
    """
    assert n_iters is None or not dev, "n_iters must be None if dev is True"

    # Load dataset
    if task in TASKS['vis']:
        dataset = load_vision_data(data_dir=data_dir, task=task, dev=dev)
        collate_fn = None
    elif task == 'mlm':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        dataset = load_mlm_data(data_dir=data_dir, 
                                dev = dev,
                                max_seq_length=max_seq_length)
        collate_fn = DataCollatorForLanguageModeling(tokenizer=tokenizer) 
    else:
        root = os.path.join(data_dir, "glue_data")
        dataset = load_glue_data(data_dir=root,
                                 task=task,
                                 dev=dev,
                                 overwrite_cache=overwrite_cache,
                                 max_seq_length=max_seq_length)
        collate_fn = None

    if GlobalState.debug:
        batch_size = 4
        dataset = LimitedDataWrapper(dataset, n_points=20)
        n_iters = 2
    elif n_points is not None:
        dataset = LimitedDataWrapper(dataset, n_points=n_points)

    # Define batch sampler for exactly N iterations
    if not dev:
        batch_sampler = CustomBatchSampler(num_of_iterations=n_iters,
                                           sampler=RandomSampler(dataset),
                                           batch_size=batch_size,
                                           drop_last=not dev)
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_sampler=batch_sampler,
                                                  num_workers=8,
                                                  persistent_workers=True,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  shuffle=False,
                                                  batch_size=batch_size,
                                                  num_workers=8,
                                                  persistent_workers=True,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)

    return data_loader


def load_vision_data(data_dir: str, task: str, dev: bool = False) -> Dataset:
    """
    Load and cache examples from vision dataset.

    Parameters:
    - data_dir: Directory where the data is located.
    - task: Name of the task (dataset).
    - dev: Whether to use evaluation data. If False, training data will be used.
    """
    if task not in TASKS['vis']:
        raise ValueError(f"Task {task} not found in {TASKS['vis']}")

    if dev:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])

    # Dynamically load the dataset class from torchvision
    assert task in TASKS['vis'], f"Task {task} not found in {TASKS['vis']}"
    return globals()[task](root_dir=data_dir, train=not dev, download=True, transform=transform)


def load_glue_data(data_dir: str,
                   task: str,
                   dev: bool = False,
                   overwrite_cache: bool = False,
                   max_seq_length: int = 128):
    """
    Load and cache examples from GLUE dataset.

    Parameters:
    - data_dir: Directory where the data is located.
    - task: Name of the task (dataset).
    - dev: Whether to use evaluation data. If False, training data will be used.
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
                               max_seq_length,
                               dev)

    # Convert to Tensors and build dataset
    return build_dataset(features, task)

def load_mlm_data(data_dir: str,
                  dev: bool = False,
                  overwrite_cache: bool = False,
                  max_seq_length: int = 128):
    """
    Load and cache examples from Wikipedia dataset.

    Parameters:
    - data_dir: Directory where the data is located.
    - dev: Whether to use evaluation data. If False, training data will be used.
    - overwrite_cache: If True, overwrite the cached data.
    - max_seq_length: Maximum sequence length for the tokenized data.

    Returns:
    - Dataset instance containing the cached data.
    """

    # Determine if to load from cache or from dataset file
    mode = "dev" if dev else "train"

    # Select task directory
    save_folder = os.path.join(data_dir, 'wikipedia', '200k_cached')
    os.makedirs(save_folder, exist_ok=True)

    cached_features_file = os.path.join(save_folder, f"cached_{mode}_{max_seq_length}.pt")

    # Load from cache if possible, otherwise load from dataset
    dataset = load_from_cache(cached_features_file,
                               overwrite_cache,
                               data_dir,
                               'mlm',
                               max_seq_length,
                               dev)
    return dataset

def load_wiki_data(data_dir: str = "~/.cache/huggingface/datasets",
                   dev: bool = False,
                   max_seq_length: int = 128):
    """
    Load masked language model (MLM) data.

    Parameters:
    - data_dir: The directory containing the data.
    - dev: If True, loads the development (dev) data. Otherwise, loads the training data. Default is False.
    - max_seq_length: The maximum sequence length for tokenization. Default is 128.

    Returns:
    - dataset: The loaded dataset in a suitable format for MLM.
    """

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    # Prepare dataset split
    dataset = load_dataset("wikipedia", "20220301.en", cache_dir=data_dir)['train'] 
    dataset = dataset.remove_columns(["id", "url","title"])
    dataset = dataset.shuffle(seed=0)
    dataset = get_split(dataset, "dev" if dev else "train")


    def tokenize(batch):
        return tokenizer(batch['text'],
                         padding= 'max_length',
                         truncation= True,
                         max_length = max_seq_length)
    
    tokenized_dataset = dataset.map(tokenize, batched = True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])

    return tokenized_dataset




def get_split(dataset: Dataset, mode: str) -> Dataset:
    """
    Split the dataset into train, validation, and test sets.

    Parameters:
    - dataset: List of strings representing sentences.
    - mode: train, dev or test for 

    Returns:
    - sampler (Subset): Sampler for the training set.
    """
 
    dataset_length = len(dataset)
    train_size = min(int(dataset_length * (1 - (2*TEST_SPLIT_RATIO))), SPLIT_MAX_TRAIN_SIZE)
    test_size = min(int(dataset_length * TEST_SPLIT_RATIO), SPLIT_MAX_TEST_SIZE)

    indices = list(range(dataset_length))
    if mode == 'train':
        split_indices = indices[:train_size]
    elif mode=='dev':
        split_indices = indices[-2*test_size:-test_size]
    elif mode=='test':
        split_indices = indices[-test_size:]
    
    sampler = dataset.select(split_indices)
    return sampler

def load_from_cache(cached_features_file: str,
                    overwrite_cache: bool,
                    data_dir: str,
                    task: str,
                    max_seq_length: int,
                    dev: bool) -> list:
    """
    Load features from cache file or dataset.

    Parameters:
    - cached_features_file: Path to the cache file.
    - overwrite_cache: If True, overwrite the cached data.
    - data_dir: Directory where the data is located.
    - task: Name of the task (dataset).
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
        if task == 'mlm':
            features = load_wiki_data(data_dir, dev, max_seq_length)
        else:
            features = create_features_from_dataset(data_dir, task, max_seq_length, dev)
        Logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    return features


def create_features_from_dataset(data_dir: str,
                                 task: str,
                                 max_seq_length: int,
                                 dev: bool,
                                 model_type: str = 'bert-base-uncased') -> list:
    """
    Create features from dataset.

    Parameters:
    - data_dir: Directory where the data is located.
    - task: Name of the task (dataset).
    - max_seq_length: Maximum sequence length for the tokenized data.
    - dev: Whether to use evaluation data. If False, training data will be used.
    - model_type: Type of the model.

    Returns:
    - features: A list of InputFeatures.
    """

    # Prepare the processor and output mode
    processor = glue_processors[task]()
    output_mode = glue_output_modes[task]

    tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=False)

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


def build_dataset(features : list, task: str) -> TensorDataset:
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


# =======================================================================
# Vision datasets
# =======================================================================


def cifar10(root_dir: str, train: bool = True, download: bool = True, transform: nn.Module = None):
    root = os.path.join(root_dir, 'cifar10')
    return datasets.CIFAR10(root=root, train=train, download=download, transform=transform)


def cifar100(root_dir: str, train: bool = True, download: bool = True, transform: nn.Module = None):
    root = os.path.join(root_dir, 'cifar100')
    return datasets.CIFAR100(root=root, train=train, download=download, transform=transform)


def pets(root_dir: str, train: bool = True, download: bool = True, transform: nn.Module = None):
    split = 'trainval' if train else 'test'
    root = os.path.join(root_dir, 'oxford_iiit_pet')
    return datasets.OxfordIIITPet(root=root, split=split, download=download, transform=transform)


def flowers(root_dir: str, train: bool = True, download: bool = True, transform: nn.Module = None):
    split = 'train' if train else 'test'
    root = os.path.join(root_dir, 'flowers102')
    return datasets.Flowers102(root=root, split=split, download=download, transform=transform)


def food(root_dir: str, train: bool = True, download: bool = True, transform: nn.Module = None):
    split = 'train' if train else 'test'
    root = os.path.join(root_dir, 'food101')
    return datasets.Food101(root=root, split=split, download=download, transform=transform)


def dtd(root_dir: str, train: bool = True, download: bool = True, transform: nn.Module = None):
    split = 'train' if train else 'test'
    root = os.path.join(root_dir, 'dtd')
    return datasets.DTD(root=root, split=split, download=download, transform=transform)


def aircraft(root_dir: str, train: bool = True, download: bool = True, transform: nn.Module = None):
    split = 'train' if train else 'test'
    root = os.path.join(root_dir, 'aircraft')
    return datasets.FGVCAircraft(root=root, split=split, download=download, transform=transform)


class CustomBatchSampler(BatchSampler):
    """
    Custom batch sampler which iterates over a dataset by providing batches of the data.
    The number of batches 'returned' by the sampler is independent of the dataset size.
    When the required length is larger than the dataset size, it starts a new round internally.
    It can be used as an 'infinite' dataloader.
    Last batch of the original epoch can be dropped or kept.
    """

    def __init__(self, num_of_iterations: int, *args, **kwargs):
        """
        param num_of_iterations: defines the number of the batches after the sampler is exhausted
        """
        super().__init__(*args, **kwargs)
        self.num_of_iterations = num_of_iterations
        self.actual_iter_idx = 0

    def __iter__(self) -> Iterator[List[int]]:
        iterator = super().__iter__()
        self.actual_iter_idx = 0
        while self.actual_iter_idx < len(self):
            try:
                yield next(iterator)
                self.actual_iter_idx += 1
            except StopIteration:
                iterator = super().__iter__()

    def reset(self):
        # -1 because it will be incremented to 0 in the __iter__ method
        self.actual_iter_idx = -1

    def __len__(self) -> int:
        return self.num_of_iterations


class LimitedDataWrapper(Dataset):

    def __init__(self, dataset, n_points=20):
        self.dataset = dataset
        self.n_points = n_points

    def __getattr__(self, name):
        # Delegate attribute access to the underlying dataset
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(self.n_points, len(self.dataset))