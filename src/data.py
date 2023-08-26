import os
import torch
from torch.utils.data import TensorDataset
from transformers import glue_processors, glue_output_modes, glue_convert_examples_to_features
from transformers import AutoTokenizer
from datasets import load_dataset
from . import Logger
import torch.utils.data as data
from torch.utils.data import TensorDataset

WIKI_TRAINING_SAMPLES = 1_000_000
WIKI_VALIDATION_SAMPLES = 10_000
WIKI_TEST_SAMPLES = 10_000

class MLM_Dataset(data.Dataset):
    "PyTorch Dataset class for Masked Language Model (MLM) data."
    def __init__(self, train_input_ids, train_attention_mask):
        """
        Initialize the MLM_Dataset.

        Parameters:
        - train_input_ids: tensor of input token IDs for the training data.
        - train_attention_mask: tensor of attention mask values for the training data.
        """
        self.train_input_ids, self.train_attention_mask = train_input_ids, train_attention_mask

    def __len__(self):
        """
        Get the total number of data points in the dataset.

        Returns:
        - int: Total number of data points in the dataset.
        """
        return len(self.train_input_ids)
    
    def __getitem__(self, idx):
        """
        Get the idx-th data point from the dataset.

        Parameters:
        - idx: int, Index of the desired data point.

        Returns:
            dict: A dictionary containing the 'input_ids' and 'attention_mask' for the idx-th data point.
        """
        train_input_ids = self.train_input_ids[idx]
        train_attention_mask = self.train_attention_mask[idx]
        return {'input_ids':train_input_ids,'attention_mask':train_attention_mask}

def load_data_from_args(args, tokenizer, dev=False):
    """
    Load and cache examples from GLUE or MLM dataset given command line arguments.

    Parameters:
    - args: Argparse arguments.
    - tokenizer: AutoTokenizer instance.
    - dev: Whether to use evaluation data. If False, training data will be used.
    """
    if args.task == 'mlm':
        return load_mlm_data(data_dir = args.data_dir, 
                             task = args.task,
                             tokenizer = tokenizer,
                             dev = dev,
                             overwrite_cache=args.overwrite_cache,
                             max_seq_length=args.max_seq_length)
    else:
        return load_glue_data(data_dir=args.data_dir,
                          task=args.task,
                          tokenizer=tokenizer,
                          dev=dev,
                          model_type=args.model_type,
                          overwrite_cache=args.overwrite_cache,
                          max_seq_length=args.max_seq_length)

def load_mlm_data(data_dir: str,
                  task: str,
                  tokenizer: AutoTokenizer,
                  dev: bool = False,
                  overwrite_cache: bool = False,
                  max_seq_length: int = 128):
    """
    Load masked language model (MLM) data.

    Parameters:
    - data_dir: The directory containing the data.
    - task: A string identifier for the specific task or dataset.
    - tokenizer: AutoTokenizer instance
    - dev: If True, loads the development (dev) data. Otherwise, loads the training data. Default is False.
    - overwrite_cache: If True, overwrite the cached features file. Default is False.
    - max_seq_length: The maximum sequence length for tokenization. Default is 128.

    Returns:
    - dataset: The loaded dataset in a suitable format for MLM.
    """
    mode = "dev" if dev else "train"

    cached_features_file = os.path.join(data_dir, f"cached_{max_seq_length}_{task}.pt")
    # Load from cache if possible, otherwise load from dataset
    dataset = load_wikipedia_from_cache(cached_features_file,
                                overwrite_cache,
                                data_dir,
                                tokenizer,
                                max_seq_length,
                                mode)

    # Convert to Tensors and build dataset
    return dataset

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
    features = load_glue_from_cache(cached_features_file,
                               overwrite_cache,
                               data_dir,
                               task,
                               model_type,
                               tokenizer,
                               max_seq_length,
                               dev)

    # Convert to Tensors and build dataset
    return build_glue_dataset(features, task)


def split_data(dataset, mode):
    """
    Split the dataset into train, validation, and test sets.

    Parameters:
    - dataset: List of strings representing sentences.
    - mode: train, dev or test for 

    Returns:
    - sampler (Subset): Sampler for the training set.
    """
 
    dataset_length = len(dataset)

    indices = list(range(dataset_length))
    if mode == 'train':
        split_indices = indices[:WIKI_TRAINING_SAMPLES]
    elif mode=='dev':
        split_indices = indices[WIKI_TRAINING_SAMPLES:WIKI_TRAINING_SAMPLES+WIKI_VALIDATION_SAMPLES]
    elif mode=='test':
        split_indices = indices[WIKI_TRAINING_SAMPLES+WIKI_VALIDATION_SAMPLES:WIKI_TRAINING_SAMPLES+WIKI_VALIDATION_SAMPLES+WIKI_TEST_SAMPLES]
    
    sampler = dataset.select(split_indices)
    return sampler

def load_glue_from_cache(cached_features_file: str,
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
        features = create_features_from_glue_dataset(data_dir,
                                                task,
                                                model_type,
                                                tokenizer,
                                                max_seq_length,
                                                dev)
        Logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    return features

def load_wikipedia_from_cache(cached_features_file: str,
                    overwrite_cache: bool,
                    data_dir: str,
                    tokenizer: AutoTokenizer,
                    max_seq_length: int,
                    mode: str):
    """
    Load features from cache file or dataset.

    Parameters:
    - cached_features_file: Path to the cache file.
    - overwrite_cache: If True, overwrite the cached data.
    - data_dir: Directory where the data is located.
    - tokenizer: AutoTokenizer instance.
    - max_seq_length: Maximum sequence length for the tokenized data.
    - mode: Either dev or train.

    Returns:
    - dataset: torch.utils.Dataset.
    """
    if os.path.exists(cached_features_file) and not overwrite_cache:
        Logger.info("Loading features from cached file %s", cached_features_file)
        dataset = torch.load(cached_features_file)
    else:
        Logger.info("Creating features from dataset file at %s", data_dir)
        dataset = create_wikipedia_dataset(tokenizer,
                                 max_seq_length,
                                 mode)
        
        Logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(dataset, cached_features_file)

    return dataset

def create_features_from_glue_dataset(data_dir: str,
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

def tokenize_batch(examples, tokenizer, max_length):
    dict =  tokenizer(examples['text'], padding= 'max_length', truncation= True,
                      max_length = max_length)
    return dict

def create_wikipedia_dataset(tokenizer: AutoTokenizer,
                                 max_seq_length: int,
                                 mode: str):
    """
    Creates features for a Wikipedia dataset to be used.

    Parameters:
        tokenizer: Tokenizer to convert text into tokenized input.
        max_seq_length: Maximum sequence length for tokenized inputs.
        mode: train or val 
    Returns:
        tokenized_dataset: A dataset containing training features.
    """
    #! pip install apache-beam
    dataset = load_dataset("wikipedia", "20220301.en")['train'] 
    dataset = dataset.remove_columns(["id", "url","title"])
    dataset = dataset.shuffle(seed=0)
    
    dataset = split_data(dataset, mode)

    tokenized_dataset = dataset.map(lambda batch: tokenize_batch(batch, tokenizer, max_seq_length), batched = True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    return tokenized_dataset

def input_features(features):
    """
    Converts a list of feature objects into tensors representing input IDs and attention masks.

    Parameters:
    - features: A list of features containing, among others, input_ids and attention_mask attributes.

    Returns:
    - tuple: A tuple containing two torch tensors - all_input_ids and all_attention_mask.
               - all_input_ids: A tensor containing input IDs for all features.
               - all_attention_mask: A tensor containing attention masks for all features.
    """
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    return all_input_ids, all_attention_mask

def build_glue_dataset(features, task: str):
    """
    Build a TensorDataset from features.

    Parameters:
    - features: A list of InputFeatures.
    - output_mode: Output mode for the task.

    Returns:
    - TensorDataset instance.
    """
    output_mode = glue_output_modes[task]

    all_input_ids, all_attention_mask = input_features(features)
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