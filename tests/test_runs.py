import GPUtil
import os
import pytest

from train import parse_args as train_parse_args, main as train_main
from mask import parse_args as mask_parse_args, main as mask_main
from retrain import parse_args as retrain_parse_args, main as retrain_main
from compare import parse_args as comapre_parse_args, main as compare_main
from evaluate import parse_args as evaluate_parse_args, main as evaluate_main


@pytest.fixture(scope="session", autouse=True)
def set_gpu_device(pytestconfig):
    """Set the GPU device for each test process, if they are available.
    """

    # Get the first available GPU that has a low load and memory usage
    max_memory_ratio = 100 / (40 * 1024)  # 100MB out of 40GB
    max_load_ratio = 0.1
    available_gpus = GPUtil.getAvailable(order='first',
                                         limit=1,
                                         maxLoad=max_load_ratio,
                                         maxMemory=max_memory_ratio)
    gpu = available_gpus[0] if len(available_gpus) > 0 else -1
    print(f"Using GPU: {gpu}")

    # Set up the CUDA_VISIBLE_DEVICES environment variable
    os.environ[f'CUDA_VISIBLE_DEVICES'] = str(gpu)


def test_train_vis_debug():
    cli_args = ['cifar10', '0', '--debug']
    args = train_parse_args(cli_args)
    train_main(args)


def test_train_nlp_debug():
    cli_args = ['cola', '0', '--debug']
    args = train_parse_args(cli_args)
    train_main(args)

def test_train_mlm_debug():
    cli_args = ['mlm', '0', '--debug']
    args = train_parse_args(cli_args)
    train_main(args)


def test_mask_vis_debug():
    cli_args = ['pets', '1', '--debug']
    args = mask_parse_args(cli_args)
    mask_main(args)


def test_mask_nlp_debug():
    cli_args = ['sst-2', '2', '--debug']
    args = mask_parse_args(cli_args)
    mask_main(args)

def test_mask_mlm_debug():
    cli_args = ['mlm', '2', '--debug']
    args = mask_parse_args(cli_args)
    mask_main(args)


def test_retrain_vis_debug():
    cli_args = ['flowers', '3', '--debug']
    args = retrain_parse_args(cli_args)
    retrain_main(args)


def test_retrain_nlp_debug():
    cli_args = ['wnli', '4', '--debug']
    args = retrain_parse_args(cli_args)
    retrain_main(args)

def test_retrain_mlm_debug():
    cli_args = ['mlm', '3', '--debug']
    args = retrain_parse_args(cli_args)
    retrain_main(args)

def test_compare_vis_debug():
    cli_args = ['cifar100', '0', 'food', '4', '--debug']
    args = comapre_parse_args(cli_args)
    compare_main(args)


def test_compare_nlp_debug():
    cli_args = ['cola', '1', 'wnli', '4', '--debug']
    args = comapre_parse_args(cli_args)
    compare_main(args)

def test_compare_mlm_debug():
    cli_args = ['mnli', '1', 'mlm', '4', '--debug']
    args = comapre_parse_args(cli_args)
    compare_main(args)

def test_evaluate_vis_debug():
    cli_args = ['pets', '0', '--debug']
    args = evaluate_parse_args(cli_args)
    evaluate_main(args)


def test_evaluate_nlp_debug():
    cli_args = ['qqp', '1', '--debug']
    args = evaluate_parse_args(cli_args)
    evaluate_main(args)

def test_evaluate_mlm_debug():
    cli_args = ['mlm', '1', '--debug']
    args = evaluate_parse_args(cli_args)
    evaluate_main(args)