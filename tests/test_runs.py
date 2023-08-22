import subprocess


def test_train_vis_debug():
    command = "CUDA_VISIBLE_DEVICES=-1 python train.py cifar10 0 --debug"
    result = subprocess.run(command, shell=True)
    assert result.returncode == 0


def test_train_nlp_debug():
    command = "CUDA_VISIBLE_DEVICES=-1 python train.py cola 0 --debug"
    result = subprocess.run(command, shell=True)
    assert result.returncode == 0


def test_mask_vis_debug():
    command = "CUDA_VISIBLE_DEVICES=-1 python mask.py pets 1 --debug"
    result = subprocess.run(command, shell=True)
    assert result.returncode == 0


def test_mask_nlp_debug():
    command = "CUDA_VISIBLE_DEVICES=-1 python mask.py sst-2 2 --debug"
    result = subprocess.run(command, shell=True)
    assert result.returncode == 0


def test_retrain_vis_debug():
    command = "CUDA_VISIBLE_DEVICES=-1 python retrain.py flowers 3 --debug"
    result = subprocess.run(command, shell=True)
    assert result.returncode == 0


def test_retrain_nlp_debug():
    command = "CUDA_VISIBLE_DEVICES=-1 python retrain.py wnli 4 --debug"
    result = subprocess.run(command, shell=True)
    assert result.returncode == 0


def test_compare_vis_debug():
    command = "CUDA_VISIBLE_DEVICES=-1 python compare.py cifar100 0 food 4 --debug"
    result = subprocess.run(command, shell=True)
    assert result.returncode == 0


def test_compare_nlp_debug():
    command = "CUDA_VISIBLE_DEVICES=-1 python compare.py cola 1 wnli 4 --debug"
    result = subprocess.run(command, shell=True)
    assert result.returncode == 0