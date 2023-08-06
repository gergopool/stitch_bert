# Analysing similarity of task-specific subnetworks

## Environment

First, install torch with cuda, e.g. through conda. Then `pip install -r requirements.txt`

## Train model

Training was tested only with a single-gpu setup:
```
python train.py [cola|mnli|mrpc|qnli|qqp|rte|sst-2|sts-b|wnli] --seed [YOUR_SEED]
```

## Structure mask model

This is a deterministic algorithm, seed is only provided to pick up the model trained on
the given seed.
```
python mask.py [cola|mnli|mrpc|qnli|qqp|rte|sst-2|sts-b|wnli] --seed [YOUR_SEED]
```