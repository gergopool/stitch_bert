import argparse
import os
import torch
import glob
import pickle
import pandas as pd

from src.static import Logger, TASKS


def process_evaluations(in_dir: str, out_dir: str):

    Logger.info(f"Summarizing evaluation results from {in_dir}...")
    filepaths = glob.glob(os.path.join(in_dir, "*.pkl"))
    df = pd.DataFrame(columns=['task', 'seed', 'orig', 'masked', 'retrained', 'mask_sparsity'])
    for filepath in filepaths:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        task, seed = os.path.basename(filepath)[:-len('.pkl')].split('_')
        results['task'] = task
        results['type'] = 'vis' if task in TASKS['vis'] else 'nlp'
        results['seed'] = int(seed)
        new_row = pd.DataFrame([results])
        df = pd.concat([df, new_row], ignore_index=True)
    save_path = os.path.join(out_dir, 'evaluation.csv')
    df.to_csv(save_path, index=False)
    Logger.info(f"Summary saved to {save_path}.")


def process_comparison(in_dir: str, out_dir: str):

    Logger.info(f"Summarizing comparison results from {in_dir}...")
    filepaths = glob.glob(os.path.join(in_dir, "*.pkl"))
    df = pd.DataFrame(columns=['task1', 'seed1', 'task2', 'seed2', 'layer', 'jaccard', 'cka', 'fs'])
    for filepath in filepaths:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
            print(filepath)
            for key in ['jaccard', 'cka', 'fs']:
                if isinstance(results[key][0], torch.Tensor):
                    results[key] = [x.item() for x in results[key]]

        task1, seed1, task2, seed2 = os.path.basename(filepath)[:-len('.pkl')].split('_')
        results['layer'] = [i for i in range(len(results['cka']))]
        results['task1'] = [task1] * len(results['cka'])
        results['seed1'] = [int(seed1)] * len(results['cka'])
        results['task2'] = [task2] * len(results['cka'])
        results['seed2'] = [int(seed2)] * len(results['cka'])
        new_rows = pd.DataFrame(results)
        df = pd.concat([df, new_rows], ignore_index=True)
    save_path = os.path.join(out_dir, 'comparison.csv')
    df.to_csv(save_path, index=False)
    Logger.info(f"Summary saved to {save_path}.")


def main(args):

    Logger.initialise(debug=False)

    # Ensure directories exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Process results
    process_evaluations(args.eval_dir, args.out_dir)
    process_comparison(args.compare_dir, args.out_dir)


def parse_args(cli_args=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir",
                        type=str,
                        default='results/evaluate/',
                        help="Directory which contains the evaluation results. " + \
                             "Default is 'results/evaluate/'.")
    parser.add_argument("--out_dir",
                        type=str,
                        default='results/summary/',
                        help="Directory to save the output. Default is './results/evaluate'.")
    parser.add_argument(
        "--compare_dir",
        type=str,
        default='results/compare/',
        help="Directory which contains the comparisons. Default is 'results/compare'.")

    # Parse the arguments
    args = parser.parse_args(cli_args)

    return args


if __name__ == "__main__":

    args = parse_args()

    # Execute the main function
    main(args)
