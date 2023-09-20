import argparse
import configparser
from . import pipeline
from adaptive_mis.common.config import load_config
from datetime import datetime
import os


def basename(path):
    return os.path.basename(path).split('.')[0]


def main():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Path to dataset config')
    parser.add_argument('--save_dir',default="auto", help='Save Results in')
    parser.add_argument('--model', help='Path to model config')
    parser.add_argument('--main', default="./configs/common.yaml", help='Path to other config')
    parser.add_argument('--eval', default="./configs/evaluation/split.yaml", help='Path to evaluation config')
    parser.add_argument('--comet', default=True, help='Log to comet')
    args = parser.parse_args()
    
    main_config = load_config(f"!include {args.main}")
    save_dir=args.save_dir
    if save_dir=="auto":
        date = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir = f'./results/{basename(args.dataset)}/{basename(args.model)}_{date}'
    data = f"""
model: !include {args.model}
dataset: !include {args.dataset}
evaluation: !include {args.eval}
run:
    comet: {args.comet}
    save_dir: {save_dir}
    """
    config = load_config(data)
    main_config.update(config)
    pipeline.execute(main_config)
