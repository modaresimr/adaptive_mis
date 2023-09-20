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
    parser.add_argument('--model', help='Path to model config')
    parser.add_argument('--main', default="./configs/common.yaml", help='Path to other config')
    parser.add_argument('--eval', default="./configs/evaluation/split.yaml", help='Path to evaluation config')
    parser.add_argument('--comet', default=False, help='Log to comet')
    args = parser.parse_args()

    main_config = load_config(f"!include {args.main}")
    date = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log = f'{basename(args.dataset)}/{basename(args.model)}-{date}'
    data = f"""
model: !include {args.model}
dataset: !include {args.dataset}
evaluation: !include {args.eval}
run:
    comet: {args.comet}
    save_dir: ./results/{log}
    """
    config = load_config(data)
    main_config.update(config)
    pipeline.execute(main_config)
