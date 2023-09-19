import argparse
import configparser
from . import pipeline
from adaptive_mis.common.config import load_config


def main():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Path to dataset config')
    parser.add_argument('--model', help='Path to model config')
    parser.add_argument('--main', default=".\configs\common.yaml", help='Path to other config')
    args = parser.parse_args()

    main_config = load_config(f"!include {args.main}")

    data = f"""
model: !include {args.model}
dataset: !include {args.dataset}
    """
    config = load_config(data)
    main_config.update(config)

    pipeline.execute(main_config)
