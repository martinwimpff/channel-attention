from argparse import ArgumentParser
import os
from pathlib import Path
import yaml

from channel_attention.train_test import train_and_test

CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "configs")
DEFAULT_CONFIG = "bcic2a_eegnet.yaml"


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    # load config
    with open(os.path.join(CONFIG_DIR, args.config)) as f:
        config = yaml.safe_load(f)

    train_and_test(config)
