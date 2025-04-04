import argparse
from pathlib import Path
import json
import yaml


def load_config(default_config="/home/cwatzenboeck/code/RA/ra_utils/runs/config_landmarks/train_landmarks_01.yaml",
                debugging_in_jupyter_nb=False, silencium=False):

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Read in the filename of the config file (JSON or YAML).")
        parser.add_argument('--config', type=str,
                            help=f"Path to the config file. Supported formats: .json, .yaml, .yml\n default = ({default_config})",
                            default=default_config  # for debugging
                            )
        return parser.parse_args()

    if debugging_in_jupyter_nb:
        config_file = Path(default_config)
        if not silencium:
            print("Debugging in Jupyter notebook")
    else:
        args = parse_args()
        config_file = Path(args.config)

    # Check if the file exists
    if not config_file.exists():
        raise FileNotFoundError(f"The file {config_file} does not exist.")

    # Determine file type based on extension and load accordingly
    if config_file.suffix.lower() == '.json':
        with open(config_file, 'r') as f:
            config = json.load(f)
    elif config_file.suffix.lower() in ['.yaml', '.yml']:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file extension '{config_file.suffix}'. Supported extensions are .json, .yaml, .yml.")

    if not silencium:
        print("CONFIG:", config_file)
        from pprint import pprint
        pprint(config)

    return config