import yaml
import os

def get_config_from_yaml(PATH):
    with open(PATH, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

def save_config(PATH, config):
    with open(PATH, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)