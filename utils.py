import yaml
import os

def get_config_from_yaml(PATH):
    with open(PATH, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict