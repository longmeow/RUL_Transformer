import yaml

def get_config_from_yaml(yaml_file_paths):
    with open(yaml_file_paths, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict