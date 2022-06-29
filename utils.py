import os
import yaml
import argparse


def get_config_from_yaml(PATH):
    with open(PATH, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def save_config(PATH, config):
    with open(PATH, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-c', '--config',
                           metavar='C',
                           default='None',
                           help='YAML configuration file')
    args = argparser.parse_args()
    return args


def create_dirs(*dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def process_config(yaml_file):
    config = get_config_from_yaml(yaml_file)
    save_dir = f"experiments/{config['experiment']}/"
    config["result_dir"] = os.path.abspath(os.path.join(save_dir, "results/")) + "/"
    config["model_dir"] = os.path.abspath(os.path.join(save_dir, "models/")) + "/"
    config["data_path"] = os.path.abspath("preprocessed_data") + "/"
    return config
