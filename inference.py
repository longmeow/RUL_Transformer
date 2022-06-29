import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import TimeSeriesDataset
from model import create_transformer
from utils import save_config, get_args, process_config, get_config_from_yaml

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda" and not torch.cuda.is_initialized():
    torch.cuda.init()


def load_model(config):
    model = create_transformer(d_model=config['d_model'],
                               nhead=config['n_head'],
                               dff=config['dff'],
                               num_layers=config['num_layers'],
                               dropout=config['dropout'],
                               l_win=config['l_win'])
    model.load_state_dict(torch.load(
        config["model_dir"] + "model__lr_{}_l_win_{}_dff_{}.pt".format(
            config['lr'], config['l_win'], config['dff'])))
    model.float()
    model.eval()
    return model


@torch.no_grad()
def main():
    start = time.perf_counter()

    try:
        args = get_args()
        config = process_config(args.config)
    except Exception as Ex:
        print(Ex)
        print("ERROR: Missing or invalid config file.")
        sys.exit(1)

    config = get_config_from_yaml(config['result_dir'] + "result_lr_{}_l_win_{}_dff_{}.yml".format(
                                  config['lr'], config['l_win'], config['dff']))

    test_data = TimeSeriesDataset(config, mode='test')
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=config['num_workers'])

    model = load_model(config)
    model.to(device)

    test_loss = 0.0
    criterion = nn.MSELoss()
    test_loss_list = list()

    with torch.no_grad():
        for x, rul in test_loader:
            out = model(x.to(device).float())
            loss = criterion(out.float(), rul.to(device).float())
            test_loss += loss.item()
            test_loss_list.append(loss.item())

    test_loss_avg = test_loss / len(test_loader)

    config['test_loss_avg'] = test_loss_avg
    config['test_loss_list_per_id'] = test_loss_list

    save_config(config['result_dir'] + "result_lr_{}_l_win_{}_dff_{}.yml".format(
        config['lr'], config['l_win'], config['dff']), config)

    print('DONE.')
    total = (time.perf_counter() - start) / 60
    print('Inference time: {}'.format(total))
    print('-----Test loss avg-----')
    print(test_loss_avg)
    print('-----Test loss list-----')
    print(test_loss_list, len(test_loss_list))


if __name__ == "__main__":
    main()
