import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import TimeSeriesDataset
from model import create_transformer
from trainer import ModelTrainer
from utils import get_config_from_yaml, save_config

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
        config["checkpoint_dir"] + "model__lr_{}_l_win_{}_dff_{}.pt".format(
                    config['lr'], config['l_win'], config['dff'])))
    model.float()
    model.eval()
    return model

@torch.no_grad()
def main():
    start = time.perf_counter()
    _config = get_config_from_yaml('/home/longmeow/Documents/RUL_Transformer/config.yml')
    config = get_config_from_yaml('{}'.format(_config['result_dir']) + "result_lr_{}_l_win_{}_dff_{}.yml".format(
        _config['lr'], _config['l_win'], _config['dff']))
    
    test_data = TimeSeriesDataset(config, mode='test')
    test_loader = DataLoader(test_data, 
                            batch_size=1,
                            shuffle=False, 
                            num_workers=config['num_workers'])
    
    model = load_model(config)
    model.to(device)
    
    test_loss = 0.0
    criterion = nn.MSELoss()
    test_lost_list = list()

    with torch.no_grad():
        for x, rul in test_loader:
            out = model(x.to(device).float())
            loss = criterion(out.float(), rul.float())
            test_loss += loss.item()
            test_lost_list.append(loss.item())

    test_loss_avg = test_loss / len(test_loader)
    
    config['test_loss_avg'] = test_loss_avg
    config['test_loss_list_per_id'] = test_lost_list
    
    save_config(config['result_dir'] + "result_lr_{}_l_win_{}_dff_{}.yml".format(
        config['lr'], config['l_win'], config['dff']), config)
 
    print('DONE.')
    total = (time.perf_counter() - start) / 60
    print('Inference time: {}'.format(total))   
    print('-----Test lost avg-----')
    print(test_loss_avg)
    print('-----Test lost list-----')
    print(test_lost_list, len(test_lost_list))
    
if __name__ == "__main__":
    main() 
