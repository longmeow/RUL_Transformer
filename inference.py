import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import TimeSeriesDataset
from model import create_transformer
from trainer import ModelTrainer
from utils import get_config_from_yaml

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
                            l_win=config['l_win'])
    model.load_state_dict(torch.load(
        config["checkpoint_dir"] + "best_model.pt"))
    model.float()
    model.eval()

    return model

@torch.no_grad()
def main():
    config = get_config_from_yaml('/home/longmeow/Documents/RUL_Transformer/config.yml')
    
    test_data = TimeSeriesDataset(config, mode='test')
    test_loader = DataLoader(test_data, 
                            batch_size=config['batch_size'],
                            shuffle=False, 
                            num_workers=config['num_workers'])
    
    model = load_model(config)
    model.to(device)
    
    test_loss = 0.0
    pred_list = list()
    criterion = nn.MSELoss()
    
    #deviation_list = list()

    for idx, (x, rul) in enumerate(test_loader):
        out = model(x.to(device).float())
        loss = criterion(out.float(), rul.float())
        test_loss += loss.item()
        pred_list.append(out.item())

    test_loss_avg = test_loss / len(test_loader)
    print(test_loss_avg)

    test_loss_avg = np.array(test_loss_avg)
    pred_list = np.array(pred_list)
    infer_dict = {'predict_rul': pred_list}
    np.savez('result.npz', **infer_dict)
    print('DONE.')   

if __name__ == "__main__":
    main() 
