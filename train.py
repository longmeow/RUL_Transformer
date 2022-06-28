import sys
import time
import yaml
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import TimeSeriesDataset
from model import create_transformer
from trainer import ModelTrainer
from utils import get_config_from_yaml

torch.manual_seed(42)

def main():
    start = time.perf_counter()
    config = get_config_from_yaml('/home/longmeow/Documents/RUL_Transformer/config.yml')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = TimeSeriesDataset(config, mode='train')
    train_loader = DataLoader(train_data, 
                            batch_size=config['batch_size'],
                            shuffle=True, 
                            num_workers=config['num_workers'])
    
    model = create_transformer(d_model=config['d_model'],
                            nhead=config['n_head'],
                            dff=config['dff'],
                            num_layers=config['num_layers'],
                            dropout=config['dropout'],
                            l_win=config['l_win'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config['weight_decay'])
    criterion = nn.MSELoss()
    trainer = ModelTrainer(model, train_loader, criterion, optimizer, device, config)

    trainer.train()
    print('DONE.') 
    total = (time.perf_counter() - start) / 60
    print(total)  
if __name__ == "__main__":
    main()