from random import shuffle
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
    config = get_config_from_yaml('/home/longmeow/Documents/RUL_Transformer/config.yml')

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_data = TimeSeriesDataset(config)
    train_loader = DataLoader(train_data, 
                            batch_size=config['batch_size'],
                            shuffle=True, 
                            num_workers=config['num_workers'])
    
    model = create_transformer(d_model=config['d_model'],
                            nhead=config['n_head'],
                            dff=config['dff'],
                            num_layers=config['num_layers'],
                            l_win=config['l_win'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()
    trainer = ModelTrainer(model, train_loader, criterion, optimizer, device, config)

    trainer.train()

if __name__ == "__main__":
    main()