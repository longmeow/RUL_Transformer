from logging import critical
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

class ModelTrainer():
    def __init__(self, model, train_data, optimizer, device, config):
        self.model = model
        self.train_data = train_data
        self.device = device
        self.config = config
        self.train_loss_list = list()
        self.min_loss = float('inf')
        self.best_model = None
        self.best_optimizer = None
        self.optimizer = optimizer

    def train_epoch(self, criterion, epoch):
        train_loss = 0.0
        self.model.train()
        for idx, (x, rul) in enumerate(self.train_data):
            self.model.zero_grad()
            out = self.model(x.to(self.device).float())
            loss = criterion(out.float(), rul.float())
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / len(self.train_data)
        self.train_loss_list.append(train_loss)

        if train_loss < self.min_loss:
            self.min_loss = train_loss
            self.best_model = deepcopy(self.model.state_dict())
            self.best_optimizer = deepcopy(self.optimizer.state_dict())
            self.best_epoch_in_round = epoch
    
    def train(self):
        self.model.to(self.device)
        criterion = nn.MSELoss()

        for epoch in range(1, self.config['n_epochs']+1):
            self.train_epoch(criterion, epoch)

