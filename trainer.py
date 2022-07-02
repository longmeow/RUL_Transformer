import torch
from copy import deepcopy

class ModelTrainer():
    def __init__(self, model, train_data, criterion, optimizer, device, config):
        self.model = model
        self.train_data = train_data
        self.device = device
        self.config = config
        self.train_loss_list = list()
        self.min_loss = float('inf')
        self.best_model = None
        self.best_optimizer = None
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self, epoch):
        train_loss = 0.0
        self.model.train()
        for x, rul in self.train_data:
            self.model.zero_grad()
            out = self.model(x.to(self.device).float())
            loss = torch.sqrt(self.criterion(out.float(), rul.to(self.device).float()))
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

        for epoch in range(1, self.config['n_epochs'] + 1):
            self.train_epoch(epoch)

        self.config['train_loss_list'] = self.train_loss_list

        torch.save(self.best_model, self.config["model_dir"] + "model__lr_{}_l_win_{}_dff_{}.pt".format(
                   self.config['lr'], self.config['l_win'], self.config['dff']))

    def update_config(self):
        return self.config
