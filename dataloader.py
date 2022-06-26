import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, config, mode='train'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.load_dataset(config['l_win'])
        
    def __getitem__(self, idx):
        data = self.data[idx, :, :]
        label = self.label[idx, :]
        label = np.expand_dims(label, 1)
        return data, label[-1]

    def __len__(self):
        return self.data.shape[0]

    def load_dataset(self, l_win):
        if self.mode == 'train':
            data_loaded = pd.read_csv('/home/longmeow/Documents/train.csv')
        else:
            data_loaded = pd.read_csv('/home/longmeow/Documents/test.csv')
        
        def gen_rolling_windows_data(data, l_win):
            rolling_windows = data[data.columns.difference(['id', 'cycle', 'RUL'])].to_numpy()
            n_samples = data.shape[0]
            for start, stop in zip(range(0, n_samples - l_win), range(l_win, n_samples)):
                yield rolling_windows[start:stop, :]
                
        def gen_rolling_windows_labels(data, l_win):
            rolling_windows = data['RUL'].to_numpy()
            n_labels = rolling_windows.shape[0]
            for start, stop in zip(range(0, n_labels - l_win), range(l_win, n_labels)):
                yield rolling_windows[start:stop]
        
        data_generator = (list(gen_rolling_windows_data(data_loaded[data_loaded['id']==id], l_win)) 
            for id in data_loaded['id'].unique())
        data = np.concatenate(list(data_generator)).astype(np.float32)
        
        label_generator = (list(gen_rolling_windows_labels(data_loaded[data_loaded['id']==id], l_win)) 
            for id in data_loaded['id'].unique())
        labels = np.concatenate(list(label_generator)).astype(np.float32)
        
        self.data = data
        self.label = labels