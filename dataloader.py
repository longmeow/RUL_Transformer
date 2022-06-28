import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.mode = mode
        self.load_dataset(config)

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self.data[idx, :, :]
            label = self.label[idx, :]
            label = np.expand_dims(label, 1)
            return data, label[-1]
        else:
            data = self.data[idx, :, :]
            label = self.label[idx]
            return data, label

    def __len__(self):
        return self.data.shape[0]

    def load_dataset(self, config):
        if self.mode == 'train':
            data_loaded = pd.read_csv('/home/longmeow/Documents/train.csv')
            
            def gen_rolling_windows_data(data):
                rolling_windows = data[data.columns.difference(['id', 'cycle', 'RUL'])].to_numpy()
                n_samples = data.shape[0]
                for start, stop in zip(range(0, n_samples - config['l_win']), range(config['l_win'], n_samples)):
                    yield rolling_windows[start:stop, :]
            
            def gen_rolling_windows_labels(data):
                rolling_windows = data['RUL'].to_numpy()
                n_labels = rolling_windows.shape[0]
                for start, stop in zip(range(0, n_labels - config['l_win']), range(config['l_win'], n_labels)):
                    yield rolling_windows[start:stop]

            data_generator = (list(gen_rolling_windows_data(data_loaded[data_loaded['id']==id])) 
                    for id in data_loaded['id'].unique())
            data = np.concatenate(list(data_generator)).astype(np.float32)
        
            label_generator = (list(gen_rolling_windows_labels(data_loaded[data_loaded['id']==id])) 
                    for id in data_loaded['id'].unique())
            label = np.concatenate(list(label_generator)).astype(np.float32)

            self.data = data
            self.label = label
            
        else:
            data_loaded = pd.read_csv('/home/longmeow/Documents/test.csv')
            
            last_rolling_window_test = [data_loaded[data_loaded.columns.difference(['id', 'cycle', 'RUL'])].to_numpy()[-config['l_win']:] 
                       for id in data_loaded['id'].unique() if len(data_loaded[data_loaded['id']==id]) >= config['l_win']]
            last_rolling_window_test = np.asarray(last_rolling_window_test).astype(np.float32)
            
            y_mask = [len(data_loaded[data_loaded['id']==id]) >= config['l_win'] for id in data_loaded['id'].unique()]
            last_label_test = data_loaded.groupby('id')['RUL'].nth(-1)[y_mask].to_numpy()
            last_label_test = last_label_test.reshape(last_label_test.shape[0],1).astype(np.float32)
            
            self.data = last_rolling_window_test
            self.label = last_label_test
