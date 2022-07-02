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
            label = self.label[idx]
            return data, label
        else:
            data = self.data[idx, :, :]
            label = self.label[idx]
            return data, label

    def __len__(self):
        return self.data.shape[0]

    def load_dataset(self, config):
        if self.mode == 'train':
            train_df = pd.read_csv(config['data_path'] + "train.csv")

            def gen_sequence(id_df, seq_length, seq_cols):
                data_array = id_df[seq_cols].values
                num_elements = data_array.shape[0]
                for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
                    yield data_array[start:stop, :]

            sensor_cols = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
            sequence_cols = ['setting1', 'setting2']
            sequence_cols.extend(sensor_cols)
            # generator for the sequences
            seq_gen = (list(gen_sequence(train_df[train_df['id']==id], self.config['l_win'], sequence_cols)) 
                      for id in train_df['id'].unique())

            # generate sequences and convert to numpy array
            seq_array = np.concatenate(list(seq_gen)).astype(np.float32)

            # function to generate labels
            def gen_labels(id_df, seq_length, label):
                data_array = id_df[label].values
                num_elements = data_array.shape[0]
                return data_array[seq_length:num_elements, :]

            # generate labels
            label_gen = [gen_labels(train_df[train_df['id']==id], self.config['l_win'], ['RUL']) 
                        for id in train_df['id'].unique()]
            label_array = np.concatenate(label_gen).astype(np.float32)

            self.data = seq_array
            self.label = label_array

        else:
            test_df = pd.read_csv(config['data_path'] + "test.csv")

            sensor_cols = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
            sequence_cols = ['setting1', 'setting2']
            sequence_cols.extend(sensor_cols)

            seq_array_test_last = [test_df[test_df['id']==id][sequence_cols].values[-config['l_win']:] 
                                  for id in test_df['id'].unique() if len(test_df[test_df['id']==id]) >= config['l_win']]

            seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)


            y_mask = [len(test_df[test_df['id']==id]) >= config['l_win'] for id in test_df['id'].unique()]

            label_array_test_last = test_df.groupby('id')['RUL'].nth(-1)[y_mask].values
            label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)

            self.data = seq_array_test_last
            self.label = label_array_test_last
