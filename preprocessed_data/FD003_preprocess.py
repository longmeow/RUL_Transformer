import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Read dataset
train_df = pd.read_csv('/home/filtestbed/Documents/RUL_Transformer/RUL_raw_data/FD003/train_FD003.txt', sep=" ", header=None)
test_df = pd.read_csv('/home/filtestbed/Documents/RUL_Transformer/RUL_raw_data/FD003/test_FD003.txt', sep=" ", header=None)
truth_df = pd.read_csv('/home/filtestbed/Documents/RUL_Transformer/RUL_raw_data/FD003/RUL_FD003.txt', sep=" ", header=None)

#Trainset
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
train_df = train_df.sort_values(['id','cycle'])

rul_train = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul_train.columns = ['id', 'max']
train_df = train_df.merge(rul_train, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)

#Testset
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']

rul_test = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul_test.columns = ['id', 'max']

truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul_test['max'] + truth_df['more']
truth_df.drop('more', axis=1, inplace=True)
test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)

# Normalize train_df
cols_normalize = train_df.columns.difference(['id','cycle','RUL'])
min_max_scaler = MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                             columns=cols_normalize, 
                             index=train_df.index)
join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)

# Normalize test_df
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                            columns=cols_normalize, 
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns = test_df.columns)
test_df = test_df.reset_index(drop=True)

#Drop unique cols
uniq_value_cols = (train_df.nunique() == 1) & (test_df.nunique() == 1)

train_df.drop(columns=train_df.columns[uniq_value_cols], inplace=True)
test_df.drop(columns=test_df.columns[uniq_value_cols], inplace=True)

# train_df.drop(columns='s6', inplace=True)
# test_df.drop(columns='s6', inplace=True)

#Save preprocessed dataset as .csv 
train_df.to_csv('train_003.csv', encoding='utf-8', index=False)
test_df.to_csv('test_003.csv', encoding='utf-8', index=False)