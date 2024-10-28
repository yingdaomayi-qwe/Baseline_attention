import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class Dataset_attention(Dataset):
    def __init__(self, root_path, S_TF=10,K=10,flag='train'):
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        # self.percent = percent
        self.root_path = root_path
        self.__read_data__()
        self.K=K
        self.S_TF=S_TF


    def __read_data__(self):

        bs_attenna_numbers=[64]
        speeds = np.arange(5, 25, 5)
        self.lenspeed=len(speeds)
        file_range=30

        gain_matrices=[]
        for i in range(file_range):
            for bs_attenna_number in bs_attenna_numbers:
                for speed in speeds:
                    # print(i)
                    self.data_path_gain = f'ODE_dataset_v_{speed}/normal_gain_a{bs_attenna_number}_v{speed}_{i}.csv'
                    whole_path_gain = os.path.join(self.root_path, self.data_path_gain)
                    df_raw_gain = pd.read_csv(whole_path_gain)
                    df_raw_gain = df_raw_gain.to_numpy()
                    df_raw_gain = df_raw_gain.reshape(256, 101, -1)
                    gain_matrices.append(df_raw_gain)

        gain = np.vstack(gain_matrices)
        border1s=np.array([0,256*8*len(speeds)*len(bs_attenna_numbers),256*9*len(speeds)*len(bs_attenna_numbers)])*(file_range//10)
        border2s = np.array(
            [256*8*len(speeds)*len(bs_attenna_numbers), 256 * 9 * len(speeds) * len(bs_attenna_numbers), 256 * 10 * len(speeds) * len(bs_attenna_numbers)]) * (
                               file_range // 10)
        border1= border1s[self.set_type]
        border2=border2s[self.set_type]
        self.data_x = gain[border1:border2]
        self.data_y = gain[border1:border2]
    def __getitem__(self, index):
        if index<0 or index>len(self.data_x):
            raise IndexError('Index out of range')
        seq_x=self.data_x[index,0:-1:self.K]
        seq_y=self.data_y[index,(self.S_TF-1)*self.K+1:self.S_TF*self.K]
        return seq_x,seq_y


    def __len__(self):
        return len(self.data_x)

