from __future__ import print_function
import torch.utils.data as data
import scipy.io
import numpy as np


class MNIST(data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        self.train_num = int(2375)
        data_0 = scipy.io.loadmat('MNIST/BDGP91.mat')
        data_dict = dict(data_0)
        data_1 = data_dict['X']

        self.data1 = data_1[0][0].astype(np.float32)
        self.data2 = data_1[0][1].astype(np.float32)
        print(self.data1.shape)
        print(self.data2.shape)

    def __getitem__(self, index):
        img_traina, img_trainb = self.data1[index, :], self.data2[index, :]
        return img_traina, img_trainb

    def __len__(self):
        return self.train_num



