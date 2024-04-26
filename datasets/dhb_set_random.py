import os
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import torch
from tqdm import tqdm
#from plyfile import PlyData, PlyElement
import pandas as pd

class DHBDataset(Dataset):
    def __init__(self, root, npoints, train=False):
        super(DHBDataset, self).__init__()
        self.root = root
        self.npoints = npoints
        self.times = []
        self.train = train
        self.total_dataset = self.get_dataset()
        self.dataset = self.make_dataset()


    def get_dataset(self):
        file = self.root
        total_dataset = []
        total_tensor = torch.load(file)
        total_tuple = torch.split(tensor=total_tensor, split_size_or_sections=1, dim=0)
        total_dataset.append(total_tuple)

        return total_dataset

    def make_dataset(self):
        total_dataset = self.total_dataset
        len_dataset = len(total_dataset)
        index_lists = []
        for i in range(0, len_dataset):
            len_tuple = len(total_dataset[i])
            for j in range(0, len_tuple - 1):
                double = [i, j]
                index_lists.append(double)
        return index_lists

    def __getitem__(self, index):
        total_dataset = self.total_dataset

        index_tuple, index_pc = self.dataset[index]

        ini_pc = total_dataset[index_tuple][index_pc]
        end_pc = total_dataset[index_tuple][index_pc+1]

        ini_pc = torch.squeeze(ini_pc, dim=0).to(dtype=torch.float32)
        end_pc = torch.squeeze(end_pc, dim=0).to(dtype=torch.float32)

        ini_color = np.zeros([self.npoints, 3]).astype('float32')
        mid_color = np.zeros([self.npoints, 3]).astype('float32')
        end_color = np.zeros([self.npoints, 3]).astype('float32')

        ini_color = torch.from_numpy(ini_color).t()
        mid_color = torch.from_numpy(mid_color).t()
        end_color = torch.from_numpy(end_color).t()

        #ini_pc = ini_pc.transpose(0, 1).contiguous()
        #end_pc = end_pc.transpose(0, 1).contiguous()

        ini_color = ini_color.transpose(0, 1).contiguous()
        end_color = end_color.transpose(0, 1).contiguous()

        return ini_pc, end_pc, ini_color, end_color

    def __len__(self):
        return len(self.dataset)

class train_DHBDataset(Dataset):
    def __init__(self, root, npoints, interframes, train=True):
        super(train_DHBDataset, self).__init__()
        self.root = root
        self.npoints = npoints
        self.interframes = interframes
        self.times = []
        self.train = train
        self.total_dataset = self.get_dataset()
        self.dataset = self.make_dataset()


    def get_dataset(self):
        files = os.listdir(self.root)
        total_dataset = []
        for file in files:
            datapt = os.path.join(self.root, file)
            total_tensor = torch.load(datapt)
            total_tuple = torch.split(tensor=total_tensor, split_size_or_sections=1, dim=0)
            total_dataset.append(total_tuple)
        return total_dataset

    def make_dataset(self):
        total_dataset = self.total_dataset
        len_dataset = len(total_dataset)
        index_lists = []
        for i in range(0, len_dataset):
            len_tuple = len(total_dataset[i])
            for j in range(0, len_tuple  - self.interframes - 1):
                double = [i, j]
                index_lists.append(double)
        return index_lists

    def __getitem__(self, index):
        total_dataset = self.total_dataset
        index_tuple, index_pc = self.dataset[index]
        ini_pc = total_dataset[index_tuple][index_pc]

        i = np.random.randint(1,self.interframes+1)

        mid_pc = total_dataset[index_tuple][index_pc+i]
        end_pc = total_dataset[index_tuple][index_pc+self.interframes+1]

        ini_pc = torch.squeeze(ini_pc, dim=0).to(dtype=torch.float32)
        mid_pc = torch.squeeze(mid_pc, dim=0).to(dtype=torch.float32)
        end_pc = torch.squeeze(end_pc, dim=0).to(dtype=torch.float32)

        ini_color = np.zeros([self.npoints, 3]).astype('float32')
        mid_color = np.zeros([self.npoints, 3]).astype('float32')
        end_color = np.zeros([self.npoints, 3]).astype('float32')

        ini_color = torch.from_numpy(ini_color).t()
        mid_color = torch.from_numpy(mid_color).t()
        end_color = torch.from_numpy(end_color).t()

        ini_color = ini_color.transpose(0, 1).contiguous()
        mid_color = mid_color.transpose(0, 1).contiguous()
        end_color = end_color.transpose(0, 1).contiguous()

        T_list = [t for t in np.linspace(0.0, 1.0, num=self.interframes+2).astype('float32')]
        #T_list = T_list[1:-1]
        t = T_list[i]
        return ini_pc, mid_pc, end_pc, ini_color, mid_color, end_color, t

    def __len__(self):
        return len(self.dataset)


class test_DHBDataset(Dataset):
    def __init__(self, root, npoints, interframes,is_8ivfb, train=False):
        super(test_DHBDataset, self).__init__()
        self.root = root
        self.npoints = npoints
        self.interframes = interframes
        self.is_8ivfb = is_8ivfb
        self.times = []
        self.train = train
        self.total_dataset = self.get_dataset()
        self.dataset = self.make_dataset()

    def pc_normalize(self, pc, max_for_the_seq):
        pc = pc.numpy()
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = max_for_the_seq
        #m = np.max(np.sqrt(np.sum(pc ** 2,axis=1)))
        pc = pc / m
        return pc


    def get_dataset(self):
        file = self.root
        total_dataset = []
        total_tensor = torch.load(file)
        total_tuple = torch.split(tensor=total_tensor, split_size_or_sections=1, dim=0)
        total_dataset.append(total_tuple)

        return total_dataset

    def make_dataset(self):
        total_dataset = self.total_dataset
        len_dataset = len(total_dataset)
        index_lists = []
        for i in range(0, len_dataset):
            len_tuple = len(total_dataset[i])
            for j in range(0, len_tuple - self.interframes - 1):
                double = [i, j]
                index_lists.append(double)
        return index_lists

    def __getitem__(self, index):
        total_dataset = self.total_dataset

        index_tuple, index_pc = self.dataset[index]
        ini_pc = total_dataset[index_tuple][index_pc]

        inter_pc_list = []
        for i in range(0, self.interframes):
            mid_pc = total_dataset[index_tuple][index_pc+i+1]
            mid_pc = torch.squeeze(mid_pc, dim=0).to(dtype=torch.float32)
            if self.is_8ivfb:
                mid_pc = self.pc_normalize(mid_pc, max_for_the_seq=583.1497484423953)
                mid_pc = torch.from_numpy(mid_pc)
            inter_pc_list.append(mid_pc)

        end_pc = total_dataset[index_tuple][index_pc+self.interframes+1]

        ini_pc = torch.squeeze(ini_pc, dim=0).to(dtype=torch.float32)
        end_pc = torch.squeeze(end_pc, dim=0).to(dtype=torch.float32)

        if self.is_8ivfb:
            ini_pc = self.pc_normalize(ini_pc, max_for_the_seq=583.1497484423953)
            end_pc = self.pc_normalize(end_pc, max_for_the_seq=583.1497484423953)
            ini_pc = torch.from_numpy(ini_pc)
            end_pc = torch.from_numpy(end_pc)


        ini_color = np.zeros([self.npoints, 3]).astype('float32')
        end_color = np.zeros([self.npoints, 3]).astype('float32')

        ini_color = torch.from_numpy(ini_color).t()
        end_color = torch.from_numpy(end_color).t()

        ini_color = ini_color.transpose(0, 1).contiguous()
        end_color = end_color.transpose(0, 1).contiguous()

        color_list = []
        for i in range(0, self.interframes):
            color = np.zeros([self.npoints, 3]).astype('float32')
            color = torch.from_numpy(color).t()
            color = color.transpose(0, 1).contiguous()
            color_list.append(color)

        T_list = [t for t in np.linspace(0.0, 1.0, num=self.interframes+2).astype('float32')]
        T_list = T_list[1:-1]

        return ini_pc, inter_pc_list, end_pc, ini_color, color_list, end_color, T_list

    def __len__(self):
        return len(self.dataset)
