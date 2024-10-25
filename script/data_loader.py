import os
import pathlib

import torch 
import numpy as np
import cv2

from embedders import PhysicsLDSE2C
from torch.utils.data import DataLoader,Dataset

class SeqDataset(Dataset):
    
    def __init__(self, im_list,x_list,a_list,seq_len=3,device ='cuda', width=64):
        
        self.device = device
        self.x = np.vstack(x_list).reshape(-1, seq_len, 2)
        self.ims = np.vstack(im_list).reshape(-1,seq_len, 3, width, width)
        self.acts = np.vstack(a_list).reshape(-1,seq_len,2)

    def __len__(self):
        return self.ims.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.ims[idx]), torch.tensor(self.acts[idx]), torch.tensor(self.x[idx])
    
class SeqDataset_v2(Dataset):
    def __init__(self, im_data, x_data, a_data, g_data, *args):
        self.im_data =  im_data
        self.x_data = x_data
        self.a_data = a_data
        self.g_data = g_data
        # self.pairwise_data = self.prc_pairwise()
        self.len, self.indices_list = self.get_len()
        self.seq_len = x_data[0].shape[0] - 1
    
    def get_len(self):
        tot_len = 0
        indices_list = []
        for im, x, a, g in zip(self.im_data, self.x_data, self.a_data, self.g_data):
            shape = im.shape
            tot_len += shape[0] - 2 - (2 - 1) # shape - items put first in the sequence - history tuples for modelling - 1
            indices_list.append(tot_len)
        return tot_len, indices_list
        
    def __len__(self):
        return len(self.im_data)
    
    def __getitem__(self, index):
        # traj_index = np.searchsorted(self.indices_list, index)
        # seq_index = index - self.indices_list[traj_index]
        # print(traj_index)
        # print(seq_index)
        # slices = np.array([seq_index+2, seq_index+3])
        im_data = self.im_data[index]
        x_data = self.x_data[index]
        a_data = self.a_data[index]
        g_data = self.g_data[index]
        # im_data = self.pairwise_data[0]
        # x_data = self.pairwise_data[1]
        # a_data = self.pairwise_data[2]
        return torch.tensor(im_data), torch.tensor(a_data), torch.tensor(x_data), torch.tensor(g_data)
    
class FetchPushDataset(Dataset):
    def __init__(self, num_traj=-1, path="/home/worker/fetchpush/fetchpush_data/20240215_065606_all_random"):
        self.path = pathlib.Path(path)
        self.num_traj = num_traj
        self._prep()
    
    def _prep(self):
        files_path = self.path.rglob("*npy")
        self.files_path = [file_path for file_path in files_path if file_path.stat().st_size < 30000]            
        self.length = len(self.files_path) if self.num_traj < 0 else self.num_traj
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError
        file_path = self.files_path[index]
        traj_data = np.load(file_path.resolve(), allow_pickle=True).tolist()
        traj_observations  = traj_data.get("obs", None) 
        traj_actions = traj_data.get("action", None)
        assert traj_observations is not None or traj_actions is not None, "data format is not correct"
        
        achieved_goals = []
        desired_goals = []
        observations = []
        actions = []
        for obs_data, act in zip(traj_observations, traj_actions):
            achieved_goals.append(obs_data["achieved_goal"])
            desired_goals.append(obs_data["desired_goal"])
            observations.append(obs_data["observation"])
            actions.append(act)
        achieved_goals = np.concatenate(achieved_goals, axis=0).astype(np.float32)
        desired_goals = np.concatenate(desired_goals, axis=0).astype(np.float32)
        observations = np.concatenate(observations, axis=0).astype(np.float32)[:, :-1]
        actions = np.concatenate(actions, axis=0).astype(np.float32)
        # print(observations.shape)
        observations = np.concatenate([observations, desired_goals], axis=-1)
        # print(observations.shape)
        return observations, actions, observations, desired_goals

if __name__ == "__main__":
    # from simulator import generate_seq_data
    # from simulator import PointMass_Maze
    import matplotlib.pyplot as plt
    # pm = PointMass_Maze(size=4)
    # trajectory_data = generate_seq_data(100, 313, pm)
    # ds = SeqDataset_v2(*trajectory_data)
    # for idx, (im, a, x, g) in enumerate(ds):
    #     print(idx, end="\r")
    #     if idx == 0:
    #         print(im.shape)
    #         plt.imshow(im.sum(dim=0).detach().cpu().numpy().transpose(1,2,0))
    #         plt.savefig("test.png")
    #         for i in range(5):
    #             plt.imshow(im[i].detach().cpu().numpy().transpose(1,2,0))
    #             plt.savefig("test{}.png".format(i))
    
    plt.figure(figsize=(20, 5))
    
    fpd = FetchPushDataset(num_traj=-1)
    n_count = np.inf
    act_arr = []
    for idx, (obs, act, state, goals) in enumerate(fpd):
        print("=================={}/{}==================".format(idx, len(fpd)), end="\r")
        # print(act)
        # act_arr.append(act)
        # print(obs.shape, act.shape, state.shape, goals.shape)
        # plt.subplot(1, 4, 1)
        # plt.hist(act[:, 0], density=True, bins=30)
        
        # plt.subplot(1, 4, 2)
        # plt.hist(act[:, 1], density=True, bins=30)
        
        # plt.subplot(1, 4, 3)
        # plt.hist(act[:, 2], density=True, bins=30)
        
        # plt.subplot(1, 4, 4)
        # plt.hist(act[:, 3], density=True, bins=30)
        
        if idx > n_count:
            break
    exit()
    plt.subplot(1, 4, 1)
    act_arr = np.concatenate(act_arr, axis=0)
    plt.hist(act_arr[:, 0], density=True, bins=30)
    
    plt.subplot(1, 4, 2)
    plt.hist(act_arr[:, 1], density=True, bins=30)
    
    plt.subplot(1, 4, 3)
    plt.hist(act_arr[:, 2], density=True, bins=30)
    
    plt.subplot(1, 4, 4)
    plt.hist(act_arr[:, 3], density=True, bins=30)
    plt.savefig("test.png")
        
