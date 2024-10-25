import os
import pathlib
import scipy.io as io

import torch
from torch.nn.functional import one_hot
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler

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
    
# class CharDataset(Dataset):
#     def __init__(self, path="/home/worker/robochars/Character_control/data/mixoutALL_shifted.mat"):
#         data = io.loadmat(path)
#         self.trajectories = data['mixout'][0,:]
#         meta = data['consts']
#         self.labels = meta[0,0][4].astype(int)
#         self.alphabet = [char[0] for char in meta[0,0][3][0]]
#         self.dt = meta[0,0][5]
        
#         self.__prep()
        
#     def __prep(self):
#         self.seq_len = max([len(tj) for tj in self.trajectories])
        
#         # pad 
#         np.pad(tj, ) for tj in self.trajectories:
            
class CharDataset(Dataset):
    
    def __init__(self, num_traj, 
                 path='/home/worker/robochars/Character_control/data/mixoutALL_shifted.mat', 
                 norm_action=True, indices=None):
        data = io.loadmat(path)
        trajectories = data['mixout'][0,:].tolist()
        meta = data['consts']
        labels = meta[0,0][4].astype(int).ravel()
        self.alphabet = np.array([char[0] for char in meta[0,0][3][0]])
        dt = meta[0,0][5]
        self.norm_action = norm_action
        self.indices = indices

        self.__prep(trajectories, labels, dt, indices)
        
    def __prep(self, trajectories, labels, dt, indices):
        max_seq_len = max([tj.shape[1] for tj in trajectories])
        # exit()
        
        tokens = []
        obs = []
        actions = []
        for i, act_seq in enumerate(trajectories):
            act_seq = np.pad(act_seq, pad_width=((0, 0), (0, max_seq_len - act_seq.shape[1])), mode="constant", constant_values=0.0)
            traj = np.cumsum(act_seq*dt,axis=-1)[:,0:-1].T
            # traj_pad = np.pad(traj, pad_width=((0, max_seq_len - traj.shape[0]), (0, 0)))
            obs.append(traj)
            
            # obs_post.append(np.cumsum(traj*dt,axis=-1)[:,1:].T)
            actions.append(act_seq[:,1:].T)
            tokens.append(labels[i]*np.ones(act_seq.shape[1]-1,dtype=int)-1)
        obs = np.stack(obs, axis=0) # num_traj x seq_len x obs_dim(28)
        actions = np.stack(actions, axis=0) # num_traj x seq_len x action_dim(3)
        tokens = np.stack(tokens, axis=0) # num_traj x seq_len
        if self.norm_action:
            actions_std = (actions - actions.min(keepdims=True)) / (actions.max(keepdims=True) - actions.min(keepdims=True))
            actions = actions_std * 2 - 1
        actions = np.concatenate([actions, np.zeros(shape=(*actions.shape[:-1], 1))], axis=-1)
            
        # assign
        self.obs = obs
        self.actions = actions
        self.tokens = tokens
        self.num_traj = len(indices) if indices is not None else obs.shape[0]
            
    def __len__(self):
        return self.num_traj
    
    def __getitem__(self, idx):
        if idx >= self.num_traj:
            raise IndexError

        if self.indices is not None:
            idx = self.indices[idx]
        
        total_cat = len(self.alphabet)
        observations = torch.from_numpy(self.obs[idx]).float()
        actions = torch.from_numpy(self.actions[idx]).float()
        goal_idx = one_hot(torch.from_numpy(self.tokens[idx].ravel()),total_cat)
        desired_goals =  torch.from_numpy(self.tokens[idx].ravel())
        
        # concatenate obs and goals
        observations = torch.concatenate([observations, goal_idx], dim=-1)
        return observations, actions, observations, desired_goals


if __name__ == "__main__":
    chards = CharDataset(num_traj=-1, norm_action=False)
    print(len(chards))
    print([el.shape for el in chards[0]])
    print("traj: ", chards[0][0][:5, :], 
          "acti: ", chards[0][1][:5, :])
    print(chards.actions.min(), chards.actions.max())
    for idx, _ in enumerate(iter(chards)):
        print("=========={}/{}===========".format(idx, len(chards) - 1), end="\r")
    print(_[0])
    print(_[1].mean(), _[1].std())
    print(chards.alphabet)
