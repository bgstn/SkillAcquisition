import os
import re

import torch
import imageio
import numpy as np
import cv2
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt

from embedders import PhysicsLDSE2C
from simulator import PointMass
from data_loader import SeqDataset

def inference(data, model, device, plot, pm, n_simu=5):
    for i in range(n_simu):
        im_list = data["im_list"]
        x_list = data["x_list"]
        a_list = data["a_list"]
        net = model
        net.eval()
        index = np.random.randint(len(im_list) // 3) * 3
        target_frame = im_list[index+2]
        target_gt_state = x_list[index+2]
        target_state = net.posterior(torch.from_numpy(target_frame)[None,:].to(device).float()).mean
        
        Kp = 10
        
        pm.reset()
        init_frame = pm.render()
        
        pre_state = net.posterior(torch.from_numpy(init_frame)[None,:].to(device).float()).mean
        
        norm_error = []
        uv_list = []
        state_list = []
        state_u_list = []
        state_v_list = []
        frames = []
        for j in range(60):
            im = pm.render()
        
            # Get current latent state
            state_posterior = net.posterior(torch.from_numpy(im)[None,:].to(device).float())
            state = state_posterior.mean
            cov = state_posterior.variance
        
            # Proportional control in latent space
            u = Kp*(state - target_state).reshape(-1,1)
        
            # Latent space error
            ground_truth_error = np.mean((pm.x - target_gt_state)**2)
            norm_error.append(ground_truth_error)
            # True dynamics
            x = pm.true_dynamics(u.detach().cpu().numpy())
        
            # Check consistency of re-encoded/decoded state (interestingly this is quite poor...)
            st = net.posterior(net.decode(state)).mean
            print("iter {}: distance x,y,theta: {}".format(j, (target_gt_state-x).tolist()))
            
            # history record
            uv_list.append(state.detach().cpu().numpy().ravel()[:2])
            state_list.append(st.detach().cpu().numpy().ravel()[:2])

            # Visualise trajectories followed
            if plot:
                fig = plt.figure(figsize=(10, 5))
                plt.subplot(2,4,1)
                plt.cla()
                plt.imshow(im.transpose(1,2,0))
                plt.title('True state')
                plt.subplot(2,4,2)
                plt.cla()
                plt.imshow(net.decode(state)[0,:,:,:].detach().cpu().numpy().transpose(1,2,0))
                plt.title('True Decoded state')
                plt.subplot(2,4,3)
                plt.cla()
                plt.imshow(target_frame.transpose(1,2,0))
                plt.title('Target state')
                plt.subplot(2,4,4)
                plt.cla()
                plt.imshow(net.decode(target_state)[0,:,:,:].detach().cpu().numpy().transpose(1,2,0))
                plt.title('Target Decoded state')

                plt.subplot(2,2,3)
                plt.plot(np.array(uv_list)[:, 0], np.array(uv_list)[:, 1],'b+')
                plt.plot(np.array(state_list)[:, 0], np.array(state_list)[:, 1],'r+')
                plt.plot(target_state.detach().cpu().numpy().ravel()[0],target_state.detach().cpu().numpy().ravel()[1],'ko',markersize=10)
                
                plt.subplot(2,2,4)
                plt.cla()
                plt.title("Changed to GT Error")
                plt.plot(norm_error)

                fig.canvas.draw()
                mat = np.array(fig.canvas.renderer._renderer)
                frames.append(mat)
                plt.close()
                # plt.savefig("pics/pic_{}".format(j))
        if plot:
            # combine_image_to_video("pics", "pics/control_simu{}.mp4".format(i))
            imageio.mimsave("pics/control_simu{}.mp4".format(i), frames, format="mp4", fps=4)


def inference_on_coord(data, device, plot, pm, n_simu=1):
    im_list = data["im_list"]
    x_list = data["x_list"]
    a_list = data["a_list"]
    index = np.random.randint(len(im_list) // 3) * 3
    target_frame = im_list[index+2]
    target_gt_state = x_list[index+2]
    target_state = target_gt_state
    
    Kp = 5
    
    pm.x = x_list[index-100]
    
    norm_error = []
    print("cur state: ", pm.x, "target state: ", target_state)
    for i in range(n_simu):
        for j in range(50):
        
            im = pm.render()
        
            # Get current latent state
            state = pm.x 
        
            # Proportional control in latent space
            u = Kp*(target_state-state).reshape(-1,1)
        
            # Latent space error
            ground_truth_error = np.mean((pm.x - target_gt_state)**2)
            norm_error.append(ground_truth_error)
            # True dynamics
            x = pm.true_dynamics(u)
            # Check consistency of re-encoded/decoded state (interestingly this is quite poor...)
            print("iter {}: distance x,y,theta: {}".format(j, (target_state-pm.x).tolist()))

            # Visualise trajectories followed
            if plot:
                plt.subplot(2,3,1)
                plt.cla()
                plt.imshow(im[0,:,:])
                plt.title('True state')
                plt.subplot(2,3,3)
                plt.cla()
                plt.imshow(target_frame[0,:,:])
                plt.title('Target state')
                plt.subplot(2,2,3)
                
                plt.subplot(2,2,4)
                plt.cla()
                plt.title("Changed to GT Error")
                plt.plot(norm_error)
                plt.savefig("pics/pic_{}".format(j))

def inference_v1(data, model, device, plot, pm, n_simu=5, control_dir=None):
    for i in range(n_simu):
        im_list = data["im_list"]
        x_list = data["x_list"]
        a_list = data["a_list"]
        net = model
        net.eval()
        index = np.random.randint(len(im_list) // 3) * 3
        target_frame = im_list[index+2]
        target_gt_state = x_list[index+2]
        # target_state = net.posterior(torch.from_numpy(target_frame)[None,:].to(device).float()).mean
        target_state = net.transition(torch.from_numpy(target_frame)[None,:].to(device).float(), 
            torch.zeros(size=(1, 2)).to(device)).mean
        
        Kp = 0.5
        
        pm.reset()
        # pm.x[2, :] = target_gt_state[2, :]
        init_frame = pm.render()
        
        pre_state = net.posterior(torch.from_numpy(init_frame)[None,:].to(device).float()).mean
        print("inference data: ", torch.from_numpy(init_frame)[None,:].to(device).float().shape, 
        torch.zeros([0, 0, 0]).reshape(1, -1).to(device).shape)
        # pre_state = net.transition(torch.from_numpy(init_frame)[None,:].to(device).float(), 
            # torch.zeros(size=(1, 3)).to(device))
        
        norm_error = []
        uv_list = []
        state_list = []
        state_u_list = []
        state_v_list = []
        frames = []
        for j in range(60):
            im = pm.render()
        
            # Get current latent state
            # state_posterior = net.posterior(torch.from_numpy(im)[None,:].to(device).float())
            state_posterior = net.transition(torch.from_numpy(im)[None,:].to(device).float(), 
                torch.zeros(size=(1, 2)).to(device))
            state = state_posterior.mean
            cov = state_posterior.variance
        
            # Proportional control in latent space
            u = Kp*(target_state - state).reshape(-1,1)
            # u[2, :] = u[2, :] % (2*np.pi)
            # Latent space error
            ground_truth_error = np.mean((pm.x - target_gt_state)**2)
            norm_error.append(ground_truth_error)
            # True dynamics
            x = pm.true_dynamics(u.detach().cpu().numpy())
        
            # Check consistency of re-encoded/decoded state (interestingly this is quite poor...)
            # st = net.posterior(net.decode(state)).mean
            st = net.transition(net.decode(state), 
                torch.zeros(size=(1, 2)).to(device)).mean
            print("iter {}: distance x,y,theta: {}".format(j, (target_gt_state-x).tolist()))
            
            # history record
            uv_list.append(state.detach().cpu().numpy().ravel()[:2])
            state_list.append(st.detach().cpu().numpy().ravel()[:2])

            # Visualise trajectories followed
            if plot:
                fig = plt.figure(figsize=(10, 5))
                plt.subplot(2,4,1)
                plt.cla()
                plt.imshow(im.transpose(1,2,0))
                plt.title('True state')
                plt.subplot(2,4,2)
                plt.cla()
                plt.imshow(net.decode(state)[0,:,:,:].detach().cpu().numpy().transpose(1,2,0))
                plt.title('True Decoded state')
                plt.subplot(2,4,3)
                plt.cla()
                plt.imshow(target_frame.transpose(1,2,0).astype(np.float32))
                plt.title('Target state')
                plt.subplot(2,4,4)
                plt.cla()
                plt.imshow(net.decode(target_state)[0,:,:,:].detach().cpu().numpy().transpose(1,2,0))
                plt.title('Target Decoded state')
                plt.subplot(2,2,3)
                plt.plot(np.array(uv_list, dtype=np.float32)[:, 0], np.array(uv_list, np.float32)[:, 1],'b+')
                plt.plot(np.array(state_list)[:, 0], np.array(state_list)[:, 1],'r+')
                plt.plot(target_state.detach().cpu().numpy().ravel()[0],target_state.detach().cpu().numpy().ravel()[1],'ko',markersize=10)
                
                plt.subplot(2,2,4)
                plt.cla()
                plt.title("Changed to GT Error")
                plt.plot(norm_error)

                fig.canvas.draw()
                mat = np.array(fig.canvas.renderer._renderer)
                frames.append(mat)
                plt.close()
                # plt.savefig("pics/pic_{}".format(j))
        if plot:
            # combine_image_to_video("pics", "pics/control_simu{}.mp4".format(i))
            imageio.mimsave("{}/control_simu{}.mp4".format(control_dir, i), frames, format="mp4", fps=4)
