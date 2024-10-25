import os
import sys
import time
import traceback

import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import SubsetRandomSampler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from torch.optim.lr_scheduler import CosineAnnealingLR

from embedders import PhysicsLDSE2C_Simplified
from embedders import PhysicsLDSE2C_Seq
from embedders import BCMDN
from embedders import BC
from embedders import BCSDN
from embedders import BC_SEQ
from embedders import MDN_SEQ
from simulator import PointMass, PointMass_v1, generate_seq_data, ColorStick, Triangular, PointMass_Maze
from data_loader import SeqDataset, SeqDataset_v2
from data_loader import FetchPushDataset
from data_loader import CharDataset
from inference import inference, inference_v1
from logger import logger
from config import parse_config

# LOSS_KEYS = ["loss", "image_loss", "action_loss", "tagging_loss", "encoder_loss", "action_mse"]
def frange_cycle_sigmoid(start, stop, n_epoch, n_cycle=4, ratio=0.5):
    L = np.ones(n_epoch)
    period = n_epoch/n_cycle
    step = (stop-start)/(period*ratio) # step is in [0,1]
    
    # transform into [-6, 6] for plots: v*12.-6.

    for c in range(n_cycle):

        v , i = start , 0
        while v <= stop:
            L[int(i+c*period)] = 1.0/(1.0+ np.exp(- (v*12.-6.)))
            v += step
            i += 1
    return L  

def train(net, epochs, device, dl, val_dl, optimizer, cfg):
    """
    train NewtonVAE
    """
    logger.info(net)
    ori_net = net
    # if not config["logger"].getboolean("debugging"):
    #     net = torch.compile(net, mode="max-autotune")
    kl_weights = frange_cycle_sigmoid(0.0, 1.0, epochs, n_cycle=100)
    tagging_loss_weight = float(cfg["trainer"]["taggloss_threshold"])
    # losses_recon = []
    # losses_trans = []
    losses = {}
    eval_losses = {}
    best_eval_loss = np.inf
    for epoch in range(epochs):
        net.train()
        # batch_losses_recon = []
        # batch_losses_trans = []
        batch_losses = {}
        for idx, (im,act,x,g) in enumerate(dl):
            im = im.to(device)
            act = act.to(device)
            x = x.to(device)
            g = g.to(device)
            try:
                out = net({'img':im,'act':act,'xgt':x, "goal": g},train=True)
                lls = net.compute_losses({'img':im,'act':act,'xgt':x, "goal": g}, out, epoch=epoch, kl_weight=kl_weights[epoch])
            
                optimizer.zero_grad()
                
                # Anealing trans kl down seems to help get a decent latent space, otherwise I see some seed dependency
                # and an interesting inversion of the latent space -> the boundaries are near zero, with the space radiating outwards.
                # Would be interesting to explore how to avoid this - clearly finding a globally smooth space by local smooth losses,
                # can allow for this, potentially room for a better objective.
                # loss = (1-epoch/epochs)*lls['trans_pos_kl']+epoch/epochs*lls['next_rec'] + 1e-8 * lls["regularizer"]
                # loss = lls['trans_pos_kl']+lls['next_rec'] + 1e-8 * lls["regularizer"]
                # loss = lls["loss"]
                # tagging_loss = torch.maximum(lls["tagging_loss"], 
                #     torch.ones_like(lls["tagging_loss"], requires_grad=False) * tagging_loss_weight)
                # loss = lls["image_loss"] + lls["action_loss"] + lls["encoder_loss"] + 0 * lls["tagging_loss"] + lls["goal_loss"]
                loss = lls["loss"]
                # loss = (1-epoch/epochs)*lls['trans_pos_kl']+epoch/epochs*lls['next_rec']
                # loss = lls['trans_pos_kl']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 100)
                optimizer.step()
            except Exception as e:
                logger.info("training error")
                logger.error(traceback.format_exc())
                torch.save({"model": ori_net, 
                            "optim": optimizer.state_dict(), 
                            'img':im,'act':act,'xgt':x}, 
                           "{}/model_diagnosis.pt".format(logger.model_dir)
                           )
                raise
            # print("iter {}/{} epoch {}/{}: loss: {} loss_rec: {} loss_trans: {}".format(
            #     idx, len(dl), epoch, epochs, loss.item(), lls['next_rec'].item(), 
            #     lls['trans_pos_kl'].item()))
            for loss_key in lls.keys():
                batch_losses[loss_key] = batch_losses.get(loss_key, []) + [round(lls[loss_key].item(),4)]
            
            # batch_losses_recon.append(lls['next_rec'].item())
            # batch_losses_trans.append(lls['image_loss'].item())
            # logger.info(
            #     "iter {}/{} epoch {}/{}: loss: {:.4f}, im_loss: {:.4f}, action_loss: {:.4f}, action_mse: {:.4f}, encoder_loss: {:.4f}, tag_loss: {:.4f}".format(
            #         idx, len(dl), epoch, epochs, 
            #         lls["loss"].item(),
            #         lls["image_loss"].item(),
            #         lls["action_loss"].item(),
            #         lls["action_mse"].item(),
            #         lls["encoder_loss"].item(),
            #         lls["tagging_loss"].item(),
            #         # lls["goal_loss"].item()
            #         )
            #     )
            
            loss_str = " ".join(["{}:{:.4f}".format(k, v.item()) for k, v in lls.items()])
            logger.info(
                "iter {}/{} epoch {}/{}: {}".format(
                    idx, len(dl), epoch, epochs, loss_str
                )
            )
        logger.info(out.get("transition_mat", None))
        logger.info(out["goal_idx_seq"][:, 0].tolist())
        
        # losses_recon.append(np.mean(batch_losses_recon))
        # losses_trans.append(np.mean(batch_losses_trans))
        for k, v in batch_losses.items():
            losses[k] = losses.get(k, []) + [np.mean(v)]
        
        eval_loss = eval(net, epochs, device, val_dl, optimizer, cfg, eval_losses)
        if best_eval_loss > eval_loss.get("action_loss")[-1]:
            best_eval_loss = eval_loss.get("action_loss")[-1]
            torch.save(ori_net, "{}/cur_best_model.pt".format(logger.model_dir))
        
        # Visualisation
        n_plots = len(losses) + 5 * 2
        n_cols = 6
        n_rows = int(n_plots / n_cols) + 1
        fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols)
        
        for idx, (lls_key, lls_value) in enumerate(losses.items()):
            row = idx // n_cols
            col = idx % n_cols
            plt.subplot(gs[row, col])
            plt.plot(lls_value, label="train_" + lls_key)
            plt.plot(eval_loss[lls_key], label="eval_" + lls_key)
            plt.grid(True)
            plt.legend()
            plt.title("{} loss, Epoch: {}".format(lls_key, epoch))

        if cfg["network"].get("model_type") in ["VAE", "VAE_SEQ"]:
            plt.subplot(gs[-1, 4:])
            rand_time_step = int(np.random.choice(im.shape[1]))
            im_act = im[0][rand_time_step].detach().cpu()
            predicted_im = out['image_dist'].mean[0, 0][rand_time_step].detach().cpu()
            x_pos = np.arange(predicted_im.shape[0])
            # plt.imshow(np.hstack((im_act, predicted_im)))
            # plt.bar(x_pos, im_act[:net.obs_dim], width=0.5, label="actual")
            # plt.bar(x_pos + 0.5, predicted_im, width=0.5, label="predicted")
            # plt.legend()
            # plt.title("image reconstruction")
            
            # ax = plt.subplot(gs[1, 4:], projection="3d")
            support = out['next_pos_posterior'].mean.reshape(-1, net.state_dim).detach().cpu().numpy()
            # cols = x[:, :, :3].reshape(-1, 1, 3).squeeze(dim=-1).detach().cpu().numpy()
            # cl = np.hstack((cols[:,-1],np.zeros((cols.shape[0],1))))/64
            # ax.scatter(support[:,0],support[:,1],support[:,2], c=cl,alpha=0.1,s=25)
            goal_mu = net.goal_mu.cpu().detach().numpy()
            # ax.scatter(goal_mu[:, 0], goal_mu[:, 1], goal_mu[:, 2], marker="^", s=100, c="red")
            
            
            cmap = plt.cm.tab10
            ax = plt.subplot(gs[-1, 2:4], projection="3d")
            segmentation_index = torch.argmax(
                out["goal_switching_posterior_probs"], dim=-1) \
                .detach().cpu().numpy()[0]
            
            ax.scatter(support[:,0], support[:,1], support[:,2],
                c=segmentation_index.reshape(-1),alpha=0.1, s=25, cmap=cmap)
            sc = ax.scatter(support.reshape(*x.shape[:2], net.state_dim)[0, :, 0],
                        support.reshape(*x.shape[:2], net.state_dim)[0, :, 1],
                        support.reshape(*x.shape[:2], net.state_dim)[0, :, 2],
                        c=segmentation_index[0, :],alpha=0.8, marker="x", s=50)
            cbar = plt.colorbar(sc, ticks=sorted(set(range(net.n_skill))))
            cbar.set_label('skill index')
            plt.title("latent space view without goals")

            ax = plt.subplot(gs[-1, :2], projection="3d")
            # segmentation_index = torch.argmax(
            #     out["goal_switching_posterior_probs"], dim=-1) \
            #     .detach().cpu().numpy()[0]
            
            if cfg["network"].get("model_type") == "VAE":
                ax.scatter(support[:,0], support[:,1], support[:,2],
                    c=segmentation_index.reshape(-1),alpha=0.1, s=25)
                sc = ax.scatter(support.reshape(*x.shape[:2], net.state_dim)[0, :, 0],
                            support.reshape(*x.shape[:2], net.state_dim)[0, :, 1],
                            support.reshape(*x.shape[:2], net.state_dim)[0, :, 2],
                            c=segmentation_index[0, :],alpha=0.8, marker="x", s=50)
                ax.scatter(goal_mu[:, 0], goal_mu[:, 1], goal_mu[:, 2], marker="^", s=100, c="red")
                cbar = plt.colorbar(sc, ticks=sorted(set(range(net.n_skill))))
                cbar.set_label('skill index')
                plt.title("latent space color by segmentations")
                
                ax = plt.subplot(gs[-2, 4:], projection="3d")
                ax.scatter(x[:, :, 0].reshape(-1).detach().cpu().numpy(),
                            x[:, :, 1].reshape(-1).detach().cpu().numpy(),
                            x[:, :, 2].reshape(-1).detach().cpu().numpy(),
                            c=segmentation_index[:, :],alpha=0.1, s=25)
                sc = ax.scatter(x.squeeze(dim=-1)[0, :, 0].detach().cpu().numpy(),
                    x.squeeze(dim=-1)[0, :, 1].detach().cpu().numpy(),
                    x.squeeze(dim=-1)[0, :, 2].detach().cpu().numpy(),
                    c=segmentation_index[0, :],alpha=0.8, marker="x", s=50)
                cbar = plt.colorbar(sc, ticks=sorted(set(range(net.n_skill))))
                cbar.set_label('skill index')
                plt.title("true space color by segmentations")
                
                plt.subplot(gs[-2, 2:4])
                for i in range(segmentation_index.shape[0]):
                    d = segmentation_index[i, ].tolist()
                    plt.plot(torch.arange(len(d)), d, alpha=0.3)
            else:
                ax = plt.subplot(gs[-2, 4:])
                ax.scatter(x[:, :, 0].reshape(-1).detach().cpu().numpy(),
                            x[:, :, 1].reshape(-1).detach().cpu().numpy(),
                            # x[:, :, 2].reshape(-1).detach().cpu().numpy(),
                            c=segmentation_index[:, :],alpha=0.1, s=25)
                sc = ax.scatter(x.squeeze(dim=-1)[0, :, 0].detach().cpu().numpy(),
                    x.squeeze(dim=-1)[0, :, 1].detach().cpu().numpy(),
                    # x.squeeze(dim=-1)[0, :, 2].detach().cpu().numpy(),
                    c=segmentation_index[0, :],alpha=0.8, marker="x", s=50)
                char_idx = g[0, 0]
                char_tgt = dl.dataset.alphabet[char_idx]
                cbar = plt.colorbar(sc, ticks=sorted(set(range(net.n_skill))))
                cbar.set_label('skill index')
                plt.title("true space color by segmentations LETTER: {}".format(char_tgt))
                
                plt.subplot(gs[-2, 2:4])
                for i in range(segmentation_index.shape[0]):
                    d = segmentation_index[i, ].tolist()
                    plt.plot(torch.arange(len(d)), d, alpha=0.3)
                    
                plt.subplot(gs[-2, 1])
                action = net.get_action(output=out)[0].detach()
                traj = torch.cumsum(action, dim=1).cpu().numpy()
                for i in range(traj.shape[0]):
                    plt.plot(traj[i, :, 0], traj[i, :, 1], alpha=0.1)
                plt.plot(traj[0, :, 0], traj[0, :, 1])
                plt.scatter(traj[0, :, 0], traj[0, :, 1], c=traj[0, :, 2], marker="x")
                plt.title("traj recovered from predicted actions")
                
                plt.subplot(gs[-2, 0])
                action = act
                traj = torch.cumsum(action, dim=1).cpu().numpy()
                for i in range(traj.shape[0]):
                    plt.plot(traj[i, :, 0], traj[i, :, 1], alpha=0.1)
                plt.plot(traj[0, :, 0], traj[0, :, 1])
                plt.scatter(traj[0, :, 0], traj[0, :, 1], c=traj[0, :, 2], marker="x")
                plt.title("traj recovered from actual actions")
        
        if cfg["network"].get("model_type") in ["BC_SEQ", "MDN_SEQ"]:
            plt.subplot(gs[-2, 1])
            action = net.get_action(None, None, output=out)[0].detach()
            traj = torch.cumsum(action, dim=1).cpu().numpy()
            for i in range(traj.shape[0]):
                plt.plot(traj[i, :, 0], traj[i, :, 1], alpha=0.1)
            plt.plot(traj[0, :, 0], traj[0, :, 1])
            plt.scatter(traj[0, :, 0], traj[0, :, 1], c=traj[0, :, 2], marker="x")
            plt.title("traj recovered from predicted actions")
            
            plt.subplot(gs[-2, 0])
            action = act
            traj = torch.cumsum(action, dim=1).cpu().numpy()
            for i in range(traj.shape[0]):
                plt.plot(traj[i, :, 0], traj[i, :, 1], alpha=0.1)
            plt.plot(traj[0, :, 0], traj[0, :, 1])
            plt.scatter(traj[0, :, 0], traj[0, :, 1], c=traj[0, :, 2], marker="x")
            plt.title("traj recovered from actual actions")
                
        # Adjust layout for better spacing
        plt.tight_layout()
        
        # save figure
        plt.savefig("{}/pic_epoch{}.png".format(logger.pic_dir, epoch))
        plt.close()
        torch.save(ori_net, "{}/model_checkpoint.pt".format(logger.model_dir))
        if (epoch + 1) in [0, 10, 50, 100, 300, 500]:
            torch.save(ori_net, "{}/model_checkpoint_{}.pt".format(logger.model_dir, epoch))
    return net, ori_net

def eval(net, epochs, device, dl, optimizer, cfg, losses):
    """
    evaluate NewtonVAE
    """
    net.eval()
    tagging_loss_weight = float(cfg["trainer"]["taggloss_threshold"])
    # losses_recon = []
    # losses_trans = []
    batch_losses = {}
    for idx, (im,act,x,g) in enumerate(dl):
        im = im.to(device)
        act = act.to(device)
        x = x.to(device)
        g = g.to(device)
        try:
            out = net({'img':im,'act':act,'xgt':x, "goal": g},train=True)
            lls = net.compute_losses({'img':im,'act':act,'xgt':x, "goal": g}, out, epoch=1)

        except Exception as e:
            logger.info("evaluation error")
            logger.error(traceback.format_exc())
            torch.save({"model": net, 
                        "optim": optimizer.state_dict(), 
                        'img':im,'act':act,'xgt':x}, 
                        "{}/model_diagnosis.pt".format(logger.model_dir)
                        )
            raise
        # print("iter {}/{} epoch {}/{}: loss: {} loss_rec: {} loss_trans: {}".format(
        #     idx, len(dl), epoch, epochs, loss.item(), lls['next_rec'].item(), 
        #     lls['trans_pos_kl'].item()))
        for loss_key in lls.keys():
            batch_losses[loss_key] = batch_losses.get(loss_key, []) + [round(lls[loss_key].item(),4)]
            

    for k, v in batch_losses.items():
        losses[k] = losses.get(k, []) + [np.mean(v)]
        
    loss_str = " ".join(["{}:{:.4f}".format(k, v[-1]) for k, v in losses.items()])
    logger.info(
        "EVAL LOSS: {}".format(
            loss_str
        )
    )
    return losses

def main(logger):
    logger.log_file()
    logger.log_config(config)

    seq_len = int(config["simulator"]["seq_len"])
    Ntrajs = int(config["simulator"]["Ntrajs"])
    dt=float(config["simulator"]["dt"])
    size=float(config["simulator"]["size"])
    pm = PointMass_Maze(dt=dt, size=size)
    fix = None
    # im_list, x_list, a_list, g_list =  generate_seq_data(seq_len, Ntrajs, pm, fix)
    # print("generating data done")
    # print("seq len range: ", min([x.shape[0] for x in im_list]),
    #       max([x.shape[0] for x in im_list])
    # )

    # img_shape = im_list[0].shape
    action_dim = int(config["network"]["action_dim"])
    state_dim = int(config["network"]["state_dim"])
    obs_dim = int(config["network"]["obs_dim"])
    batch_size = int(config["network"]["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(config["network"]["epochs"])
    # model_path = "./model_20230802_044830.pt"
    # model_path = "./model_20230811_101308.pt"
    model_path = config["network"].get("model_path")
    train_flag = config["network"].getboolean("train_flag")
    save_fig = config["network"].getboolean("save_fig")

    # training_data = SeqDataset_v2(im_list,x_list,a_list,g_list, seq_len,device)
    # Assuming `dataset` is your PyTorch Dataset
    dataset_size = Ntrajs
    indices = list(range(dataset_size))
    split = 100
    random_seed = 42
    # Shuffle the indices
    torch.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.random.default_generator.manual_seed(random_seed)
    # Split the indices
    train_indices, val_indices = indices[split:], indices[:split]

    # Define samplers
    training_data = CharDataset(num_traj=Ntrajs, indices=train_indices)
    val_data = CharDataset(num_traj=Ntrajs, indices=val_indices)
    if not config["logger"].getboolean("debugging"):
        dl = DataLoader(training_data, batch_size=batch_size, shuffle=True, 
                        num_workers=8, prefetch_factor=1, persistent_workers=True)
    else:
        dl = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    im, act, x, g = next(iter(dl))
    print("train data", im.shape, act.shape)

    # from embedders import PhysicsLDSE2C_Simplified
    # from embedders import BCMDN
    # from embedders import BC
    # from embedders import BCSDN
    if config["network"].get("model_type") == "VAE":
        model_class = PhysicsLDSE2C_Simplified
    elif config["network"].get("model_type") == "MDN":
        model_class = BCMDN
    elif config["network"].get("model_type") == "SDN":
        model_class = BCSDN
    elif config["network"].get("model_type") == "BC":
        model_class = BC
    elif config["network"].get("model_type") == "VAE_SEQ":
        model_class = PhysicsLDSE2C_Seq
    elif config["network"].get("model_type") == "BC_SEQ":
        model_class = BC_SEQ
    elif config["network"].get("model_type") == "MDN_SEQ":
        model_class = MDN_SEQ
    else:
        raise NotImplementedError
    net = model_class(None, action_dim, state_dim, obs_dim,
                      simu=int(config["network"]["n_simu"]),
                      n_skill=int(config["network"]["n_skill"]),
            ).to(device)
    # optimizer = torch.optim.Adam(net.parameters(),
    #                              lr=float(config["optimizer"]["lr"]),
    #                              weight_decay=float(config["optimizer"]["weight_decay"]),
    #                             )
    
    optimizer = torch.optim.AdamW(net.parameters(),
                                 lr=float(config["optimizer"]["lr"]),
                                 weight_decay=float(config["optimizer"]["weight_decay"]),
                                 amsgrad=True
                                )

    if 'None' not in model_path:
        print("loading model from: {}".format(model_path))
        net = torch.load(model_path, map_location=device)
        # net.load_state_dict(weight)
        
    pytorch_total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info("model total trainable params: {}".format(pytorch_total_trainable_params))
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    logger.info("model total params: {}".format(pytorch_total_params))

    if train_flag:
        net.train()
        cnet, ori_net = train(net, epochs, device, dl, val_dl, optimizer, config)
        path = "{}/model_final.pt".format(logger.model_dir)
        torch.save(ori_net, path)
        logger.info("model saves to {}".format(path))
    # logger.info("inference ...")
    # inference_v1({"im_list": im_list, "x_list": x_list, "a_list": a_list}, net, device, plot=save_fig, 
    #              pm=pm, n_simu=10, control_dir=logger.control_dir)
    # from analysis import plot_latent_state_in_graph, plot_control_annimation
    # plot_latent_state_in_graph(net, pm, [im_list, x_list, a_list, g_list], "all", slices=None)
    # plot_latent_state_in_graph(net, pm, [im_list, x_list, a_list, g_list], "0_9", slices=[0,9])
    # plot_latent_state_in_graph(net, pm, [im_list, x_list, a_list, g_list], "9_18", slices=[9, 18])
    # plot_latent_state_in_graph(net, pm, [im_list, x_list, a_list, g_list], "18_30", slices=[18, 30])
    # plot_control_annimation(net, pm, [im_list, x_list, a_list, g_list], filename="action", slices=None)


if __name__ == "__main__":
    try:
        # message = sys.argv[1]
        config, config_str = parse_config()
        logger = logger(config)
        logger.info(config["experiment"]["message"])
        logger.info("configurations: \n {}".format(config_str))
        main(logger)
    except Exception as e:
        msg = traceback.format_exc()
        print(msg)
        logger.error(msg)


