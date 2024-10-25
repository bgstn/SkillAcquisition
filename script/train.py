import os
import sys
import time
import traceback

import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.optim.lr_scheduler import CosineAnnealingLR

from embedders import PhysicsLDSE2C_Simplified
from embedders import BCMDN
from embedders import BC
from embedders import BCSDN
from simulator import PointMass, PointMass_v1, generate_seq_data, ColorStick, Triangular, PointMass_Maze
from data_loader import SeqDataset, SeqDataset_v2
from data_loader import FetchPushDataset
from inference import inference, inference_v1
from logger import logger
from config import parse_config

def train(net, epochs, device, dl, val_dl, optimizer, cfg):
    """
    train NewtonVAE
    """
    logger.info(net)
    ori_net = net
    if not config["logger"].getboolean("debugging"):
        net = torch.compile(net, mode="max-autotune")
    kl_weights = 1.0 
    tagging_loss_weight = float(cfg["trainer"]["taggloss_threshold"])
    losses = {}
    eval_losses = {}
    best_eval_loss = np.inf
    for epoch in range(epochs):
        net.train()
        for idx, (im,act,x,g) in enumerate(dl):
            im = im.to(device)
            act = act.to(device)
            x = x.to(device)
            g = g.to(device)
            try:
                out = net({'img':im,'act':act,'xgt':x, "goal": g},train=True)
                lls = net.compute_losses({'img':im,'act':act,'xgt':x, "goal": g}, out, epoch=epoch, kl_weight=kl_weights[epoch])
            
                optimizer.zero_grad()
                
                loss = lls["loss"]
                
                loss.backward()
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
            for loss_key in lls.keys():
                batch_losses[loss_key] = batch_losses.get(loss_key, []) + [round(lls[loss_key].item(),4)]
            
            loss_str = " ".join(["{}:{:.4f}".format(k, v.item()) for k, v in lls.items()])
            logger.info(
                "iter {}/{} epoch {}/{}: {}".format(
                    idx, len(dl), epoch, epochs, loss_str
                )
            )
        logger.info(out.get("transition_mat", None))
        logger.info(out["goal_idx_seq"][:, 0].tolist())
        
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

        if cfg["network"].get("model_type") == "VAE":
            plt.subplot(gs[-1, 4:])
            rand_time_step = int(np.random.choice(im.shape[1]))
            im_act = im[0][rand_time_step].detach().cpu()
            predicted_im = out['image_dist'].mean[0, 0][rand_time_step].detach().cpu()
            x_pos = np.arange(predicted_im.shape[0])
            # plt.imshow(np.hstack((im_act, predicted_im)))
            plt.bar(x_pos, im_act[:net.obs_dim], width=0.5, label="actual")
            plt.bar(x_pos + 0.5, predicted_im, width=0.5, label="predicted")
            plt.legend()
            plt.title("image reconstruction")
            
            support = out['next_pos_posterior'].mean.reshape(-1, net.state_dim).detach().cpu().numpy()
            goal_mu = net.goal_mu.cpu().detach().numpy()
            
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
                plt.plot(torch.arange(50), segmentation_index[i, ].tolist(), alpha=0.3)
                
        # Adjust layout for better spacing
        plt.tight_layout()
        
        # save figure
        plt.savefig("{}/pic_epoch{}.png".format(logger.pic_dir, epoch))
        plt.close()
        torch.save(ori_net, "{}/model_checkpoint.pt".format(logger.model_dir))
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
        
    loss_str = " ".join(["{}:{:.4f}".format(k, v.item()) for k, v in lls.items()])
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

    action_dim = int(config["network"]["action_dim"])
    state_dim = int(config["network"]["state_dim"])
    obs_dim = int(config["network"]["obs_dim"])
    batch_size = int(config["network"]["batch_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(config["network"]["epochs"])
    model_path = config["network"].get("model_path")
    train_flag = config["network"].getboolean("train_flag")
    save_fig = config["network"].getboolean("save_fig")

    training_data = FetchPushDataset(num_traj=Ntrajs)
    val_data = FetchPushDataset(path="/home/worker/fetchpush/fetchpush_data/20240412_165500_all_random_testset")
    dl = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=8, prefetch_factor=1, persistent_workers=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    im,act,x, g = next(iter(dl))
    print("train data", im.shape, act.shape)

    if config["network"].get("model_type") == "VAE":
        model_class = PhysicsLDSE2C_Simplified
    elif config["network"].get("model_type") == "MDN":
        model_class = BCMDN
    elif config["network"].get("model_type") == "SDN":
        model_class = BCSDN
    elif config["network"].get("model_type") == "BC":
        model_class = BC
    else:
        raise NotImplementedError
    net = model_class(None, action_dim, state_dim, obs_dim,
                      simu=int(config["network"]["n_simu"]),
                      n_skill=int(config["network"]["n_skill"]),
            ).to(device)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=float(config["optimizer"]["lr"]),
                                 weight_decay=float(config["optimizer"]["weight_decay"])
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


