import os
import logging

import numpy as np
from abc import abstractmethod

import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.kl import kl_divergence

device = tc.device("cuda" if tc.cuda.is_available() else "cpu")
logger = logging.getLogger("tc")

class MLPDecoder(nn.Module):
    def __init__(self, z_dim, n_block=3):
        super().__init__()
        self.state_dim = z_dim
        self.n_block = n_block
        self._prep()
        
    def _prep(self):
        self.blocks = nn.ModuleList()
        input_dims = self.state_dim
        for _ in range(self.n_block):
            block = nn.Sequential(
                nn.Linear(28, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.Linear(64, 28),
                nn.BatchNorm1d(28)
            )
            self.blocks.append(block)
        self.input_norm = nn.BatchNorm1d(input_dims)
        self.output_layer = nn.Linear(input_dims, 28)
    
    def forward(self, x):
        x_norm = self.input_norm(x)
        x_norm = self.output_layer(x_norm)
        
        prev_x = x_norm
        for block in self.blocks:
            x_norm = block(prev_x)
            prev_x = prev_x + x_norm
        return prev_x

class PhysicsLDSE2C_Simplified(nn.Module):
    def __init__(self, img_shape, action_dim, state_dim, obs_dim,
                 rank="diag", dt=0.1, steps=1, window=1, kl_reg=1.0, divider=1, decoder="ha", simu=1, n_skill=5):
        super(PhysicsLDSE2C_Simplified, self).__init__()
        self.act_dim = action_dim
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.img_shape = img_shape
        self.window = window
        self.n_skill = n_skill
        self.backbone_nn = nn.Sequential(
                            nn.BatchNorm1d(obs_dim),
                            nn.Linear(obs_dim, 256),
                            nn.Tanh(),
                            nn.Linear(256, 256),
                            nn.Tanh()
                            )
        
        self.encoder = nn.Sequential(
            self.backbone_nn,
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2*state_dim),
        )
        
        self.prir_goal_switcher = nn.Sequential(
            nn.Linear(state_dim, n_skill)
        )
        
        self.action_mu_layer = nn.Sequential(
            nn.Linear(state_dim, action_dim * n_skill)
        )
        self.action_std_layer = nn.Sequential(
            nn.Linear(state_dim, action_dim * n_skill)
        )
        
        self.decoder = MLPDecoder(z_dim=state_dim)
        
        self.simu = simu
        self.log_image_std = nn.Parameter(tc.ones(size=(1, obs_dim)) * 1.6, requires_grad=True)

        self.gain = nn.Parameter(tc.randn(size=(n_skill, action_dim * state_dim)), requires_grad=True)
        
        self.states_prior_mu = nn.Parameter(tc.zeros(size=(1, 1, state_dim)), requires_grad=True)
        self.states_prior_logstd = nn.Parameter(tc.ones(size=(1, 1, state_dim)), requires_grad=True)
        self.goal_switching_prior_unnormalized = nn.Parameter(tc.ones(size=(1, n_skill)), requires_grad=False)
        self.goal_mu = nn.Parameter(tc.randn(size=(n_skill, state_dim)), requires_grad=True)
        
    def init_weights(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                tc.nn.init.xavier_normal_(module.weight)
            
    def goal_switching_prior(self, states):
        simu_sample, batch_size, seq_len, feat_dim = states.shape
        states = states.reshape(simu_sample * batch_size * seq_len, feat_dim)
        res = self.prir_goal_switcher(states)
        res = res.reshape(simu_sample, batch_size, seq_len, self.n_skill)
        return res

    def posterior(self, im):
        """
        encode image to latent
        """
        batch_size, seq_len, feat_dim = im.shape
        im = im.reshape(batch_size * seq_len, feat_dim)
        res = self.encoder(im)
        mu, logsigma = tc.split(res, [self.state_dim, self.state_dim], dim=1)
        mu = mu.reshape(batch_size, seq_len, self.state_dim)
        logsigma = logsigma.reshape(batch_size, seq_len, self.state_dim)
        sigma = F.softplus(logsigma) + 1e-4
        return Normal(mu, sigma)
    
    def decode_dist(self, z):
        """
        decode latent to image distribution
        """
        simu_sample, batch_size, seq_len, feat_dim = z.shape
        z = z.reshape(simu_sample*batch_size*seq_len, feat_dim)
        res = self.decoder(z)
        mu = res
        mu = mu.view(simu_sample, batch_size, seq_len, self.obs_dim)
        logsigma = self.log_image_std
        sigma = F.softplus(logsigma) + 1e-4
        return Normal(mu, sigma)
    
    def decode(self, z):
        """
        decode latent to image
        """
        res = self.decoder(z)
        mu = res
        return mu
    
    def cal_transition(self, batch):
        # Compute the transition counts
        transition_counts = (tc.matmul(batch[:, :-1].transpose(1, 2), batch[:, 1:]) + 1).sum(dim=0)

        # Normalize the transition matrix to get transition probabilities
        transition_matrix = transition_counts / transition_counts.sum(dim=1, keepdim=True)
        return transition_matrix.float()
    
    def action_posterior(self, gain, goals, states_mu):
        sh = states_mu.shape
        action = self.action_mu_layer(states_mu.reshape(-1, self.state_dim))
        action = F.tanh(action)
        action = action.reshape(*sh[:3], self.n_skill, self.act_dim)
        
        action_sigma = self.action_std_layer(states_mu)
        action_sigma = F.softplus(action_sigma) + 1e-4
        action_sigma = action_sigma.reshape(*action.shape)
        return Independent(Normal(action, action_sigma), 1)
    
    def safe_kl_div(self, p, q, min_value=1e-10):
        p = tc.clamp(p, min=min_value)
        q = tc.clamp(q, min=min_value)
        return tc.sum(p * tc.log(p / q), dim=-1)

    def forward(self, input_data, train=False):
        whole_seq = input_data["img"]
        batch_size, seq_len, obs_dim = whole_seq.shape
        encoder_input = input_data["img"]
        
        states_prior = Normal(self.states_prior_mu, F.softplus(self.states_prior_logstd))
        states = self.posterior(encoder_input) # (batch_size x seq_len) x state_dim
        states_sample = states.rsample(sample_shape=(self.simu, ))
        
        goal_switching_prior_logits = self.goal_switching_prior(states_sample) # logits: batch_size x seq_len x n_skill
        goal_switching_prior = Categorical(logits=goal_switching_prior_logits)
        
        # goal posterior
        goal_mu = self.goal_mu.view(1, 1, 1, self.n_skill, self.state_dim) \
            .expand(self.simu, batch_size, seq_len, self.n_skill, self.state_dim)
        batch_gain = self.gain.view(1, 1, 1, -1, self.act_dim, self.state_dim) # batch_size x seq_len-1 x n_skill x 1
        action_dist = self.action_posterior(
            batch_gain, goal_mu, 
            states_sample.view(self.simu, batch_size, seq_len, 1, self.state_dim),
        ) # batch_size x seq_len - 1 x n_skill x state_dim
        act_t = input_data["act"].squeeze(dim=-1)[:, :, :-1].view(1, batch_size, seq_len, 1, self.act_dim)
        posterior_probs_unnormalized = F.log_softmax(goal_switching_prior_logits, dim=-1) + \
            action_dist.log_prob(act_t) # batch_size x seq_len-1 x n_skill
        goal_switching_posterior_log_probs = posterior_probs_unnormalized - posterior_probs_unnormalized.logsumexp(dim=-1, keepdim=True)
        goal_switching_posterior_probs = F.softmax(goal_switching_posterior_log_probs, dim=-1)
        goal_onehot_idx = goal_switching_posterior_probs
        
        goal_seq = goal_mu
        
        image_dist = self.decode_dist(states_sample)
        transition_mat = self.cal_transition(goal_onehot_idx.reshape(-1, seq_len, self.n_skill).float())
        return {"latent_action_t": action_dist, "next_pos_posterior": states, 
                "goal_switching_posterior_probs": goal_switching_posterior_probs, "goal_onehot_idx": goal_onehot_idx,
                "goal_idx_seq": tc.argmax(goal_onehot_idx, dim=-1),
                "goal_seq": goal_seq,
                "states_prior": states_prior,
                "image_dist": image_dist,
                "goal_switching_prior": goal_switching_prior,
                "batch_gain": batch_gain,
                "transition_mat": transition_mat
                }

    def compute_losses(self, input_data, output, epoch, **kwargs):
        batch_size, seq_len, obs_dim = input_data["img"].shape
        im_seq = input_data["img"].unsqueeze(dim=0)
        act_t = input_data["act"].squeeze(dim=-1)[:, :, :-1].view(1, batch_size, seq_len, self.act_dim)
        
        # Reconstruction loss
        # action loss
        goal_switching_posterior = Categorical(probs=output["goal_switching_posterior_probs"])

        action_compo = output["latent_action_t"]
        mixture = MixtureSameFamily(
            mixture_distribution=output["goal_switching_prior"],
            component_distribution=action_compo
            )
        action_loss = -1  * mixture.log_prob(act_t).mean()
        action_mse = (((output["latent_action_t"].mean - act_t.unsqueeze(dim=-2))**2).sum(dim=-1) * output["goal_onehot_idx"]).mean()
        image_loss = -1 * output["image_dist"].log_prob(im_seq).mean()
        
        # encoder loss
        encoder_loss = kl_divergence(output["next_pos_posterior"], output["states_prior"]).mean()
        
        # tagging loss
        tagging_loss = self.safe_kl_div(goal_switching_posterior.probs.detach(), output["goal_switching_prior"].probs).mean()
        
        # diversity loss
        div_prob = output["goal_switching_prior"].probs.mean(dim=[0, 1, 2])
        diversity_loss_prior = -1 * Categorical(probs = div_prob).entropy()
        div_prob = goal_switching_posterior.probs.mean(dim=[0, 1, 2])
        diversity_loss_posterior = -1 * Categorical(probs = div_prob).entropy()

        label_loss = -F.softmax(output["goal_switching_posterior_probs"], dim=-1).log().sum(dim=-1).mean() * 1/self.n_skill
        
        weight = 1.0
        loss = image_loss + action_loss + weight * (encoder_loss + tagging_loss)
        return {"loss": loss, 
                "next_rec": action_loss, "trans_pos_kl": action_loss, 
                "action_loss": action_loss, "label_loss": label_loss, 
                "action_mse": action_mse, "encoder_loss": encoder_loss,
                "tagging_loss": tagging_loss, "image_loss": image_loss, 
                "diversity_loss_prior": diversity_loss_prior, "diversity_loss_posterior": diversity_loss_posterior
                }

class BC(nn.Module):
    def __init__(self, img_shape, action_dim, state_dim, 
                 rank="diag", dt=0.1, steps=1, window=1, kl_reg=1.0, divider=1, decoder="ha", simu=1, n_skill=9):
        super(BC, self).__init__()
        self.act_dim = action_dim
        self.state_dim = state_dim
        self.img_shape = img_shape
        # self.image_pixel_num = np.prod(img_shape)
        self.window = window
        self.n_skill = n_skill
        # self.encoder = nn.Sequential(
        #                     MLPEncoder(28),
        #                     nn.Linear(28, 3)
        #                     )
        self.backbone_nn = nn.Sequential(
                    # MLPEncoder(28),
                    nn.BatchNorm1d(28),
                    nn.Linear(28, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh()
                    )
        self.encoder = nn.Sequential(
                            self.backbone_nn,
                            nn.Linear(256, 16),
                            nn.Linear(16, 3),
                           )
    
    def forward(self, input_data, train=False):
        whole_seq = input_data["img"]
        act_all = input_data["act"][:, :, :-1]
        batch_size, seq_len, obs_dim = whole_seq.shape
        encoder_input = input_data["img"][:, :, :3]
        action = self.encoder(whole_seq.view(-1, obs_dim))
        action_dist = Normal(action, tc.ones_like(action, requires_grad=False))
        return {"latent_action_t": action_dist, "next_pos_posterior": action_dist, 
                "goal_switching_posterior_probs": None, "goal_onehot_idx": None,
                "goal_idx_seq": tc.ones(size=(batch_size, seq_len, 1)).to(action.device),
                "goal_seq": None,
                "states_prior": action_dist,
                "image_dist": action_dist,
                "goal_switching_prior": None,
                "goal_posterior": None,
                "goal_prior": None}
        
    def compute_losses(self, input_data, output, epoch, **kwargs):
        im_seq = input_data["img"][:, :, :3]
        act_t = input_data["act"][:, :-1].squeeze(dim=-1).unsqueeze(dim=-2)[:, :, :, :-1]
        batch_size, seq_len, obs_dim = im_seq.shape
        # print(output["latent_action_t"].mean.reshape(batch_size, seq_len, self.state_dim).shape, act_t.shape)
        loss = ((output["latent_action_t"].mean.reshape(batch_size, seq_len, 1, self.act_dim)[:, :-1] - act_t)**2).mean()
        return {"loss": loss, "next_rec": loss, "trans_pos_kl": loss, 
                "action_loss": loss, "label_loss": loss, "action_mse": loss,
                "transition_mat": loss, "encoder_loss": loss,
                "tagging_loss": loss, "image_loss": loss, "goal_loss": loss} 

class BCMDN(nn.Module):
    def __init__(self, img_shape, action_dim, state_dim, 
                 rank="diag", dt=0.1, steps=1, window=1, kl_reg=1.0, divider=1, decoder="ha", simu=1, n_skill=5):
        super(BCMDN, self).__init__()
        self.act_dim = action_dim
        self.state_dim = state_dim
        self.img_shape = img_shape
        # self.image_pixel_num = np.prod(img_shape)
        self.window = window
        self.n_skill = n_skill
        # self.encoder = nn.Sequential(
        #                     MLPEncoder(28),
        #                     nn.Linear(28, 3)
        #                     )
        self.backbone_nn = nn.Sequential(
                    # MLPEncoder(28),
                    nn.BatchNorm1d(28),
                    nn.Linear(28, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh()
                    )
        self.encoder = nn.Sequential(
                            self.backbone_nn,
                            nn.Linear(256, 16),
                            nn.Linear(16, 2*action_dim*n_skill)
                           )
        
        self.prir_goal_switcher = nn.Sequential(
            self.backbone_nn,
            nn.Linear(256, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, n_skill)
        )
        
    def goal_switching_prior(self, im):
        # res = self.encoder[4](states)
        batch_size, seq_len, obs_dim = im.shape
        res = self.prir_goal_switcher(im.view(-1, obs_dim))
        res = res.reshape(batch_size, seq_len, self.n_skill)
        return res
    
    def posterior(self, im):
        batch_size, seq_len, obs_dim = im.shape
        res = self.encoder(im.view(-1, obs_dim))
        mu, _ = tc.split(res, [self.act_dim*self.n_skill, self.act_dim*self.n_skill], dim=1)
        mu = mu.reshape(batch_size, seq_len, self.n_skill, self.act_dim)
        # sigma = logsigma.exp() + 1e-4
        # sigma = sigma.reshape(batch_size, seq_len, self.n_skill, self.act_dim)
        sigma = tc.ones_like(mu, requires_grad=False) * 0.05
        return Normal(mu, sigma)
    
    def cal_transition(self, batch):
        # Compute the transition counts
        transition_counts = (tc.matmul(batch[:, :-1].transpose(1, 2), batch[:, 1:]) + 1).sum(dim=0)

        # Normalize the transition matrix to get transition probabilities
        transition_matrix = transition_counts / transition_counts.sum(dim=1, keepdim=True)
        return transition_matrix.float()
    
    def forward(self, input_data, train=False):
        whole_seq = input_data["img"]
        # act_all = input_data["act"][:, :, :-1]
        batch_size, seq_len, obs_dim = whole_seq.shape
        # encoder_input = input_data["img"][:, :, :]
        # action = self.encoder(whole_seq.view(-1, obs_dim))
        # action_dist = Normal(action, tc.ones_like(action, requires_grad=False))
        action_dist = self.posterior(whole_seq) # batch_size x seq_len x n_skill x action_dim
        logits = self.goal_switching_prior(whole_seq) # batch_size x seq_len x n_skill
        goal_switching_prior = Categorical(logits=logits)
        goal_onehot_idx = goal_switching_prior.probs
        transition_mat = self.cal_transition(goal_onehot_idx.float())
        return {"latent_action_t": action_dist, "next_pos_posterior": action_dist, 
                "goal_switching_posterior_probs": None, "goal_onehot_idx": goal_onehot_idx,
                "goal_idx_seq": tc.argmax(goal_onehot_idx, dim=-1),
                "goal_seq": None,
                "states_prior": action_dist,
                "image_dist": action_dist,
                "goal_switching_prior": goal_switching_prior,
                "goal_posterior": None,
                "goal_prior": None,
                "transition_mat": transition_mat}
        
    def compute_losses(self, input_data, output, epoch, **kwargs):
        im_seq = input_data["img"][:, :, :3]
        act_t = input_data["act"].squeeze(dim=-1).unsqueeze(dim=-2)[:, :, :, :-1]
        batch_size, seq_len, obs_dim = im_seq.shape
        # print(output["latent_action_t"].mean.reshape(batch_size, seq_len, self.state_dim).shape, act_t.shape)
        loss = -1 * (output["latent_action_t"].log_prob(act_t).sum(dim=-1) * output["goal_switching_prior"].probs).mean()
        action_mse = (((output["latent_action_t"].mean  - act_t)**2).sum(dim=-1) * output["goal_switching_prior"].probs).mean()
        return {"loss": loss, "next_rec": loss, "trans_pos_kl": loss, 
                "action_loss": loss, "label_loss": loss, "action_mse": action_mse,
                "encoder_loss": loss,
                "tagging_loss": loss, "image_loss": loss, "goal_loss": loss}
        
class BCSDN(nn.Module):
    def __init__(self, img_shape, action_dim, state_dim, 
                 rank="diag", dt=0.1, steps=1, window=1, kl_reg=1.0, divider=1, decoder="ha", simu=1, n_skill=16):
        super(BCSDN, self).__init__()
        self.act_dim = action_dim
        self.state_dim = state_dim
        self.img_shape = img_shape
        # self.image_pixel_num = np.prod(img_shape)
        self.window = window
        self.n_skill = n_skill
        # self.encoder = nn.Sequential(
        #                     MLPEncoder(28),
        #                     nn.Linear(28, 3)
        #                     )
        self.backbone_nn = nn.Sequential(
                    # MLPEncoder(28),
                    nn.BatchNorm1d(28),
                    nn.Linear(28, 256),
                    nn.Tanh(),
                    nn.Linear(256, 256),
                    nn.Tanh()
                    )
        self.tau = 0.05
        
        self.prir_goal_switcher = nn.Sequential(
            self.backbone_nn,
            nn.Linear(256, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, n_skill)
        )
        
        self.goal_mu = nn.Parameter(tc.randn(size=(n_skill, state_dim)), requires_grad=True)
        self.gain = nn.Parameter(tc.randn(size=(n_skill, action_dim * state_dim)), requires_grad=True)
        
    def goal_switching_prior(self, im):
        batch_size, seq_len, obs_dim = im.shape
        # res = self.encoder[4](states)
        res = self.prir_goal_switcher(im.view(-1, obs_dim))
        res = res.reshape(batch_size, seq_len, self.n_skill)
        return res
    
    def cal_transition(self, batch):
        # Compute the transition counts
        transition_counts = (tc.matmul(batch[:, :-1].transpose(1, 2), batch[:, 1:]) + 1).sum(dim=0)

        # Normalize the transition matrix to get transition probabilities
        transition_matrix = transition_counts / transition_counts.sum(dim=1, keepdim=True)
        return transition_matrix.float()
    
    def action_posterior(self, gain, goals, states_mu, states_var):
        action = tc.matmul(gain, (goals - states_mu).unsqueeze(dim=-1)).squeeze(dim=-1)
        action_sigma = tc.ones_like(action, requires_grad=False) * self.tau
        # action_sigma = self.action_logstd.exp().view(1, 1, 1, -1).expand(*action.shape)
        # states_var = gain**2 * (states_var + action_var)
        return Normal(action, action_sigma)
    
    def forward(self, input_data, train=False):
        whole_seq = input_data["img"]
        batch_size, seq_len, obs_dim = whole_seq.shape

        # action_dist = self.posterior(whole_seq.view(-1, obs_dim)) # batch_size x n_skill x action_dim
        logits = self.goal_switching_prior(whole_seq) # batch_size x n_skill
        goal_switching_prior = Categorical(logits=logits)
        goal_onehot_idx = goal_switching_prior.probs.reshape(batch_size, seq_len, self.n_skill)
        
        batch_goals = self.goal_mu.view(1, 1, self.n_skill, self.state_dim)
        batch_gains = self.gain.view(1, 1, self.n_skill, self.act_dim, self.state_dim)
        states = whole_seq.unsqueeze(dim=-2)
        action_dist = self.action_posterior(batch_gains, batch_goals, states, None)
        transition_mat = self.cal_transition(goal_onehot_idx.float())
        return {"latent_action_t": action_dist, "next_pos_posterior": action_dist, 
                "goal_switching_posterior_probs": None, "goal_onehot_idx": goal_onehot_idx,
                "goal_idx_seq": tc.argmax(goal_onehot_idx, dim=-1),
                "goal_seq": None,
                "states_prior": action_dist,
                "image_dist": action_dist,
                "goal_switching_prior": goal_switching_prior,
                "goal_posterior": None,
                "goal_prior": None,
                "transition_mat": transition_mat}
        
    def compute_losses(self, input_data, output, epoch, **kwargs):
        im_seq = input_data["img"][:, :, :3]
        act_t = input_data["act"].squeeze(dim=-1).unsqueeze(dim=-2)[:, :, :, :-1]
        batch_size, seq_len, obs_dim = im_seq.shape
        # print(output["latent_action_t"].mean.reshape(batch_size, seq_len, self.state_dim).shape, act_t.shape)
        loss = -1 * (output["latent_action_t"].log_prob(act_t).sum(dim=-1) * output["goal_switching_prior"].probs).mean()
        action_mse = (((output["latent_action_t"].mean  - act_t)**2).sum(dim=-1) * output["goal_switching_prior"].probs).mean()
        label_loss = -((F.gumbel_softmax(output["goal_switching_prior"].logits, hard=True).mean(dim=[0, 1]) + 1e-3).log() * 1/self.n_skill).sum(dim=-1)

        # gain loss
        gain_loss = 0
        for i in range(self.n_skill):
            A = self.gain[i].view(self.act_dim, self.state_dim)
            square_A = tc.matmul(A.T, A)
            c = (tc.linalg.eigvals(square_A).abs()[:self.act_dim]- 1)
            # c = torch.eig(square_matrix, eigenvectors=False)
            mask = c > 0
            gain_loss += c[mask].sum()
        return {"loss": loss + label_loss + gain_loss, "next_rec": loss, "trans_pos_kl": loss, 
                "action_loss": loss, "label_loss": label_loss, "action_mse": action_mse,
                "encoder_loss": loss, "gain_loss": gain_loss,
                "tagging_loss": loss, "image_loss": loss, "goal_loss": loss} 

