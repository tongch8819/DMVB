import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from BSIBO.utils import utils
from BSIBO.encoder import make_rssm_encoder
from BSIBO.utils.rssm import get_feat, get_dist, flatten_states, RSSMState
from BSIBO.utils.schedulers import ExponentialScheduler
from BSIBO.utils import pytorch_util as ptu
from BSIBO.ac import Actor, Critic
from BSIBO.transition_model import make_transition_model


class BSIBO(nn.Module):
    """
    BSIBO
    """
    def __init__(
        self, obs_shape, z_dim,
        encoder, encoder_target, output_type="continuous"
    ):
        super(BSIBO, self).__init__()

        self.encoder = encoder

        self.encoder_target = encoder_target

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(
        self, x, actions, ema=False
    ):
        """
        Encoder: z_t = e(o_t, s_{t-1}, a_{t-1})
        :param x: obs at t
        :return: z_t, value in r2
        """
        batch_t, batch_b, ch, h, w = x.size()

        # Obtain prev actions
        prev_actions = actions[:-1]
        prev_act = torch.zeros(
            batch_b, self.encoder.action_shape,
            device=ptu.device, dtype=prev_actions.dtype
        ).unsqueeze(0)
        prev_actions = torch.cat([prev_act, prev_actions], dim=0)
        # Embed the pixel observation
        prev_state = self.encoder.representation.initial_state(
            batch_b, device=ptu.device
        )
        # Rollout model by taking the same series of actions as the real model
        if ema:
            with torch.no_grad():
                embeds = self.encoder_target.observation_encoder(x)
                prior, post = self.encoder_target.rollout.\
                    rollout_representation(batch_t, embeds,
                                           prev_actions, prev_state)
        else:
            embeds = self.encoder.observation_encoder(x)
            prior, post = self.encoder.rollout.rollout_representation(
                batch_t, embeds, prev_actions, prev_state
            )

        return prior, post

    def multi_view_encode(self, obs, z, actions, ema=False):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        # Obtain prev states

        def unsqueeze_states(rssm_states, dim):
            return RSSMState(
                rssm_states.mean.unsqueeze(dim),
                rssm_states.std.unsqueeze(dim),
                rssm_states.stoch.unsqueeze(dim),
                rssm_states.deter.unsqueeze(dim),
            )

        def cat_states(rssm_states: list, dim):
            return RSSMState(
                torch.cat([state.mean for state in rssm_states], dim=dim),
                torch.cat([state.std for state in rssm_states], dim=dim),
                torch.cat([state.stoch for state in rssm_states], dim=dim),
                torch.cat([state.deter for state in rssm_states], dim=dim),
            )

        prev_states = z[:-1]
        prev_state = self.encoder.representation.initial_state(
            batch_shape[1], device=ptu.device
        )
        prev_state = unsqueeze_states(prev_state, dim=0)
        prev_states = cat_states([prev_state, prev_states], dim=0)
        # Obtain prev actions
        prev_actions = actions[:-1]
        prev_act = torch.zeros(
            batch_shape[1], self.encoder.action_shape,
            device=ptu.device, dtype=prev_actions.dtype
        ).unsqueeze(0)
        prev_actions = torch.cat([prev_act, prev_actions], dim=0)

        # Flatten inputs
        flatten_batch = np.prod(batch_shape)
        obs = torch.reshape(obs, (-1, *img_shape))
        prev_states = flatten_states(prev_states, flatten_batch)
        prev_actions = torch.reshape(prev_actions, (flatten_batch, -1))

        if ema:
            with torch.no_grad():
                states = self.encoder_target(obs, prev_actions, prev_states)
        else:
            states = self.encoder(obs, prev_actions, prev_states)
        return states

    def compute_logits(self, z1, z2):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z1 (W z2.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy
          with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z2.T)  # (z_dim, B)
        logits = torch.matmul(z1, Wz)  # (B, B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def compute_skl(self, z1_dist, z2_dist):
        kl_1_2 = torch.mean(
            torch.distributions.kl.kl_divergence(z1_dist, z2_dist)
        )
        kl_2_1 = torch.mean(
            torch.distributions.kl.kl_divergence(z2_dist, z1_dist)
        )
        skl = (kl_1_2 + kl_2_1) / 2.
        return skl

    def compute_multi_view_skl(self, obs1, obs2, z1, z2, actions, ema=False):
        z1_ = self.multi_view_encode(obs1, z2, actions)
        z2_ = self.multi_view_encode(obs2, z1, actions, ema=ema)
        z1_dist = get_dist(z1_)
        z2_dist = get_dist(z2_)

        kl_1_2 = torch.mean(
            torch.distributions.kl.kl_divergence(z1_dist, z2_dist)
        )
        kl_2_1 = torch.mean(
            torch.distributions.kl.kl_divergence(z2_dist, z1_dist)
        )
        skl = (kl_1_2 + kl_2_1) / 2.
        return skl

    def compute_kl_balancing(
        self, z1_prior, z1_post
    ):
        def get_dist(rssm_state: RSSMState, ema=False):
            if not ema:
                return td.independent.Independent(
                    td.Normal(rssm_state.mean, rssm_state.std), 1
                )
            else:
                return td.independent.Independent(
                    td.Normal(
                        rssm_state.mean.detach(), rssm_state.std.detach()
                    ), 1
                )
        kl_t = 0.8 * torch.mean(
            torch.distributions.kl.kl_divergence(
                get_dist(z1_post, ema=True), get_dist(z1_prior)
            )
        )
        kl_q = 0.2 * torch.mean(
            torch.distributions.kl.kl_divergence(
                get_dist(z1_post), get_dist(z1_prior, ema=True)
            )
        )
        return kl_t + kl_q


class BSIBOSacAgent(object):
    """BSIBO representation learning with SAC."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        transition_model_type,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='rssm',
        encoder_feature_dim=50,
        stochastic_size=30,
        deterministic_size=200,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        mib_update_freq=1,
        log_interval=100,
        multi_view_skl=False,
        mib_batch_size=10,
        mib_seq_len=50,
        beta_start_value=1e-3,
        beta_end_value=1,
        grad_clip=500,
        kl_balancing=False,
        # parameters for DBC
        encoder_stride=2,
        bisim_coef=0.5,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.mib_update_freq = mib_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.encoder_type = encoder_type
        self.multi_view_skl = multi_view_skl
        self.batch_size = mib_batch_size
        self.seq_len = mib_seq_len
        self.grad_clip = grad_clip
        self.kl_balancing = kl_balancing

        self.transition_model_type = transition_model_type
        self.bisim_coef = bisim_coef

        self.encoder = make_rssm_encoder(
            encoder_type, obs_shape, action_shape, encoder_feature_dim,
            stochastic_size, deterministic_size, num_layers,
            num_filters, hidden_dim, output_logits=True
        )
        self.encoder_target = make_rssm_encoder(
            encoder_type, obs_shape, action_shape, encoder_feature_dim,
            stochastic_size, deterministic_size, num_layers,
            num_filters, hidden_dim, output_logits=True
        )
        feature_dim = stochastic_size + deterministic_size

        # TODO: check diff between DBC and DRIBO
        # decouple encoder from actor in DBC
        self.actor = Actor(
            action_shape, hidden_dim, feature_dim,
            actor_log_std_min, actor_log_std_max
        ).to(device)

        self.critic = Critic(
            action_shape, hidden_dim, feature_dim
        ).to(device)

        self.critic_target = Critic(
            action_shape, hidden_dim, feature_dim
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.encoder_target.load_state_dict(self.encoder.state_dict())

        # -------------------------------------------------
        # transition model related components
        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape
        ).to(device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )
        # -------------------------------------------------

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            [{'params': self.critic.parameters()},
             {'params': self.encoder.parameters()}],
            lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type == 'rssm':
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.BSIBO = BSIBO(
                obs_shape, feature_dim,
                self.encoder, self.encoder_target, output_type='continuous'
            ).to(self.device)

            # optimizer for encoder for sequential MIB loss
            self.encoder_optimizer = torch.optim.Adam(
                self.encoder.parameters(), lr=encoder_lr
            )
            self.BSIBO_optimizer = torch.optim.Adam(
                self.BSIBO.encoder.parameters(), lr=encoder_lr
            )

        self.beta_scheduler = ExponentialScheduler(
            start_value=beta_start_value, end_value=beta_end_value,
            n_iterations=5e4, start_iteration=10000
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()
        self.encoder_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.encoder.train(training)
        if self.encoder_type == 'rssm':
            self.BSIBO.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, a=None, prev_state: RSSMState = None):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            prev_state = self.encoder(obs, a, prev_state)
            latent_states = get_feat(prev_state)
            # latent_states = self.encoder.ln(latent_states)
            mu, _, _, _ = self.actor(
                latent_states, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten(), mu, prev_state

    def sample_action(self, obs, a=None, prev_state: RSSMState = None):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            prev_state = self.encoder(obs, a, prev_state)
            latent_states = get_feat(prev_state)
            # latent_states = self.encoder.ln(latent_states)
            mu, pi, _, _ = self.actor(latent_states, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten(), pi, prev_state

    def update_critic(
        self, obs, action, reward, next_obs, target_next_obs,
        not_done, L, step
    ):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(
                target_next_obs, policy_action
            )
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(),
        #                                self.grad_clip)
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs)
        actor_Q1, actor_Q2 = self.critic(obs, pi)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_mib(self, obs1, obs2, actions, mib_kwargs, L, step):
        """
        Update Mutiview Information Bottleneck 
        """
        batch_t, batch_b, ch, h, w = obs1.size()

        z1_prior, z1 = self.BSIBO.encode(obs1, actions)
        z2_prior, z2 = self.BSIBO.encode(obs2, actions, ema=True)

        # Maximize mutual information of task-relevant features
        latent_states1 = get_feat(z1).reshape(batch_t * batch_b, -1)
        latent_states2 = get_feat(z2).reshape(batch_t * batch_b, -1)
        # latent_states1 = self.encoder.ln(latent_states1)
        # latent_states2 = self.encoder.ln(latent_states2)
        logits = self.BSIBO.compute_logits(latent_states1, latent_states2)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        # Minimize the task-irrelevant information
        z1_dist = get_dist(z1)
        z2_dist = get_dist(z2)
        if not self.multi_view_skl:
            skl = self.BSIBO.compute_skl(z1_dist, z2_dist)
        else:
            skl = self.BSIBO.compute_multi_view_skl(
                obs1, obs2, z1, z2, actions, ema=True
            )

        if self.kl_balancing:
            # z1_prior_dist = get_dist(z1_prior)
            # z2_prior_dist = get_dist(z2_prior)
            kl = self.BSIBO.compute_kl_balancing(
                z1_prior, z1
            )
            beta = self.beta_scheduler(step)
            loss += beta * kl
        beta = self.beta_scheduler(step)
        loss += beta * skl

        self.encoder_optimizer.zero_grad()
        self.BSIBO_optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(),
        #                                self.grad_clip)
        self.encoder_optimizer.step()
        self.BSIBO_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/BSIBO_loss', loss, step)
            L.log('train/beta', beta, step)
            L.log('train/skl', skl, step)





    def update(self, replay_buffer, L, step):
        self.update_dribo(replay_buffer, L, step)
        self.update_dbc(replay_buffer, L, step)

    def update_dbc(self, replay_buffer, L, step):
        # update encoder using DBC loss
        def compute_bisim_loss(obs, action, reward):
            """
            Update encoder using DBC loss (Bisimulation Based)
            """
            h_rssm_state = self.encoder(obs)  # this encoder returns a RSSMState object
            h = h_rssm_state.deter + h_rssm_state.stoch

    
            # Sample random states across episodes at random
            batch_size = obs.size(0)
            perm = np.random.permutation(batch_size)
            h2 = h[perm]
    
            with torch.no_grad():
                # action, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
                pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
                # reward = self.reward_decoder(pred_next_latent_mu1)
                reward2 = reward[perm]
            if pred_next_latent_sigma1 is None:
                pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
            if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
                pred_next_latent_mu2 = pred_next_latent_mu1[perm]
                pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
            elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
                pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
                pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
            else:
                raise NotImplementedError
    
            z_dist = F.smooth_l1_loss(h, h2, reduction='none')
            r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')
            if self.transition_model_type == '':
                transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none')
            else:
                transition_dist = torch.sqrt(
                    (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
                    (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
                )
                # transition_dist  = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none') \
                #     +  F.smooth_l1_loss(pred_next_latent_sigma1, pred_next_latent_sigma2, reduction='none')
    
            bisimilarity = r_dist + self.discount * transition_dist
            loss = (z_dist - bisimilarity).pow(2).mean()
            L.log('train_ae/encoder_loss', loss, step)
            return loss

        def compute_transition_reward_loss(obs, action, next_obs, reward):
            h = self.encoder(obs)
            pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
            if pred_next_latent_sigma is None:
                pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
    
            next_h = self.encoder(next_obs)
            diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
            loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
            L.log('train_ae/transition_loss', loss, step)
    
            pred_next_latent = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
            pred_next_reward = self.reward_decoder(pred_next_latent)
            reward_loss = F.mse_loss(pred_next_reward, reward)
            total_loss = loss + reward_loss
            return total_loss

        # random sample instead of sequential sampling
        obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        dbc_loss = compute_bisim_loss(obs, action, reward)
        transition_reward_loss = compute_transition_reward_loss(obs, action, next_obs, reward)
        total_loss = self.bisim_coef * dbc_loss + transition_reward_loss

        self.encoder_optimizer.zero_grad()
        # decoder optimizer binds to parameters of 1) reward decoder; 2) transition model.
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()


    def update_dribo(self, replay_buffer, L, step):
        # rssm sample multi-view
        obs, action, reward, \
            not_done, mib_kwargs = replay_buffer.sample_multi_view(
                self.batch_size, self.seq_len
            )

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        """
        Obtain latent states
        """
        batch_t, batch_b, ch, h, w = obs.size()

        flat_actions = action[:-1].reshape(
            (batch_t - 1) * batch_b, -1
        )
        rewards = reward[:-1].reshape(
            (batch_t - 1) * batch_b, -1
        )
        not_done = not_done[:-1].reshape(
            (batch_t - 1) * batch_b, -1
        )

        # obtain latent states
        _, post = self.BSIBO.encode(obs, action)
        feat = get_feat(post)
        # feat = self.encoder.ln(
        #     feat.reshape(batch_t * batch_b, -1)
        # ).reshape(batch_t, batch_b, -1)
        # [t, t + batch_t] - > [t, t + batch_t - 1]
        latent_states = feat[:-1].reshape((batch_t - 1) * batch_b, -1).detach()
        q_latent_states = feat[:-1].reshape((batch_t - 1) * batch_b, -1)
        # [t, t + batch_t] - > [t + 1, t + batch_t]
        next_latent_states = feat[1:].reshape(
            (batch_t - 1) * batch_b, -1
        ).detach()

        # obtain target latent states
        _, target_post = self.BSIBO.encode(obs, action, ema=True)
        target_feat = get_feat(target_post)
        # target_feat = self.encoder_target.ln(
        #     target_feat.reshape(batch_t * batch_b, -1)
        # ).reshape(batch_t, batch_b, -1)
        # [t, t + batch_t] - > [t + 1, t + batch_t]
        target_next_latent_states = target_feat[1:].reshape(
            (batch_t - 1) * batch_b, -1
        ).detach()

        self.update_critic(
            q_latent_states, flat_actions, rewards, next_latent_states,
            target_next_latent_states, not_done, L, step
        )

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(latent_states, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.encoder, self.encoder_target,
                self.encoder_tau
            )

        if step % self.mib_update_freq == 0 and self.encoder_type == 'rssm':
            obs1, obs2 = mib_kwargs["view1"], \
                mib_kwargs["view2"]
            self.update_mib(obs1, obs2, action, mib_kwargs, L, step)



    def save_BSIBO(self, model_dir, step):
        params = dict(BSIBO=self.BSIBO, encoder=self.encoder, actor=self.actor)
        torch.save(
            params, '%s/bsibo.pt' % (model_dir)
        )
