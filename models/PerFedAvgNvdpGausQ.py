import math
import os
import uuid
from copy import deepcopy
from typing import Tuple, Union

import numpy as np
import torch
from functorch import make_functional
from torch.utils.data import DataLoader, Dataset

from models.BaseModel import BaseModel
from utils.HyperNetClasses import NVDPModel
from utils.models import CNNCifarNVDP
from utils.train_utils import DatasetSplit, clip_norm_


class PerFedAvgNvdpGausQ(BaseModel):
    def __init__(self, args, num_classes):
        args.aggw = 0
        args.delta = 1e-4  # 1e-3 ~ 1e-5
        args.gradclip = 5.0  # 5, 10

        args.uuid = str(uuid.uuid1())[:8]
        args.exp = (
            f"_{args.algorithm}_(mfrac:{args.frac_m}_serverlr:{args.server_lr}_localepochs:{args.local_epochs}_clr:{args.client_lr}_ilr_{args.inner_lr}_beta:{args.beta}_alpha:{args.alpha}_adepochs:{args.adaptation_epochs}_dp:{args.drop_prob}_cyc:{args.temp_cycle}_ratio:{args.temp_ratio})"
            + f"_{args.uuid}"
        )
        args.exp += f"test_dist:{args.test_dist}_"
        print(args.exp)

        super().__init__(args)

        self.args.num_classes = num_classes
        self.model = CNNCifarNVDP(self.args.num_classes, gauss=True)
        self.model = self.model.cuda()
        self.hpnet = NVDPModel(self.model, self.args.num_users, self.args.num_layers, self.args.hidden_dim)
        self.server_optimizer = torch.optim.SGD(
            [
                {"params": self.hpnet.w_global.parameters(), "lr": self.args.server_lr, "momentum": self.args.momentum, "weight_decay": 0},
                {"params": self.hpnet.w_logits.parameters(), "lr": self.args.server_lr, "momentum": self.args.momentum1, "weight_decay": self.args.decay1},
                {"params": self.hpnet.embeds.parameters(), "lr": self.args.server_lr, "momentum": self.args.momentum2, "weight_decay": self.args.decay1},
                # {'params': hpnet.tau.parameters(), 'lr':args.embed_lr, 'momentum':0, 'weight_decay':0}
            ]
        )
        self.functional, self.gparam = make_functional(self.model)

    def client_update(self, ldr_train, wt, w_local=None):
        numiter = math.ceil(len(ldr_train) / 3)
        loss_list = []
        acc_list = []
        kl_list = []
        for i_ in range(numiter):
            images, labels = next(ldr_train.__iter__())
            x = images.cuda()
            y = labels.cuda()

            wt0 = [torch.Tensor(w) for w in wt]
            # 1 step adaptation
            y_pred = self.functional(wt, x.cuda())
            loss = self.criteria(y_pred, y)
            grads_ = torch.autograd.grad(loss, wt)
            grads_[10].fill_(0)  # logit grad zero,  added
            clip_norm_(grads_, self.args.gradclip)
            wt = [w - self.args.inner_lr * g for w, g in zip(wt, grads_)]

            # gradient 1st
            images, labels = next(ldr_train.__iter__())
            x = images.cuda()
            y = labels.cuda()
            y_pred = self.functional(wt, x.cuda())
            loss = self.criteria(y_pred, y)
            kl_loss = self.model.kl_div(wt[10], wt[8])  # added
            losses = loss + self.args.beta * kl_loss  # added
            grads_1st = torch.autograd.grad(losses, wt)

            # gradient 2nd
            images, labels = next(ldr_train.__iter__())
            x_ = images.cuda()
            y_ = labels.cuda()
            grads_2nd = self.approximate_hessian_nvdp(wt0, self.functional, [x_, y_], grads_1st, self.args.delta)

            clip_norm_(grads_1st, self.args.gradclip)
            clip_norm_(grads_2nd, self.args.gradclip)

            wt = [w - self.args.client_lr * (g1 - self.args.inner_lr * g2) for w, g1, g2 in zip(wt0, grads_1st, grads_2nd)]  # HF-MAML

            losses = loss + self.args.beta * kl_loss
            mean_loss = losses.mean()
            mean_acc = y_pred.argmax(1).eq(y).sum().item() / len(y)
            mean_kl = kl_loss

            loss_list.append(mean_loss.item())
            acc_list.append(mean_acc)
            kl_list.append(mean_kl.item())

        loss_avg = np.mean(loss_list)
        acc_avg = np.mean(acc_list)
        kl_avg = np.mean(kl_list)
        return wt, loss_avg, acc_avg, kl_avg

    def server_aggregation(self, user_idxs, users_datasize, collection):
        collected_weights = collection[0]
        iter = collection[2]
        drop_prob = collection[3]

        weights_size = []
        for idx in user_idxs:
            weights_size.append(users_datasize[idx])
        weights = torch.Tensor(weights_size) / sum(weights_size)

        weights2 = weights.reshape(10, 1, 1).cuda()

        mudist = torch.cat([collected_weights[i][8].unsqueeze(0) for i in range(self.args.frac_m)])
        vardist = torch.cat([collected_weights[i][10].clamp(-8.5, 8.5).exp().unsqueeze(0) for i in range(self.args.frac_m)])
        # vardist = torch.cat([ torch.nn.functional.softplus(collected_weights[i][10].clamp(-20,40)).unsqueeze(0) for i in range(self.args.frac_m)])
        # vardist = torch.cat([collected_weights[i][10].exp().unsqueeze(0) * collected_weights[i][8].unsqueeze(0)**2 for i in range(self.args.frac_m)])

        # Reparameterization Trick Case
        # vardist = torch.cat([torch.nn.functional.softplus(collected_weights[i][10]).unsqueeze(0) for i in range(self.args.frac_m)])
        varinv = torch.ones(1).cuda() / (vardist + 1e-10)
        # mu = (mudist * varinv).sum(0) / ((varinv).sum(0)+1e-10)
        mu = (weights2 * varinv * mudist).sum(0) / ((weights2 * varinv).sum(0) + 1e-10)

        for i, idx in enumerate(user_idxs):
            # mu_dist.append(collected_weights[i][8].unsqueeze(0))
            # var_dist.append( collected_weights[i][10].exp().unsqueeze(0) * collected_weights[i][8].unsqueeze(0)**2 )
            # collected_weights[i][10] = torch.log(var + 1e-13)
            collected_weights[i][8] = mu

        # for i, idx in enumerate(user_idxs):
        for i, idx in enumerate(user_idxs):
            # delta_theta = [ torch.Tensor((wg - wl).data).detach().clone() for wg , wl in zip(hpnet.w_global, collected_weights[i],)]

            # w_local = self.hpnet(idx)
            w_local = self.hpnet(idx) if iter < self.args.dropstart else self.hpnet(idx, ood=False, drop_prob=drop_prob)
            # w_local = self.hpnet(idx) if collection[2] >= self.args.nvdpstart else self.hpnet(idx, ood=True)

            delta_theta = [torch.Tensor((wg - wl).data).detach().clone() for wg, wl in zip(w_local, collected_weights[i])]
            # hnet_grads = torch.autograd.grad(w_local, hpnet(idx), delta_theta)
            hnet_grads = torch.autograd.grad(w_local, self.hpnet.parameters(), delta_theta, allow_unused=True)

            # for p, g in zip(hpnet(idx), hnet_grads):
            for (name, p), g in zip(self.hpnet.named_parameters(), hnet_grads):
                if p.grad == None:
                    p.grad = torch.zeros_like(p)
                if g == None:
                    g = torch.zeros_like(p)
                # p.grad = p.grad + g / self.args.frac_m
                p.grad = p.grad + g * weights[i]

        torch.nn.utils.clip_grad_norm_(self.hpnet.parameters(), 100)

        self.server_optimizer.step()
        self.server_optimizer.zero_grad()

    def client_adapt(self, ldr_train, wt):
        steps = len(ldr_train) if self.args.adaptation_steps < 0 else self.args.adaptation_steps
        for _ in range(steps):
            images, labels = next(ldr_train.__iter__())
            x = images.cuda()
            y = labels.cuda()

            y_pred = self.functional(wt, x.cuda())
            loss = self.criteria(y_pred, y)
            kl_loss = self.model.kl_div(wt[10], wt[8])
            losses = self.criteria(y_pred, y) + self.args.beta * kl_loss

            grad = torch.autograd.grad(losses, wt)  # FO-MAML
            clip_norm_(grad, self.args.gradclip)
            wt = [w - self.args.inner_lr * g for w, g in zip(wt, grad)]

        return wt  # , mean_loss, mean_acc, torch.zeros(1)

    def approximate_hessian_nvdp(self, w_local, functional, data_batch: Tuple[torch.Tensor, torch.Tensor], grad, delta=1e-4):
        w_local = [torch.Tensor(w.data).detach().clone().requires_grad_(True) for w in w_local]
        criterion = torch.nn.CrossEntropyLoss()
        x, y = data_batch

        wt_1 = [torch.Tensor(w) for w in w_local]
        wt_2 = [torch.Tensor(w) for w in w_local]

        wt_1 = [w + delta * g for w, g in zip(wt_1, grad)]
        wt_2 = [w - delta * g for w, g in zip(wt_1, grad)]

        logit_1 = functional(wt_1, x)
        loss_1 = criterion(logit_1, y)
        kl_loss = self.model.kl_div(wt_1[10], wt_1[8])  # added
        loss_1 = loss_1 + self.args.beta * kl_loss  # added

        grads_1 = torch.autograd.grad(loss_1, w_local)

        logit_2 = functional(wt_2, x)
        loss_2 = criterion(logit_2, y)
        kl_loss = self.model.kl_div(wt_2[10], wt_2[8])  # added
        loss_2 = loss_2 + self.args.beta * kl_loss  # added
        grads_2 = torch.autograd.grad(loss_2, w_local)

        with torch.no_grad():
            grads_2nd = deepcopy(grads_1)
            for g_2nd, g1, g2 in zip(grads_2nd, grads_1, grads_2):
                g_2nd.data = (g1 - g2) / (2 * delta)

        return grads_2nd
