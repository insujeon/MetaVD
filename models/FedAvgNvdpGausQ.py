import os
import uuid

import numpy as np
import torch
from functorch import make_functional
from torch.utils.data import DataLoader, Dataset

from models.BaseModel import BaseModel
from utils.HyperNetClasses import NVDPModel
from utils.models import CNNCifarNVDP
from utils.train_utils import DatasetSplit, clip_norm_


class FedAvgNvdpGausQ(BaseModel):
    def __init__(self, args, num_classes):
        args.aggw = 0
        args.client_lr = 0
        args.inner_steps = 0
        args.gradclip = 5
        args.momentum = 0.0

        args.uuid = str(uuid.uuid1())[:8]
        args.exp = f'{getattr(args, "pre", "")}_{args.algorithm}_{args.dataset}_{args.alpha}_mfrac:{args.frac_m}('
        args.exp += f"lepochs:{args.local_epochs}_lbs:{args.local_bs}_"
        args.exp += f"ilr:{args.inner_lr}_slr:{args.server_lr}_mom:{args.momentum}_"
        args.exp += f"mom1:{args.momentum1}_mom2:{args.momentum2}_decay1:{args.decay1}_"
        args.exp += f"_beta:{args.beta}_cyc:{args.temp_cycle}_ratio:{args.temp_ratio}_"
        args.exp += f"adepochs:{args.adaptation_epochs}_adsteps:{args.adaptation_steps}_adbs:{args.adaptation_bs}_"
        args.exp += f"test_dist:{args.test_dist}_"
        args.exp += f"u:{args.num_users}_ou:{args.ood_users}_"
        args.exp += f")" + f"_{args.uuid}"
        print(args.exp)

        super().__init__(args)

        self.args.num_classes = num_classes
        self.model = CNNCifarNVDP(self.args.num_classes, gauss=True, gamma=self.args.gamma)
        self.model = self.model.cuda()
        self.hpnet = NVDPModel(self.model, self.args.num_users, self.args.num_layers, self.args.hidden_dim)
        self.server_optimizer = torch.optim.SGD(
            [
                # {'params': self.hpnet.w_global.parameters(), 'lr':self.args.server_lr, 'momentum':0.9, 'weight_decay': 0},
                {"params": self.hpnet.w_global.parameters(), "lr": 1.0, "momentum": 0.0, "weight_decay": 0},
                {"params": self.hpnet.w_logits.parameters(), "lr": self.args.server_lr, "momentum": self.args.momentum1, "weight_decay": self.args.decay1},
                {"params": self.hpnet.embeds.parameters(), "lr": self.args.server_lr, "momentum": self.args.momentum2, "weight_decay": self.args.decay1},
                # {'params': hpnet.tau.parameters(), 'lr':args.embed_lr, 'momentum':0, 'weight_decay':0}
            ]
        )
        self.functional, self.gparam = make_functional(self.model)

    def client_update(self, ldr_train, wt, w_local=None):
        loss_list = []
        acc_list = []
        kl_list = []
        for images, labels in ldr_train:
            # images, labels = next(ldr_train.__iter__())
            x = images.cuda()
            y = labels.cuda()

            y_pred = self.functional(wt, x.cuda())
            loss = self.criteria(y_pred, y)

            # kl_loss = self.model.kl_div(wt[10])
            kl_loss = self.model.kl_div(wt[10], wt[8])
            losses = loss + self.args.beta * kl_loss

            grad = torch.autograd.grad(losses, wt)  # FO-MAML
            clip_norm_(grad, self.args.gradclip)

            wt = [w - self.args.inner_lr * g for w, g in zip(wt, grad)]

            mean_loss = losses.mean()
            mean_acc = y_pred.argmax(1).eq(y).sum().item() / len(y)
            mean_kl = kl_loss

            loss_list.append(mean_loss.item())
            acc_list.append(mean_acc)
            kl_list.append(mean_kl.item())

        mean_loss = np.mean(loss_list)
        mean_acc = np.mean(acc_list)
        mean_kl = np.mean(kl_list)
        return wt, mean_loss, mean_acc, mean_kl

    def server_aggregation(self, user_idxs, users_datasize, collection):
        collected_weights = collection[0]
        iter = collection[2]
        drop_prob = collection[3]

        weights_size = []
        for idx in user_idxs:
            weights_size.append(users_datasize[idx])
        weights = torch.Tensor(weights_size) / sum(weights_size)

        # weights2 = weights.reshape(10,1,1).cuda()
        weights2 = weights.reshape(self.args.frac_m, 1, 1).cuda()

        mudist = torch.cat([collected_weights[i][8].unsqueeze(0) for i in range(self.args.frac_m)])
        vardist = torch.cat([collected_weights[i][10].clamp(-8.5, 8.5).exp().unsqueeze(0) for i in range(self.args.frac_m)])
        # vardist = torch.cat([ torch.nn.functional.softplus(collected_weights[i][10].clamp(-20,40)).unsqueeze(0) for i in range(self.args.frac_m)])
        # vardist = torch.cat([collected_weights[i][10].exp().unsqueeze(0) * collected_weights[i][8].unsqueeze(0)**2 for i in range(self.args.frac_m)])

        # Reparameterization Trick Case
        # vardist = torch.cat([torch.nn.functional.softplus(collected_weights[i][10]).unsqueeze(0) for i in range(self.args.frac_m)])
        varinv = torch.ones(1).cuda() / (vardist + 1e-10)
        # var = torch.ones(1).cuda() / (varinv.sum(0)+1e-10)
        mu = (weights2 * varinv * mudist).sum(0) / ((weights2 * varinv).sum(0) + 1e-10)

        for i, idx in enumerate(user_idxs):
            # mu_dist.append(collected_weights[i][8].unsqueeze(0))
            # var_dist.append( collected_weights[i][10].exp().unsqueeze(0) * collected_weights[i][8].unsqueeze(0)**2 )
            # collected_weights[i][10] = torch.log(var + 1e-13)
            collected_weights[i][8] = mu

        # for i, idx in enumerate(user_idxs):
        for i, idx in enumerate(user_idxs):
            # delta_theta = [ torch.Tensor((wg - wl).data).detach().clone() for wg , wl in zip(hpnet.w_global, collected_weights[i],)]

            w_local = self.hpnet(idx)
            # w_local = self.hpnet(idx) if iter < self.args.dropstart else self.hpnet(idx, ood=False, drop_prob=drop_prob)

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
        return wt
