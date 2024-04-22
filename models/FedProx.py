import os
import uuid

import numpy as np
import torch
from functorch import make_functional
from torch.utils.data import DataLoader, Dataset

from models.BaseModel import BaseModel
from models.FedAvg import FedAvg
from utils.HyperNetClasses import IdentityModel
from utils.models import CNNCifar
from utils.train_utils import DatasetSplit, clip_norm_


class FedProx(BaseModel):
    def __init__(self, args, num_classes):
        args.beta = 0
        args.aggw = 0
        args.client_lr = 0
        args.inner_steps = 0
        args.gradclip = 5
        args.use_cyclical_beta = False

        args.uuid = str(uuid.uuid1())[:8]
        args.exp = f'{getattr(args, "pre", "")}_{args.algorithm}_{args.dataset}_{args.alpha}_mfrac:{args.frac_m}('
        args.exp += f"lepochs:{args.local_epochs}_lbs:{args.local_bs}_"
        args.exp += f"ilr:{args.inner_lr}_mu:{args.mu}_"
        args.exp += f"test_dist:{args.test_dist}_"
        args.exp += f")" + f"_{args.uuid}"
        print(args.exp)
        super().__init__(args)

        self.args.num_classes = num_classes
        self.model = CNNCifar(self.args.num_classes)
        self.model = self.model.cuda()
        self.hpnet = IdentityModel(self.model, self.args.num_users)
        self.server_optimizer = torch.optim.SGD(self.hpnet.w_global.parameters(), lr=1.0, momentum=0.0, weight_decay=0)  # 1e-5
        self.functional, self.gparam = make_functional(self.model)

    def client_update(self, ldr_train, wt, w_local):
        loss_list = []
        acc_list = []
        for images, labels in ldr_train:
            # images, labels = next(ldr_train.__iter__())
            x = images.cuda()
            y = labels.cuda()

            y_pred = self.functional(wt, x.cuda())
            loss = self.criteria(y_pred, y)
            losses = loss

            proximal = self.args.mu * 0.5 * torch.norm(torch.cat([w.view(-1) for w in wt]) - torch.cat([w.view(-1) for w in w_local])) ** 2
            losses += proximal

            grad = torch.autograd.grad(losses, wt)  # FO-MAML
            clip_norm_(grad, self.args.gradclip)
            wt = [w - self.args.inner_lr * g for w, g in zip(wt, grad)]

            mean_loss = losses.mean()
            mean_acc = y_pred.argmax(1).eq(y).sum().item() / len(y)
            loss_list.append(mean_loss.item())
            acc_list.append(mean_acc)

        loss_avg = np.mean(loss_list)
        acc_avg = np.mean(acc_list)
        return wt, loss_avg, acc_avg, torch.zeros(1)

    def server_aggregation(self, user_idxs, users_datasize, collection):
        collected_weights = collection[0]

        weights_size = []
        for idx in user_idxs:
            weights_size.append(users_datasize[idx])
        weights = torch.Tensor(weights_size) / sum(weights_size)

        # for i, idx in enumerate(user_idxs):
        for i, idx in enumerate(user_idxs):
            # delta_theta = [ torch.Tensor((wg - wl).data).detach().clone() for wg , wl in zip(hpnet.w_global, collected_weights[i],)]
            w_local = self.hpnet(idx)
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
