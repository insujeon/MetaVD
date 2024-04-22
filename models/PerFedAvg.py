import math
import os
import uuid

import numpy as np
import torch
from functorch import make_functional
from torch.utils.data import DataLoader, Dataset

from models.BaseModel import BaseModel
from utils.HyperNetClasses import IdentityModel
from utils.models import CNNCifar
from utils.train_utils import DatasetSplit, approximate_hessian, clip_norm_


class PerFedAvg(BaseModel):
    def __init__(self, args, num_classes):
        args.beta = 0
        args.aggw = 0
        args.inner_steps = 0
        args.local_steps = 17
        args.delta = 1e-4  # 1e-3 ~ 1e-5
        args.gradclip = 5.0  # 5, 10
        args.use_cyclical_beta = False

        args.uuid = str(uuid.uuid1())[:8]
        args.exp = (
            f"_{args.algorithm}_(mfrac:{args.frac_m}_serverlr:{args.server_lr}_localepochs:{args.local_epochs}_clr:{args.client_lr}_ilr_{args.inner_lr}_beta:{args.beta}_alpha:{args.alpha}_adepochs:{args.adaptation_epochs}_dp:{args.drop_prob}_cyc:{args.temp_cycle}_ratio:{args.temp_ratio})"
            + f"_{args.uuid}"
        )
        args.exp += f"test_dist:{args.test_dist}_"
        print(args.exp)

        super().__init__(args)

        self.args.num_classes = num_classes
        self.model = CNNCifar(self.args.num_classes)
        self.model = self.model.cuda()
        self.hpnet = IdentityModel(self.model, self.args.num_users)
        self.server_optimizer = torch.optim.SGD(self.hpnet.w_global.parameters(), lr=self.args.server_lr, momentum=self.args.momentum, weight_decay=0)  # 1e-5
        self.functional, self.gparam = make_functional(self.model)

    def client_update(self, ldr_train, wt, w_local=None):
        numiter = math.ceil(len(ldr_train) / 3)  ## the number of iters for 1 epoch
        loss_list = []
        acc_list = []
        for i in range(numiter):
            images, labels = next(ldr_train.__iter__())
            x = images.cuda()
            y = labels.cuda()

            wt0 = [torch.Tensor(w) for w in wt]

            # 1 step adaptation
            y_pred = self.functional(wt, x)
            loss = self.criteria(y_pred, y)
            grads_ = torch.autograd.grad(loss, wt)
            clip_norm_(grads_, self.args.gradclip)
            wt = [w - self.args.inner_lr * g for w, g in zip(wt, grads_)]

            # gradient 1st
            images, labels = next(ldr_train.__iter__())
            x = images.cuda()
            y = labels.cuda()
            y_pred = self.functional(wt, x)
            loss = self.criteria(y_pred, y)
            grads_1st = torch.autograd.grad(loss, wt)

            # gradient 2nd
            images, labels = next(ldr_train.__iter__())
            x_ = images.cuda()
            y_ = labels.cuda()
            grads_2nd = approximate_hessian(wt0, self.functional, [x_, y_], grads_1st, self.args.delta)
            kl_loss = torch.zeros([1]).cuda()

            clip_norm_(grads_1st, self.args.gradclip)
            clip_norm_(grads_2nd, self.args.gradclip)

            wt = [w - self.args.client_lr * (g1 - self.args.inner_lr * g2) for w, g1, g2 in zip(wt0, grads_1st, grads_2nd)]  # Hessian-Free MAML

            losses = loss + self.args.beta * kl_loss
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
        steps = len(ldr_train) if self.args.adaptation_steps < 0 else self.args.adaptation_steps
        for _ in range(steps):
            images, labels = next(ldr_train.__iter__())
            x = images.cuda()
            y = labels.cuda()

            y_pred = self.functional(wt, x.cuda())
            loss = self.criteria(y_pred, y)
            losses = loss

            grad = torch.autograd.grad(losses, wt)  # FO-MAML
            clip_norm_(grad, self.args.gradclip)
            wt = [w - self.args.inner_lr * g for w, g in zip(wt, grad)]

        return wt  # , mean_loss, mean_acc, torch.zeros(1)
