import os
import uuid

import numpy as np
import torch
from functorch import make_functional
from torch.utils.data import DataLoader, Dataset

from models.BaseModel import BaseModel
from utils.HyperNetClasses import SnipModel
from utils.models import CNNCifar
from utils.train_utils import DatasetSplit, clip_norm_


class ReptileSnip(BaseModel):
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
        args.exp += f"ilr:{args.inner_lr}_slr:{args.server_lr}_mom:{args.momentum}_"
        args.exp += f"adepochs:{args.adaptation_epochs}_adsteps:{args.adaptation_steps}_adbs:{args.adaptation_bs}_"
        args.exp += f"test_dist:{args.test_dist}_"
        args.exp += f")" + f"_{args.uuid}"
        print(args.exp)

        super().__init__(args)

        self.args.num_classes = num_classes
        self.model = CNNCifar(self.args.num_classes)
        self.model = self.model.cuda()
        self.hpnet = SnipModel(self.model, self.args.num_users, self.args.droplayer)
        # self.server_optimizer = torch.optim.SGD(self.hpnet.w_global.parameters(), lr=self.args.server_lr, momentum=.9, weight_decay=0) # 1e-5
        self.server_optimizer = torch.optim.SGD(self.hpnet.w_global.parameters(), lr=self.args.server_lr, momentum=self.args.momentum, weight_decay=0)  # 1e-5
        self.functional, self.gparam = make_functional(self.model)

    def client_update(self, ldr_train, wt, w_local=None):
        loss_list = []
        acc_list = []

        for images, labels in ldr_train:
            # images, labels = next(ldr_train.__iter__())
            x = images.cuda()
            y = labels.cuda()

            y_pred = self.functional(wt, x.cuda())
            loss = self.criteria(y_pred, y)
            losses = loss

            grad = torch.autograd.grad(losses, wt)  # FO-MAML
            clip_norm_(grad, self.args.gradclip)
            # wt = [w - self.args.inner_lr * g for w, g in zip(wt, grad)]
            # wt = [w - self.args.inner_lr * g * self.hpnet.mask if i==8 else w - self.args.inner_lr *g for i, (w,g) in enumerate(zip(wt, grad))] # masking for pruned weight
            wt = [w - self.args.inner_lr * g * self.hpnet.mask[i] if i in self.hpnet.droplayer else w - self.args.inner_lr * g for i, (w, g) in enumerate(zip(wt, grad))]  # masking for pruned weight

            mean_loss = losses.mean()
            mean_acc = y_pred.argmax(1).eq(y).sum().item() / len(y)
            # val_kl += kl_loss

            loss_list.append(mean_loss.item())
            acc_list.append(mean_acc)

        mean_loss = np.mean(loss_list)
        mean_acc = np.mean(acc_list)

        return wt, mean_loss, mean_acc, torch.zeros(1)

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
        # for images, labels in ldr_train:
        #     #images, labels = next(ldr_train.__iter__())
        #     x = images.cuda()
        #     y = labels.cuda()

        #     y_pred = self.functional(wt, x.cuda())
        #     loss = self.criteria(y_pred, y)
        #     losses = loss
        #     # losses = criteria(y_pred, y) + args.beta * kl_loss

        #     grad = torch.autograd.grad(losses, wt) # FO-MAML
        #     clip_norm_(grad, self.args.gradclip)
        #     wt = [w - self.args.inner_lr * g for w, g in zip(wt, grad)]

        #     # mean_loss = losses.mean()
        #     # mean_acc = y_pred.argmax(1).eq(y).sum().item() / len(y)

        # return wt #, mean_loss, mean_acc, torch.zeros(1)

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
            # wt = [w - self.args.inner_lr * g * self.hpnet.mask if i==8 else w - self.args.inner_lr *g for i, (w,g) in enumerate(zip(wt, grad))]

        return wt

    def snip(self, user_idxs, users_datasize, collection):
        collected_weights = collection[0]

        weights_size = []
        for idx in user_idxs:
            weights_size.append(users_datasize[idx])
        weights = torch.Tensor(weights_size) / sum(weights_size)

        grad_dict = dict()
        weight_dict = dict()

        for i, idx in enumerate(user_idxs):
            w_local = self.hpnet(idx)
            delta_theta = [torch.Tensor((wg - wl).data).detach().clone() for wg, wl in zip(w_local, collected_weights[i])]
            hnet_grads = torch.autograd.grad(w_local, self.hpnet.parameters(), delta_theta, allow_unused=True)

            for mi, ((name, p), g) in enumerate(zip(self.hpnet.named_parameters(), hnet_grads)):
                # if name == 'w_global.8':
                if mi in self.hpnet.droplayer:
                    if i == 0:
                        grad_dict[name] = torch.zeros_like(p)
                    if g == None:
                        g = torch.zeros_like(p)
                    grad_dict[name] = grad_dict[name] + g / self.args.inner_lr * weights[i]

        for mi, (name, p) in enumerate(self.hpnet.named_parameters()):
            # if name == 'w_global.8':
            if mi in self.hpnet.droplayer:
                weight_dict[name] = p.clone()

        abs_all_wg = None
        for (name_w, w), (name_g, g) in zip(weight_dict.items(), grad_dict.items()):
            assert name_w == name_g
            if abs_all_wg is None:
                abs_all_wg = (w * g).view(-1).abs()
            else:
                abs_all_wg = torch.cat([abs_all_wg, (w * g).view(-1).abs()], dim=0)

        threshold = abs_all_wg.sort(descending=True)[0][int((1.0 - self.args.sparsity) * abs_all_wg.nelement())]

        for (name_w, w), (name_g, g) in zip(weight_dict.items(), grad_dict.items()):
            assert name_w == name_g
            # if name_w == 'w_global.8':
            layer_id = int(name_w[-1])
            if layer_id in self.hpnet.droplayer:
                abs_layer_wg = (w * g).view(-1).abs()
                mask = abs_layer_wg >= threshold
                self.hpnet.mask[layer_id] = mask.reshape(self.hpnet.mask[layer_id].shape).clone()
