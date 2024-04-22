import os
import uuid
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from functorch import make_functional
from torch.utils.data import DataLoader, Dataset, TensorDataset

from models.BaseModel import BaseModel
from models.FedAvg import FedAvg
from utils.HyperNetClasses import IdentityModel
from utils.models import CNNCifar
from utils.swa import AveragedModel
from utils.train_utils import DatasetSplit, clip_norm_


class FedBE(BaseModel):
    def __init__(self, args, num_classes):
        args.beta = 0
        args.aggw = 0
        args.client_lr = 0
        args.inner_steps = 0
        args.gradclip = 5
        args.use_cyclical_beta = False
        # teachers
        args.num_teacher_samples = getattr(args, "num_teacher_samples", 10)  # 10 # M (Figure 2)
        args.teacher_sample_mode = "gaussian"  # gaussian, random, (dirichlet)

        # train server (knowledge distillation)
        # args.server_steps = 20 * 4 # (5.1 Baselines, 5.1 FedBE)
        # args.server_bs = args.local_bs # 128 # (5.1 Baselines, 5.1 FedBE)
        args.swa_start_step = getattr(args, "swa_start_step", args.server_steps // 2)
        args.swa_period = getattr(args, "swa_period", 4)

        # distill
        args.no_be = False
        args.client_adapt = False
        args.dist_type = 2
        args.use_w_mean = False
        args.no_distill = False
        args.no_swa = False

        args.sharpen = False

        args.uuid = str(uuid.uuid1())[:8]
        args.exp = f'{getattr(args, "pre", "")}_{args.algorithm}_{args.dataset}_{getattr(args, "alpha", None)}('
        args.exp += f"lepochs:{args.local_epochs}_lbs:{args.local_bs}_"
        # args.exp += f'ilr:{args.inner_lr}_sharpen:{args.sharpen}_'
        args.exp += f"ilr:{args.inner_lr}_"
        args.exp += f"slr:{args.server_lr}_ss:{args.server_steps}_sbs:{args.server_bs}_"
        args.exp += f"slt:{args.swa_lr_type}_min_slr:{args.min_server_lr}_sp:{args.swa_period}_"
        args.exp += f"nts:{args.num_teacher_samples}_rpu:{args.ratio_per_user}_"
        # args.exp += f'teachears:[{int(args.no_client_average)},{int(args.no_clients)},{int(args.no_teacher_samples)}]_'
        # args.exp += f'distill:[{int(args.client_adapt)},{int(args.no_be)},{int(args.dist_type)},{int(args.use_w_mean)},{int(args.no_distill)},{int(args.no_swa)}]_'
        args.exp += f"test_dist:{args.test_dist}_"
        args.exp += f")" + f"_{args.uuid}"
        print(args.exp)
        super().__init__(args)

        self.args.num_classes = num_classes
        self.model = CNNCifar(self.args.num_classes)
        self.model = self.model.cuda()
        self.hpnet = IdentityModel(self.model, self.args.num_users)
        self.server_optimizer = torch.optim.SGD(self.hpnet.w_global.parameters(), lr=1.0, momentum=0.0, weight_decay=0)
        self.functional, self.gparam = make_functional(self.model)

    # TODO: adaptive inner_lr schedule (along global iter(round) not local training, B.2) & weight_decay (resnet32:0.0002, cnn: 0.001, in local training, B.3)
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
            wt = [w - self.args.inner_lr * g for w, g in zip(wt, grad)]

            mean_loss = losses.mean()
            mean_acc = y_pred.argmax(1).eq(y).sum().item() / len(y)
            loss_list.append(mean_loss.item())
            acc_list.append(mean_acc)

        loss_avg = np.mean(loss_list)
        acc_avg = np.mean(acc_list)
        return wt, loss_avg, acc_avg, torch.zeros(1)

    def server_aggregation(self, user_idxs, users_datasize, collection):
        # Construct w_bar & update self.hpnet to w_bar (= FedAvg)
        list_w_i = collection[0]
        iter = collection[2]
        server_dataset = collection[4]
        server_data_idxs = collection[5]

        weights_size = []
        for idx in user_idxs:
            weights_size.append(users_datasize[idx])
        weights = torch.Tensor(weights_size) / sum(weights_size)

        # for i, idx in enumerate(user_idxs):
        for i, idx in enumerate(user_idxs):
            w_local = self.hpnet(idx)
            delta_theta = [torch.Tensor((wg - wl).data).detach().clone() for wg, wl in zip(w_local, list_w_i[i])]
            hnet_grads = torch.autograd.grad(w_local, self.hpnet.parameters(), delta_theta, allow_unused=True)

            for (name, p), g in zip(self.hpnet.named_parameters(), hnet_grads):
                if p.grad == None:
                    p.grad = torch.zeros_like(p)
                if g == None:
                    g = torch.zeros_like(p)
                p.grad = p.grad + g * weights[i]

        torch.nn.utils.clip_grad_norm_(self.hpnet.parameters(), 100)

        self.server_optimizer.step()
        self.server_optimizer.zero_grad()

        if getattr(self.args, "no_be", False):  # FedAvg aggregation only
            return

        # Construct teachers
        w_teachers = self.construct_w_teachers(user_idxs, users_datasize, list_w_i, iter)

        # Use mean of w_teachers as w_bar
        if getattr(self.args, "use_w_mean", False):
            w_mean = [torch.zeros(param.size(), device=param.device) for param in w_teachers[0]]
            for param_idx in range(len(w_mean)):
                for i in range(len(w_teachers)):
                    w_i = w_teachers[i]
                    w_mean[param_idx] += w_i[param_idx] / len(w_teachers)

            for p_swa, p_model in zip(self.hpnet.parameters(), w_mean):
                device = p_swa.device
                p_model_ = p_model.detach().to(device)
                p_swa.detach().copy_(p_model_)

            # test
            for i, j in zip(self.hpnet.parameters(), w_mean):
                assert torch.allclose(i, j)

        if not getattr(self.args, "no_distill", False):
            # Construct unlabeled dataset(U), then turn it into a pseudo-labeled dataset(T)
            images, pseudo_logits, labels = self.construct_pseudo_labeled_dataloader(w_teachers, server_dataset, server_data_idxs, sharpen=getattr(self.args, "sharpen", False), sharpen_temp=0.5)
            pseudo_acc = pseudo_logits.argmax(1).eq(labels).sum().item() / len(labels)
            if iter % 10 == 0:
                self.writer.add_scalar("server/pseudo_acc", pseudo_acc, iter)
            ds = TensorDataset(images, pseudo_logits, labels)
            dataloader = DataLoader(ds, batch_size=self.args.server_bs, shuffle=True)

            if getattr(self.args, "no_swa", False):  # no swa, just adapt w_bar
                wt = self.server_adapt(dataloader, iter)
            else:  # Train self.hpnet (w_bar) with swa
                wt = self.knowledge_distillation(dataloader, iter)

            for p_swa, p_model in zip(self.hpnet.parameters(), wt):
                device = p_swa.device
                p_model_ = p_model.detach().to(device)
                p_swa.detach().copy_(p_model_)

            # test
            for i, j in zip(self.hpnet.parameters(), wt):
                assert torch.allclose(i, j)

        return

    def construct_w_teachers(self, user_idxs, users_datasize, list_w_i, iter=-1):
        Dist = {1: GlobalModelDistribution, 2: GlobalModelDistribution2}[self.args.dist_type]

        w_bar = [torch.Tensor(w) for w in self.hpnet(-1)]  # = deepcopy
        w_teachers = []

        if not getattr(self.args, "no_client_average", False):
            w_teachers.append(w_bar)
        if not getattr(self.args, "no_clients", False):
            w_teachers += list_w_i  # w_i
        if not getattr(self.args, "no_teacher_samples", False):
            global_model_distribution = Dist(user_idxs, users_datasize, w_bar, list_w_i)
            for _ in range(self.args.num_teacher_samples):
                w_sample = global_model_distribution.sample_model(list_w_i, mode=self.args.teacher_sample_mode)
                w_teachers.append(w_sample)

            if iter >= 0 and iter % 10 == 0:
                cv = global_model_distribution.compute_dispersion()
                self.writer.add_scalar("server/samples_cv", cv, iter)

        dist = Dist([i for i in range(len(w_teachers))], [1 for _ in range(len(w_teachers))], w_bar, w_teachers)
        if iter >= 0 and iter % 10 == 0:
            cv = dist.compute_dispersion()
            self.writer.add_scalar("server/teachers_cv", cv, iter)
            print(f"len(w_teachers): {len(w_teachers)}, teachers_cv: {cv}, options: ({not self.args.no_client_average}, {not self.args.no_client_average}, {not self.args.no_client_average})")
        return w_teachers

    @torch.no_grad()
    def construct_pseudo_labeled_dataloader(self, w_teachers, dataset, server_data_idxs, sharpen=True, sharpen_temp=0.5):
        # Construct unlabeled dataset(U), then turn it into a pseudo-labeled dataset(T)
        dataloader = DataLoader(DatasetSplit(dataset, server_data_idxs), batch_size=self.args.server_bs, shuffle=True)

        images_total = []
        labels_total = []
        logits_total = []

        for batch_idx, (images, labels) in enumerate(dataloader):
            x = images.cuda()
            y = labels.cuda()
            images_total.append(x)
            labels_total.append(y)

            logits = []
            for w_teacher in w_teachers:
                logits_teacher = self.functional(w_teacher, x)
                logits.append(logits_teacher)  # (N, num_classes)
            logits = torch.stack(logits, dim=0)  # (num_teachers, N, num_classes)
            logits_total.append(logits)  # (num_teachers, N, num_classes)

        logits_total = torch.cat(logits_total, dim=1)  # (num_teachers, N, num_classes)
        images_total = torch.cat(images_total, dim=0)  # (N, C, H, W)
        labels_total = torch.cat(labels_total, dim=0)  # (N)

        pseudo_logits = logits_total.mean(dim=0)  # (num_teachers, N, num_classes) -> (N, num_classes)

        if sharpen:  # (5.1 Baselines)
            pseudo_logits = torch.maximum(pseudo_logits ** (1 / sharpen_temp), torch.tensor(1e-8))
            pseudo_logits = pseudo_logits / pseudo_logits.sum(dim=-1, keepdim=True)
        return images_total, pseudo_logits, labels_total

    def loss_wrapper(self, logits, pseudo_logits, labels):
        probs = F.softmax(logits, dim=-1)
        P = pseudo_logits
        Q = probs  # logits of wt (softmax X)

        loss = (P * (P.log() - Q.log())).mean()  # KL loss

        return loss

    def loss_wrapper_ce(self, logits, pseudo_logits, labels):
        loss_fn = torch.nn.CrossEntropyLoss()
        pseudo_labels = F.softmax(pseudo_logits, dim=-1)
        loss = loss_fn(logits, pseudo_labels)
        return loss

    def loss_wrapper_ce_labels(self, logits, pseudo_logits, labels):
        loss_fn = torch.nn.CrossEntropyLoss()
        pseudo_labels = pseudo_logits.argmax(1)
        loss = loss_fn(logits, pseudo_labels)
        return loss

    # cyclical schedule with the step size, lr decaying from init_lr to min_lr
    def swa_lr_scheduler(self, step, total_steps, type=1):
        if type == 1:
            return self.swa_lr_scheduler1(step, total_steps)
        elif type == 2:
            return self.swa_lr_scheduler2(step, total_steps)
        elif type == 3:
            return self.args.server_lr, step + 1 == total_steps
        else:
            raise NotImplementedError("invalid lr type")

    def swa_lr_scheduler1(self, step, total_steps):
        init_lr = self.args.server_lr
        min_lr = self.args.min_server_lr
        period = self.args.swa_period
        swa_start_step = self.args.swa_start_step

        if swa_start_step > step:
            return init_lr, False

        step -= swa_start_step
        d = init_lr - min_lr
        lr = init_lr - d / period * (step % period + 1)
        flag_update_params = step % period == period - 1
        return lr, flag_update_params

    # https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/#how-to-use-swa-in-pytorch (Figure 2.)
    def swa_lr_scheduler2(self, step, total_steps):
        init_lr = self.args.server_lr
        min_lr = self.args.min_server_lr
        d = init_lr - min_lr

        s1 = int(total_steps * 0.5)
        s2 = int(total_steps * 0.75)

        if step + 1 < s1:
            return init_lr, False

        if step + 1 < s2:
            lr = init_lr - d / (s2 - s1) * (step + 1 - s1)
            return lr, False

        return min_lr, (step + 1 - s2) % self.args.swa_period == 0

    def knowledge_distillation(self, dataloader, iter):
        swa_model = AveragedModel(deepcopy(self.hpnet))
        wt = [torch.Tensor(w) for w in self.hpnet(-1)]
        # wt == self.hpnet.parameters() == swa_model.module.parameters()

        ws = []

        loss_total = torch.tensor(0.0).cuda()
        acc_total = torch.tensor(0.0).cuda()

        for step in range(self.args.server_steps):
            images, pseudo_logits, labels = next(dataloader.__iter__())
            images = images.cuda()
            pseudo_logits = pseudo_logits.cuda()
            labels = labels.cuda()

            logits = self.functional(wt, images)
            # loss = self.loss_wrapper(logits, pseudo_logits, labels)
            loss = self.loss_wrapper_ce(logits, pseudo_logits, labels)
            # loss = self.loss_wrapper_ce_labels(logits, pseudo_logits, labels)
            acc = logits.argmax(1).eq(pseudo_logits.argmax(1)).sum().item() / len(labels)
            loss_total += loss.detach()
            acc_total += acc

            grad = torch.autograd.grad(loss, wt)
            clip_norm_(grad, self.args.gradclip)

            lr, flag_update_params = self.swa_lr_scheduler(step, self.args.server_steps, type=self.args.swa_lr_type)
            if iter == 0:
                self.writer.add_scalar("server/swa_lr", lr, step)
                self.writer.add_scalar("server/flag_update_params", int(flag_update_params), step)
            wt = [w - lr * g for w, g in zip(wt, grad)]

            if flag_update_params:
                swa_model.update_parameters(wt)
                ws.append([torch.Tensor(w) for w in wt])

        loss_mean = loss_total / self.args.server_steps
        acc_mean = acc_total / self.args.server_steps

        if iter % 10 == 0:
            self.writer.add_scalar("server/loss_swa", loss_mean.item(), iter)
            self.writer.add_scalar("server/acc_for_pseudo_labels", acc_mean, iter)

        # check
        w_mean = [torch.zeros_like(w) for w in wt]
        for j in range(len(w_mean)):
            for i in range(len(ws)):
                w_mean[j] += ws[i][j]
            w_mean[j] /= len(ws)
        for i, j in zip(w_mean, swa_model.parameters()):
            assert torch.allclose(i, j)

        print(f"iter: {iter}, n_averaged: {swa_model.n_averaged}, len(ws): {len(ws)}")
        # return swa_model

        wt = [torch.Tensor(w) for w in swa_model.parameters()]
        return wt

    def server_adapt(self, dataloader, iter):
        wt = [torch.Tensor(w) for w in self.hpnet(-1)]

        loss_total = torch.tensor(0.0).cuda()
        acc_total = torch.tensor(0.0).cuda()

        for step in range(self.args.server_steps):
            images, pseudo_logits, labels = next(dataloader.__iter__())
            images = images.cuda()
            pseudo_logits = pseudo_logits.cuda()
            labels = labels.cuda()

            logits = self.functional(wt, images)
            # loss = self.loss_wrapper(logits, pseudo_logits, labels)
            loss = self.loss_wrapper_ce(logits, pseudo_logits, labels)
            # loss = self.loss_wrapper_ce_labels(logits, pseudo_logits, labels)
            acc = logits.argmax(1).eq(pseudo_logits.argmax(1)).sum().item() / len(labels)
            loss_total += loss.detach()
            acc_total += acc

            grad = torch.autograd.grad(loss, wt)
            clip_norm_(grad, self.args.gradclip)

            wt = [w - self.args.server_lr * g for w, g in zip(wt, grad)]

        loss_mean = loss_total / self.args.server_steps
        acc_mean = acc_total / self.args.server_steps

        if iter % 10 == 0:
            self.writer.add_scalar("server/loss_adapt", loss_mean.item(), iter)
            self.writer.add_scalar("server/acc_for_pseudo_labels", acc_mean, iter)

        return wt

    def client_adapt(self, ldr_train, wt):
        if not getattr(self.args, "client_adapt", False):
            return wt
        else:
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


# https://github.com/hongyouc/FedBE/blob/c7c41728dfb50f30c2844f3b704f8230ba251f50/swag.py#L50
class GlobalModelDistribution(torch.nn.Module):  # without BatchNorm
    def __init__(self, user_idxs, users_datasize, w_bar, list_w_i, var_clamp=1e-5, concentrate_num=1):
        self.user_idxs = user_idxs
        self.users_datasize = users_datasize
        self.list_w_i = list_w_i
        self.w_bar = w_bar
        self.var_clamp = var_clamp
        self.concentrate_num = concentrate_num
        self.var_scale = 0.1
        self.swag_stepsize = 1.0

        self.weights = self.compute_weights()
        self.w_avg, self.w_sq_avg, self.w_norm = self.compute_mean_sq()
        self.w_var = self.compute_var(self.w_avg, self.w_sq_avg)

    def compute_weights(self):
        weights_size = []

        for idx in self.user_idxs:
            weights_size.append(self.users_datasize[idx])
        weights = torch.Tensor(weights_size) / sum(weights_size)

        return weights

    def compute_mean_sq(self):
        w_avg = [torch.zeros(param.size(), device=param.device) for param in self.list_w_i[0]]
        w_sq_avg = [torch.zeros(param.size(), device=param.device) for param in self.list_w_i[0]]
        w_norm = [torch.tensor(0.0, device=param.device) for param in self.list_w_i[0]]

        for param_idx in range(len(w_avg)):
            for i in range(len(self.list_w_i)):
                w_i = self.list_w_i[i]
                grad = w_i[param_idx] - self.w_bar[param_idx]
                norm = torch.norm(grad, p=2)

                grad = grad / norm
                sq_grad = grad**2

                w_avg[param_idx] += grad * self.weights[i]
                w_sq_avg[param_idx] += sq_grad * self.weights[i]
                w_norm[param_idx] += norm * self.weights[i]

        return w_avg, w_sq_avg, w_norm

    def compute_var(self, mean, sq_mean):
        assert len(mean) == len(sq_mean)
        list_var = []

        for i in range(len(mean)):
            var = torch.clamp(sq_mean[i] - mean[i] ** 2, self.var_clamp)
            list_var.append(var)

        return list_var

    def compute_dispersion(self):
        # return standard deviation / mean (coefficient of variation)
        cv = 0.0
        for i in range(len(self.w_var)):
            mean = torch.clamp(torch.abs(self.w_avg[i]), 1e-6)
            var = torch.clamp(self.w_var[i], 1e-6)
            cv += (torch.sqrt(var) / mean).mean()
        return cv / len(self.w_var)

    def sample_model(self, list_w_i, mode="gaussian"):
        if mode == "gaussian":
            mean_grad = [p.clone().detach().requires_grad_(True) for p in self.w_avg]
            for i in range(self.concentrate_num):
                for param_idx in range(len(self.w_avg)):
                    mean = self.w_avg[param_idx]
                    var = torch.clamp(self.w_var[param_idx], 1e-6)

                    eps = torch.randn_like(mean)
                    sample_grad = mean + torch.sqrt(var) * eps * self.var_scale
                    mean_grad[param_idx] = (i * mean_grad[param_idx] + sample_grad) / (i + 1)

            for param_idx in range(len(self.w_avg)):
                mean_grad[param_idx] = mean_grad[param_idx] * self.swag_stepsize * self.w_norm[param_idx] + self.w_bar[param_idx]

            return mean_grad

        else:
            raise NotImplementedError("mode should be gaussian or random")


class GlobalModelDistribution2(torch.nn.Module):  # without BatchNorm
    def __init__(self, user_idxs, users_datasize, w_bar, list_w_i, var_clamp=1e-5, concentrate_num=1):
        self.user_idxs = user_idxs
        self.users_datasize = users_datasize
        self.list_w_i = list_w_i
        self.w_bar = w_bar
        self.var_clamp = var_clamp
        self.concentrate_num = concentrate_num
        self.var_scale = 0.1
        self.swag_stepsize = 1.0

        self.weights = self.compute_weights()
        self.w_mean = self.compute_mean()
        self.w_var = self.compute_var()

    def compute_weights(self):
        weights_size = []

        for idx in self.user_idxs:
            weights_size.append(self.users_datasize[idx])
        weights = torch.Tensor(weights_size) / sum(weights_size)

        return weights

    def compute_mean(self):
        w_mean = [torch.zeros(param.size(), device=param.device) for param in self.list_w_i[0]]

        for param_idx in range(len(w_mean)):
            for i in range(len(self.list_w_i)):
                w_i = self.list_w_i[i]
                w_mean[param_idx] += w_i[param_idx] * self.weights[i]

        return w_mean

    def compute_var(self):
        w_var = [torch.zeros(param.size(), device=param.device) for param in self.w_mean]

        for param_idx in range(len(w_var)):
            for i in range(len(self.list_w_i)):
                w_i = self.list_w_i[i]
                w_var[param_idx] += (w_i[param_idx] - self.w_mean[param_idx]) ** 2 * self.weights[i]

            w_var[param_idx] = torch.clamp(w_var[param_idx], 1e-6)

        return w_var

    def compute_dispersion(self):
        # return standard deviation / mean (coefficient of variation)
        cv = 0.0
        for i in range(len(self.w_var)):
            mean = torch.clamp(torch.abs(self.w_mean[i]), 1e-6)
            var = self.w_var[i]
            cv += (torch.sqrt(var) / mean).mean()
        return cv / len(self.w_var)

    def sample_model(self, list_w_i, mode="gaussian"):
        if mode == "gaussian":
            w_sample = [torch.zeros(param.size(), device=param.device) for param in self.w_mean]
            for param_idx in range(len(self.w_mean)):
                mean = self.w_mean[param_idx]
                var = self.w_var[param_idx]

                eps = torch.randn_like(mean)
                w_sample[param_idx] += mean + torch.sqrt(var) * self.var_scale * eps

            return w_sample

        else:
            raise NotImplementedError("mode should be gaussian or random")
