import torch
from models.BaseModel import BaseModel
from functorch import make_functional
import numpy as np
from torch.utils.data import DataLoader, Dataset
from utils.train_utils import DatasetSplit


from utils.models import CNNCifarVD
from utils.HyperNetClasses import VDModel

class Vd(BaseModel):
    def __init__(self, args, num_classes):
        args.aggw = 0
        args.client_lr = 0
        args.inner_steps = 0
        super().__init__(args)

        self.args.num_classes = num_classes
        self.model = CNNCifarVD(self.args.num_classes, gauss=False)
        self.model = self.model.cuda()
        self.hpnet = VDModel(self.model, self.args.num_users)
        self.server_optimizer = torch.optim.SGD(
            [
                {'params': self.hpnet.w_global.parameters(), 'lr':self.args.server_lr, 'momentum':0.9, 'weight_decay': 0},
                {'params': self.hpnet.w_logits.parameters(), 'lr':self.args.embed_lr, 'momentum':0.1, 'weight_decay':0}
            ]
         )
        self.functional, self.gparam = make_functional(self.model)


    def client_update(self, ldr_train, wt, w_local=None, iter=None, ldr_train_train=None, ldr_train_val=None):
            images, labels = next(ldr_train.__iter__())
            x = images.cuda()
            y = labels.cuda()

            y_pred = self.functional(wt, x.cuda())
            loss = self.criteria(y_pred, y)
            kl_loss = self.model.kl_div(wt[8])

            losses = loss  + self.args.beta * kl_loss

            grad = torch.autograd.grad(losses, wt) # FO-MAML
            wt = [w - self.args.inner_lr * g for w, g in zip(wt, grad)]

            mean_loss = losses.mean()
            mean_acc = y_pred.argmax(1).eq(y).sum().item() / len(y)
            mean_kl =  kl_loss

            return wt, mean_loss, mean_acc, mean_kl, torch.zeros(1)


    def server_aggregation(self, user_idxs, collection):
        collected_weights = collection[0]
        # mudist = torch.cat([collected_weights[i][8].unsqueeze(0) for i in range(self.args.frac_m) ])
        # # vardist = torch.cat([collected_weights[i][10].exp().unsqueeze(0) for i in range(args.frac_m)])
        # vardist = torch.cat([torch.nn.functional.softplus(collected_weights[i][10]).unsqueeze(0) for i in range(self.args.frac_m)])
        # # vardist = torch.cat([collected_weights[i][10].exp().unsqueeze(0) * collected_weights[i][8].unsqueeze(0)**2 for i in range(args.frac_m)])
        # varinv = torch.ones(1).cuda() / (vardist+1e-10)
        # mu = (mudist * varinv).sum(0) / ((varinv).sum(0)+1e-10)

        # for i, idx in enumerate(user_idxs):
        #     # mu_dist.append(collected_weights[i][8].unsqueeze(0))
        #     # var_dist.append( collected_weights[i][10].exp().unsqueeze(0) * collected_weights[i][8].unsqueeze(0)**2 )
        #     # collected_weights[i][10] = torch.log(var + 1e-13)
        #     collected_weights[i][8] = mu

        # for i, idx in enumerate(user_idxs):
        for i, idx in enumerate(user_idxs):
            # delta_theta = [ torch.Tensor((wg - wl).data).detach().clone() for wg , wl in zip(hpnet.w_global, collected_weights[i],)]
            w_local = self.hpnet(idx)
            delta_theta = [ torch.Tensor((wg - wl).data).detach().clone() for wg , wl in zip(w_local, collected_weights[i])]
            # hnet_grads = torch.autograd.grad(w_local, hpnet(idx), delta_theta)
            hnet_grads = torch.autograd.grad(w_local, self.hpnet.parameters(), delta_theta, allow_unused=True)

            # for p, g in zip(hpnet(idx), hnet_grads):
            for (name, p), g in zip(self.hpnet.named_parameters(), hnet_grads):
                if p.grad == None:
                    p.grad = torch.zeros_like(p)
                if g == None:
                    g = torch.zeros_like(p)
                p.grad = p.grad + g / self.args.frac_m

        torch.nn.utils.clip_grad_norm_(self.hpnet.parameters(), 100)

        self.server_optimizer.step()
        self.server_optimizer.zero_grad()


    def client_adapt(self, ldr_train, wt):
            images, labels = next(ldr_train.__iter__())
            x = images.cuda()
            y = labels.cuda()

            y_pred = self.functional(wt, x.cuda())
            loss = self.criteria(y_pred, y)
            kl_loss = self.model.kl_div(wt[8])
            losses = self.criteria(y_pred, y) + self.args.beta * kl_loss

            grad = torch.autograd.grad(losses, wt) # FO-MAML
            wt = [w - self.args.inner_lr * g for w, g in zip(wt, grad)]

            return wt #, mean_loss, mean_acc, torch.zeros(1)
