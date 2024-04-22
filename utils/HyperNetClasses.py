# import higher
import copy
import typing

import numpy as np
import torch

from utils.models import *

# from _utils import intialize_parameters, vector_to_list_parameters


class IdentityModel(torch.nn.Module):
    def __init__(self, base_net: torch.nn.Module, num_users, **kwargs) -> None:
        """
        Args:
            base_net: the base network
        """
        super(IdentityModel, self).__init__()

        # dict of parameters of based network
        self.base_state_dict = base_net.state_dict()

        # initialize parameters
        self.w_global = torch.nn.ParameterList([torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" not in n])

    def forward(self, idx) -> typing.List[torch.Tensor]:
        """Output the parameters of the base network in list format to pass into functional of functorch"""
        out = []
        out.append(self.w_global[0])
        out.append(self.w_global[1])
        out.append(self.w_global[2])
        out.append(self.w_global[3])
        out.append(self.w_global[4])
        out.append(self.w_global[5])
        out.append(self.w_global[6])
        out.append(self.w_global[7])
        out.append(self.w_global[8])
        out.append(self.w_global[9])
        out.append(self.w_global[10])  # added
        out.append(self.w_global[11])  # added

        return out


class SnipModel(torch.nn.Module):
    def __init__(self, base_net: torch.nn.Module, num_users, droplayer, **kwargs) -> None:
        """
        Args:
            base_net: the base network
        """
        super(SnipModel, self).__init__()

        # dict of parameters of based network
        self.base_state_dict = base_net.state_dict()

        # initialize parameters
        self.w_global = torch.nn.ParameterList([torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" not in n])

        self.droplayer = droplayer
        for i in range(len(self.droplayer)):
            self.droplayer[i] = int(self.droplayer[i])

        self.mask = dict()
        for i in droplayer:
            self.mask[i] = torch.ones_like(self.w_global[i])

    def forward(self, idx) -> typing.List[torch.Tensor]:
        """Output the parameters of the base network in list format to pass into functional of functorch"""
        out = []

        for i in range(12):
            if i in self.droplayer:
                out.append(self.w_global[i] * self.mask[i])
            else:
                out.append(self.w_global[i])

        return out


class VDModel(torch.nn.Module):
    """A simple neural network that simulate the
    reparameterization trick. Its parameters are
    the mean and std-vector
    """

    def __init__(self, base_net: torch.nn.Module, num_users, **kwargs) -> None:
        """
        Args:
            base_net: the base network
        """
        super(VDModel, self).__init__()

        # dict of parameters of based network
        self.base_state_dict = base_net.state_dict()

        # initialize parameters
        self.w_global = torch.nn.ParameterList([torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" not in n])

        w_logits = [torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" in n]

        self.w_logits = []
        for i in range(1):  # Simple VD
            # for i in range(num_users):   # Ensemble VD
            self.w_logits.append(w_logits[0].clone().detach().requires_grad_(True))
        self.w_logits = torch.nn.ParameterList(self.w_logits)

        self.w_logits_temp = torch.nn.Parameter(torch.ones_like(w_logits[0]) * -8.5)

    # def forward(self, idx, ood=False) -> typing.List[torch.Tensor]:
    def forward(self, idx, ood=False, drop_prob=None) -> typing.List[torch.Tensor]:
        """Output the parameters of the base network in list format to pass into higher monkeypatch"""
        out = []
        # for i, n in enumerate(self.base_state_dict.keys()):
        out.append(self.w_global[0])
        out.append(self.w_global[1])
        out.append(self.w_global[2])
        out.append(self.w_global[3])
        out.append(self.w_global[4])
        out.append(self.w_global[5])
        out.append(self.w_global[6])
        out.append(self.w_global[7])

        out.append(self.w_global[8])
        out.append(self.w_global[9])

        out.append(self.w_logits[0].clamp(-8.5, 8.5))  # Simple VD

        # if ood == False:
        #     out.append(self.w_logits[idx].clamp(-8,8)) # Ensemble VD
        # else:
        #     out.append(self.w_logits_temp.cuda())

        out.append(self.w_global[10])  # added
        out.append(self.w_global[11])  # added

        return out


class VDModelEm(torch.nn.Module):
    """A simple neural network that simulate the
    reparameterization trick. Its parameters are
    the mean and std-vector
    """

    def __init__(self, base_net: torch.nn.Module, num_users, **kwargs) -> None:
        """
        Args:
            base_net: the base network
        """
        super(VDModelEm, self).__init__()

        # dict of parameters of based network
        self.base_state_dict = base_net.state_dict()

        # initialize parameters
        self.w_global = torch.nn.ParameterList([torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" not in n])

        w_logits = [torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" in n]

        self.w_logits = []
        # for i in range(1):  # Simple VD
        for i in range(num_users):  # Ensemble VD
            self.w_logits.append(w_logits[0].clone().detach().requires_grad_(True))
        self.w_logits = torch.nn.ParameterList(self.w_logits)

        self.w_logits_temp = torch.nn.Parameter(torch.ones_like(w_logits[0]) * -8.5)

    # def forward(self, idx, ood=False) -> typing.List[torch.Tensor]:
    def forward(self, idx, ood=False, drop_prob=None) -> typing.List[torch.Tensor]:
        """Output the parameters of the base network in list format to pass into higher monkeypatch"""
        out = []
        # for i, n in enumerate(self.base_state_dict.keys()):
        out.append(self.w_global[0])
        out.append(self.w_global[1])
        out.append(self.w_global[2])
        out.append(self.w_global[3])
        out.append(self.w_global[4])
        out.append(self.w_global[5])
        out.append(self.w_global[6])
        out.append(self.w_global[7])

        out.append(self.w_global[8])
        out.append(self.w_global[9])

        # out.append(self.w_logits[0].clamp(-8.5,8.5)) # Simple VD

        if ood == False:
            out.append(self.w_logits[idx].clamp(-8.5, 8.5))  # Ensemble VD
        else:
            out.append(self.w_logits_temp.cuda())

        out.append(self.w_global[10])  # added
        out.append(self.w_global[11])  # added

        return out


class NVDPModel(torch.nn.Module):
    """A simple neural network that simulate the
    reparameterization trick. Its parameters are
    the mean and std-vector
    """

    def __init__(self, base_net: torch.nn.Module, num_users, *args) -> None:
        """
        Args:
            base_net: the base network
        """
        super(NVDPModel, self).__init__()

        # dict of parameters of based network
        self.base_state_dict = base_net.state_dict()

        embed_dim = int(1 + num_users / 4)
        # embed_dim = int(1 + num_users / 2) # ! increase model size (multi_6)
        # embed_dim = 256
        # hidden_dim = 100
        # num_layers = 2
        hidden_dim = args[1]
        # hidden_dim = args[1]*2 # ! increase model size (multi_6)
        num_layers = args[0]
        # num_layers = args[0]*2 # ! increase model size (multi_6)

        w_logits = [torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" in n]
        self.output_dim = w_logits[0].shape[0]
        self.output_dim2 = w_logits[0].shape[1]

        self.embeds = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        # self.embeds = torch.nn.utils.parametrizations.orthogonal(embeds, name='weight')

        modules = []
        modules.append(torch.nn.Linear(embed_dim, hidden_dim))
        modules.append(torch.nn.LeakyReLU(0.1, inplace=True))
        for j in range(0, num_layers):
            modules.append(torch.nn.Linear(hidden_dim, hidden_dim))
            modules.append(torch.nn.LeakyReLU(0.1, inplace=True))
        modules.append(torch.nn.Linear(hidden_dim, self.output_dim * self.output_dim2))
        self.w_logits = torch.nn.Sequential(*modules)

        # self.w_logits = torch.nn.Sequential(
        #     torch.nn.Linear(embed_dim, hidden_dim),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     # torch.nn.Linear(hidden_dim, hidden_dim),
        #     # torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(hidden_dim, self.output_dim * self.output_dim2)
        # )

        self.w_logits_temp = torch.nn.Parameter(torch.ones(self.output_dim, self.output_dim2) * -8.5)

        self.w_logits.apply(init_weights)

        # initialize parameters
        self.w_global = torch.nn.ParameterList(
            [torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" not in n and "tau" not in n]
        )

    def forward(self, idx, ood=False, drop_prob=None) -> typing.List[torch.Tensor]:
        """Output the parameters of the base network in list format to pass into higher monkeypatch"""

        out = []
        # for i, n in enumerate(self.base_state_dict.keys()):
        out.append(self.w_global[0])
        out.append(self.w_global[1])
        out.append(self.w_global[2])
        out.append(self.w_global[3])
        out.append(self.w_global[4])
        out.append(self.w_global[5])
        out.append(self.w_global[6])
        out.append(self.w_global[7])

        if ood == False:
            emd = self.embeds(torch.tensor(idx, dtype=torch.long))
            logits = self.w_logits(emd).cuda().view(self.output_dim, self.output_dim2).clamp(-8.5, 8.5) - 8.5

            if drop_prob is not None:
                alpha = logits.exp() / (self.w_global[8] ** 2 + 1e-10)
                probs = alpha / (1 + alpha)
                masks = (1 - (probs > drop_prob) * 1).detach()
            else:
                masks = torch.ones_like(self.w_global[8])
        else:
            masks = torch.ones_like(self.w_global[8])

        out.append(self.w_global[8] * masks)
        out.append(self.w_global[9])

        if ood == False:
            # emd = self.embeds(torch.tensor(idx, dtype=torch.long))
            # out.append(self.w_logits(emd).cuda().view(self.output_dim,self.output_dim2).clamp(-8,8) - 8)
            out.append(logits)
        else:
            # emd_all = self.embeds(torch.arange(0,100))  #TODO mean_fec
            # out.append(self.w_logits(emd_all.mean(0)).view(self.output_dim, self.output_dim2).clamp(-8,8).cuda() - 8)
            out.append(self.w_logits_temp.cuda())

        out.append(self.w_global[10])
        out.append(self.w_global[11])

        # if self.prior is True:
        #     if ood == False:
        #         emd_all = self.embeds(torch.arange(0,100))
        #         out.append(self.w_logits(emd_all.mean(0)).view(self.output_dim, self.output_dim2).clamp(-8,8).cuda() - 8)
        #     else:
        #         emd_all = self.embeds(torch.arange(0,100))
        #         out.append(self.w_logits(emd_all.mean(0)).view(self.output_dim, self.output_dim2).clamp(-8,8).cuda() - 8)
        #         # out.append(self.w_logits_temp.cuda())

        return out


class NVDPModelPlus(torch.nn.Module):
    """A simple neural network that simulate the
    reparameterization trick. Its parameters are
    the mean and std-vector
    """

    def __init__(self, base_net: torch.nn.Module, num_users, *args) -> None:
        """
        Args:
            base_net: the base network
        """
        super(NVDPModelPlus, self).__init__()

        # dict of parameters of based network
        self.base_state_dict = base_net.state_dict()

        embed_dim = int(1 + num_users / 4)
        # embed_dim = 256
        # hidden_dim = 100
        # num_layers = 2
        hidden_dim = args[1]
        num_layers = args[0]

        # def kl_div(model):
        #     kl = 0
        #     numl = 0
        #     for module in model.modules():
        #         if isinstance(module, LinearNVDPGDRep):
        #             kl += module.kl_div()
        #             numl += 1
        #     return kl / numl

        self.embeds = torch.nn.Embedding(num_embeddings=num_users, embedding_dim=embed_dim)
        # self.embeds = torch.nn.utils.parametrizations.orthogonal(embeds, name='weight')

        w_logits = [torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" in n]
        self.w_logits_temp = [nn.Parameter(torch.Tensor(p).clone().detach()) for p in w_logits]

        modules = []
        modules.append(torch.nn.Linear(embed_dim, hidden_dim))
        # modules.append(torch.nn.LeakyReLU(0.1, inplace=True))
        for j in range(0, num_layers):
            modules.append(torch.nn.LeakyReLU(0.1, inplace=True))
            modules.append(torch.nn.Linear(hidden_dim, hidden_dim))
            # modules.append(torch.nn.LeakyReLU(0.1, inplace=True))
        self.w_feclayer = torch.nn.Sequential(*modules)

        self.w_logits = []
        for i in range(0, len(w_logits)):
            output_dim = w_logits[i].shape[0]
            output_dim2 = w_logits[i].shape[1]
            modules2 = []
            modules2.append(torch.nn.Linear(hidden_dim, output_dim * output_dim2))
            self.w_logits.append(torch.nn.Sequential(*modules2))

        # self.w_logits = torch.nn.Sequential(*modules)

        # self.w_logits = torch.nn.Sequential(
        #     torch.nn.Linear(embed_dim, hidden_dim),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     # torch.nn.Linear(hidden_dim, hidden_dim),
        #     # torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(hidden_dim, self.output_dim * self.output_dim2)
        # )

        # self.w_logits_temp = torch.nn.Parameter(torch.ones(self.output_dim, self.output_dim2)*-8.5)

        # self.w_logits.apply(init_weights)
        for fun in self.w_logits:
            fun.apply(init_weights)

        # initialize parameters
        self.w_global = torch.nn.ParameterList(
            [torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" not in n and "tau" not in n]
        )

    def forward(self, idx, ood=False, drop_prob=None) -> typing.List[torch.Tensor]:
        """Output the parameters of the base network in list format to pass into higher monkeypatch"""

        # ['features.0.0.weight', #0
        #  'features.0.0.bias', #1
        #  'features.1.0.weight', #2
        #  'features.1.0.bias', #3
        #  'features.2.0.weight', #4
        #  'features.2.0.bias', #5

        #  'linear0.weight', #6
        #  'linear0.bias', #7
        #  'linear0.logit', #8

        #  'linear1.weight', #9
        #  'linear1.bias', #10
        #  'linear1.logit', #11

        #  'linear2.weight', #12
        #  'linear2.bias', #13
        #  'linear2.logit'] #14

        out = []
        # for i, n in enumerate(self.base_state_dict.keys()):
        out.append(self.w_global[0])
        out.append(self.w_global[1])
        out.append(self.w_global[2])
        out.append(self.w_global[3])
        out.append(self.w_global[4])
        out.append(self.w_global[5])

        out.append(self.w_global[6])
        out.append(self.w_global[7])

        if ood == False:
            emd = self.embeds(torch.tensor(idx, dtype=torch.long))
            out.append(self.w_logits[0](self.w_feclayer(emd)).cuda().view(self.w_logits_temp[0].size()).clamp(-8.5, 8.5) - 8.5)
        else:
            out.append(self.w_logits_temp[0])
        # if ood == False:
        #     emd = self.embeds(torch.tensor(idx, dtype=torch.long))
        #     logits= (self.w_logits(emd).cuda().view(self.output_dim,self.output_dim2).clamp(-8.5,8.5) - 8.5)

        #     if drop_prob is not None:
        #         alpha = logits.exp() / (self.w_global[8]**2 + 1e-10)
        #         probs = alpha / (1+alpha)
        #         masks = (1 - (probs > drop_prob)*1).detach()
        #     else:
        #         masks = torch.ones_like(self.w_global[8])
        # else:
        #     masks = torch.ones_like(self.w_global[8])

        # out.append(self.w_global[8] * masks)
        # out.append(self.w_global[9])

        # if ood == False:
        #     # emd = self.embeds(torch.tensor(idx, dtype=torch.long))
        #     # out.append(self.w_logits(emd).cuda().view(self.output_dim,self.output_dim2).clamp(-8,8) - 8)
        #     out.append(logits)
        # else:
        #     # emd_all = self.embeds(torch.arange(0,100))  #TODO mean_fec
        #     # out.append(self.w_logits(emd_all.mean(0)).view(self.output_dim, self.output_dim2).clamp(-8,8).cuda() - 8)
        #     out.append(self.w_logits_temp.cuda())

        out.append(self.w_global[8])

        out.append(self.w_global[9])
        # out.append(self.w_global[11])
        if ood == False:
            emd = self.embeds(torch.tensor(idx, dtype=torch.long))
            out.append(self.w_logits[1](self.w_feclayer(emd)).cuda().view(self.w_logits_temp[1].size()).clamp(-8.5, 8.5) - 8.5)
        else:
            out.append(self.w_logits_temp[1])

        out.append(self.w_global[10])

        out.append(self.w_global[11])
        # out.append(self.w_global[14])
        if ood == False:
            emd = self.embeds(torch.tensor(idx, dtype=torch.long))
            out.append(self.w_logits[2](self.w_feclayer(emd)).cuda().view(self.w_logits_temp[2].size()).clamp(-8.5, 8.5) - 8.5)
        else:
            out.append(self.w_logits_temp[2])

        # if self.prior is True:
        #     if ood == False:
        #         emd_all = self.embeds(torch.arange(0,100))
        #         out.append(self.w_logits(emd_all.mean(0)).view(self.output_dim, self.output_dim2).clamp(-8,8).cuda() - 8)
        #     else:
        #         emd_all = self.embeds(torch.arange(0,100))
        #         out.append(self.w_logits(emd_all.mean(0)).view(self.output_dim, self.output_dim2).clamp(-8,8).cuda() - 8)
        #         # out.append(self.w_logits_temp.cuda())

        return out


class NVDPModelV2(torch.nn.Module):
    """A simple neural network that simulate the
    reparameterization trick. Its parameters are
    the mean and std-vector
    """

    def __init__(self, base_net: torch.nn.Module, num_users, ood_users, *args) -> None:
        """
        Args:
            base_net: the base network
        """
        super(NVDPModelV2, self).__init__()

        # dict of parameters of based network
        self.base_state_dict = base_net.state_dict()

        # embed_dim = int(1 + num_users / 4)
        embed_dim = 1024
        # hidden_dim = 100
        # num_layers = 2
        hidden_dim = args[1]
        num_layers = args[0]

        w_logits = [torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" in n]
        self.output_dim = w_logits[0].shape[0]
        self.output_dim2 = w_logits[0].shape[1]

        # self.embeds = torch.nn.Embedding(num_embeddings=num_users,embedding_dim=embed_dim)
        # self.embeds = torch.nn.utils.parametrizations.orthogonal(embeds, name='weight')

        self.embeds = torch.zeros(num_users + ood_users, embed_dim)

        modules = []
        modules.append(torch.nn.Linear(embed_dim, hidden_dim))
        modules.append(torch.nn.LeakyReLU(0.1, inplace=True))
        for j in range(0, num_layers):
            modules.append(torch.nn.Linear(hidden_dim, hidden_dim))
            modules.append(torch.nn.LeakyReLU(0.1, inplace=True))
        modules.append(torch.nn.Linear(hidden_dim, self.output_dim * self.output_dim2))
        self.w_logits = torch.nn.Sequential(*modules)

        # self.w_logits = torch.nn.Sequential(
        #     torch.nn.Linear(embed_dim, hidden_dim),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(hidden_dim, hidden_dim),
        #     torch.nn.LeakyReLU(0.1, inplace=True),
        #     # torch.nn.Linear(hidden_dim, hidden_dim),
        #     # torch.nn.LeakyReLU(0.1, inplace=True),
        #     torch.nn.Linear(hidden_dim, self.output_dim * self.output_dim2)
        # )

        self.w_logits_temp = torch.nn.Parameter(torch.ones(self.output_dim, self.output_dim2) * -8.5)

        self.w_logits.apply(init_weights)

        # initialize parameters
        self.w_global = torch.nn.ParameterList(
            [torch.Tensor(p).clone().detach().requires_grad_(True) for n, p in base_net.named_parameters() if "logit" not in n and "tau" not in n]
        )

    # def forward(self, idx, feature_stat=None, ood=False, drop_prob=None) -> typing.List[torch.Tensor]:
    def forward(self, idx, ood=False, drop_prob=None) -> typing.List[torch.Tensor]:
        """Output the parameters of the base network in list format to pass into higher monkeypatch"""

        out = []
        # for i, n in enumerate(self.base_state_dict.keys()):
        out.append(self.w_global[0])
        out.append(self.w_global[1])
        out.append(self.w_global[2])
        out.append(self.w_global[3])
        out.append(self.w_global[4])
        out.append(self.w_global[5])
        out.append(self.w_global[6])
        out.append(self.w_global[7])

        # if ood == False:
        #     emd = self.embeds[idx]
        #     logits= (self.w_logits(emd).cuda().view(self.output_dim,self.output_dim2).clamp(-8.5,8.5) - 8.5)

        #     if drop_prob is not None:
        #         alpha = logits.exp() / (self.w_global[8]**2 + 1e-10)
        #         probs = alpha / (1+alpha)
        #         masks = (1 - (probs > drop_prob)*1).detach()
        #     else:
        #         masks = torch.ones_like(self.w_global[8])
        # else:
        #     if feature_stat is None:
        #         masks = torch.ones_like(self.w_global[8])
        #     else:
        #         logits= (self.w_logits(feature_stat.cpu()).cuda().view(self.output_dim,self.output_dim2).clamp(-8.5,8.5) - 8.5)

        #         if drop_prob is not None:
        #             alpha = logits.exp() / (self.w_global[8]**2 + 1e-10)
        #             probs = alpha / (1+alpha)
        #             masks = (1 - (probs > drop_prob)*1).detach()
        #         else:
        #             masks = torch.ones_like(self.w_global[8])

        if ood == False:
            emd = self.embeds[idx]
            logits = self.w_logits(emd).cuda().view(self.output_dim, self.output_dim2).clamp(-8.5, 8.5) - 8.5

            if drop_prob is not None:
                alpha = logits.exp() / (self.w_global[8] ** 2 + 1e-10)
                probs = alpha / (1 + alpha)
                masks = (1 - (probs > drop_prob) * 1).detach()
            else:
                masks = torch.ones_like(self.w_global[8])
        else:
            masks = torch.ones_like(self.w_global[8])

        out.append(self.w_global[8] * masks)
        out.append(self.w_global[9])

        # if ood == False:
        #     # emd = self.embeds(torch.tensor(idx, dtype=torch.long))
        #     # emd = self.embeds[idx]
        #     # out.append(self.w_logits(emd).cuda().view(self.output_dim, self.output_dim2).clamp(-8, 8) -8)
        #     out.append(logits)
        # else:
        #     if feature_stat is None:
        #         out.append(self.w_logits_temp.cuda())  #TODO mean_fec
        #         # emd_all = self.embeds[0:100].mean(0)
        #         # out.append(self.w_logits(emd_all).clamp(-8,8).cuda().view(self.output_dim, self.output_dim2) - 8)
        #     else:
        #         out.append(logits)

        if ood == False:
            # emd = self.embeds(torch.tensor(idx, dtype=torch.long))
            # out.append(self.w_logits(emd).cuda().view(self.output_dim,self.output_dim2).clamp(-8,8) - 8)
            out.append(logits)
        else:
            # emd_all = self.embeds(torch.arange(0,100))  #TODO mean_fec
            # out.append(self.w_logits(emd_all.mean(0)).view(self.output_dim, self.output_dim2).clamp(-8,8).cuda() - 8)
            out.append(self.w_logits_temp.cuda())

        out.append(self.w_global[10])
        out.append(self.w_global[11])

        # if self.prior is True:
        #     if ood == False:
        #         emd_all = self.embeds[0:100].mean(0)
        #         out.append(self.w_logits(emd_all).clamp(-8,8).cuda().view(self.output_dim, self.output_dim2) - 8)
        #     else:
        #         emd_all = self.embeds[0:100].mean(0)
        #         out.append(self.w_logits(emd_all).clamp(-8,8).cuda().view(self.output_dim, self.output_dim2) - 8)
        #         # out.append(self.w_logits_temp.cuda())

        return out


def vector_to_list_parameters(vec: torch.Tensor, parameter_shapes: typing.List) -> torch.Tensor:
    """ """
    params = []

    # Pointer for slicing the vector for each parameter
    pointer = 0

    for param_shape in parameter_shapes:
        # The length of the parameter
        num_param = np.prod(a=param_shape)

        params.append(vec[pointer : pointer + num_param].view(param_shape))

        # Increment the pointer
        pointer += num_param

    return params


def intialize_parameters(state_dict: dict) -> typing.List[torch.Tensor]:
    """"""
    p = list(state_dict.values())
    for m in p:
        if m.ndim > 1:
            torch.nn.init.kaiming_normal_(tensor=m, nonlinearity="relu")
        else:
            torch.nn.init.zeros_(tensor=m)

    return p


# def torch_module_to_functional(torch_net: torch.nn.Module) -> higher.patch._MonkeyPatchBase:
#     """Convert a conventional torch module to its "functional" form
#     """
#     f_net = higher.patch.make_functional(module=torch_net)
#     f_net.track_higher_grads = False
#     f_net._fast_params = [[]]

#     return f_net


def intialize_parameters(state_dict: dict) -> typing.List[torch.Tensor]:
    """"""
    p = list(state_dict.values())
    for m in p:
        if m.ndim > 1:
            torch.nn.init.kaiming_normal_(tensor=m, nonlinearity="relu")
        else:
            torch.nn.init.zeros_(tensor=m)

    return p


@torch.no_grad()
def init_weights(m):
    if type(m) == torch.nn.Linear:
        if hasattr(m, "weight"):
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
            torch.nn.init.xavier_normal_(m.weight)
            # m.weight.fill_(0.01)
        if hasattr(m, "bias") and m.bias is not None:
            # m.bias.fill_(0.001)
            torch.nn.init.normal_(m.bias, mean=0, std=1e-3)
            # m.bias.fill_(0.01)

    if type(m) == torch.nn.Conv2d:
        if hasattr(m, "weight"):
            # gain = torch.nn.init.calculate_gain('relu')
            # nn.init.xavier_normal_(m.weight, gain=gain)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.xavier_normal_(m.weight)
            # m.weight.fill_(0.01)
        if hasattr(m, "bias") and m.bias is not None:
            #    m.bias.fill_(0.01)
            torch.nn.init.normal_(m.bias, mean=0, std=1e-3)


# class IdentityNet(torch.nn.Module):
#     """Identity hyper-net class for MAML"""
#     def __init__(self, base_weight: torch.nn.Module, **kwargs) -> None:
#         super(IdentityNet, self).__init__()

#         # base_state_dict = base_net.state_dict()
#         # params = intialize_parameters(state_dict=base_state_dict)
#         # self.params = torch.nn.ParameterList([torch.nn.Parameter(p.float()) for p in params])

#         # REPTILE possible (gradient not flow)
#     #    weights_after = [w.clone().detach() for w in weights]
#     #    weights_after = deepcopy(weights)
#     #    weights_after = [w.detach().clone().requires_grad_(True) for w in weights]
#     #    weights_after = [torch.empty_like(w).new_tensor(w.data, requires_grad=True) for w in weights]
#         base_weight_copy = [torch.Tensor(w).clone().detach().requires_grad_(True) for w in base_weight]

#         # MAML possible (gradient flow)
#     # #    weights_after = [w.clone() for w in weights]
#     #     base_weight = [torch.Tensor(w) for w in base_weight]

#         self.params = torch.nn.ParameterList([torch.nn.Parameter(p.float()) for p in base_weight_copy])
#         self.identity = torch.nn.Identity()

#     def forward(self) -> typing.List[torch.Tensor]:
#         out = []
#         for param in self.params:
#             temp = self.identity(param)
#             out.append(temp)
#         return out


# class MultiNet(torch.nn.Module):
#     """Identity hyper-net class for MAML"""
#     def __init__(self, base_weight: torch.nn.Module, **kwargs) -> None:
#         super(MultiNet, self).__init__()

#         # base_state_dict = base_net.state_dict()
#         # params = intialize_parameters(state_dict=base_state_dict)
#         # self.params = torch.nn.ParameterList([torch.nn.Parameter(p.float()) for p in params])
#         self.params = torch.nn.ParameterList([torch.nn.Parameter(p.float()) for p in base_weight])
#         self.identity = torch.nn.Identity()

#     def forward(self) -> typing.List[torch.Tensor]:
#         out = []
#         for param in self.params:
#             temp = self.identity(param)
#             out.append(temp)
#         return out

# class NormalVariationalNet(torch.nn.Module):
#     """A simple neural network that simulate the
#     reparameterization trick. Its parameters are
#     the mean and std-vector
#     """
#     def __init__(self, base_net: torch.nn.Module, **kwargs) -> None:
#         """
#         Args:
#             base_net: the base network
#         """
#         super(NormalVariationalNet, self).__init__()

#         # dict of parameters of based network
#         base_state_dict = base_net.state_dict()

#         mean = intialize_parameters(state_dict=base_state_dict)

#         # initialize parameters
#         self.mean = torch.nn.ParameterList([torch.nn.Parameter(m) \
#             for m in mean])
#         self.log_std = torch.nn.ParameterList([torch.nn.Parameter(torch.rand_like(v) - 4) \
#             for v in base_state_dict.values()])

#         self.num_base_params = np.sum([torch.numel(p) for p in self.mean])

#     def forward(self) -> typing.List[torch.Tensor]:
#         """Output the parameters of the base network in list format to pass into higher monkeypatch
#         """
#         out = []
#         for m, log_s in zip(self.mean, self.log_std):
#             eps_normal = torch.randn_like(m, device=m.device)
#             temp = m + eps_normal * torch.exp(input=log_s)
#             out.append(temp)
#         return out

# class GaussianDropoutNet(torch.nn.Module):
#     """A simple neural network that simulate the
#     reparameterization trick. Its parameters are
#     the mean and std-vector
#     """
#     def __init__(self, base_net: torch.nn.Module, **kwargs) -> None:
#         """
#         Args:
#             base_net: the base network
#         """
#         super(GaussianDropoutNet, self).__init__()

#         # dict of parameters of based network
#         base_state_dict = base_net.state_dict()

#         mean = intialize_parameters(state_dict=base_state_dict)

#         # initialize parameters
#         self.mean = torch.nn.ParameterList([torch.nn.Parameter(m) \
#             for m in mean])
#         # self.log_std = torch.nn.ParameterList([torch.nn.Parameter(torch.rand_like(v) - 4) \
#         #     for v in base_state_dict.values()])

#         self.log_alpha = torch.nn.ParameterList([torch.nn.Parameter(torch.randn_like(v) - 4) \
#             for v in base_state_dict.values()])

#         self.num_base_params = np.sum([torch.numel(p) for p in self.mean])

#     def forward(self) -> typing.List[torch.Tensor]:
#         """Output the parameters of the base network in list format to pass into higher monkeypatch
#         """

#         out = []
#         # for m, log_s in zip(self.mean, self.log_std):
#         for m, log_a in zip(self.mean, self.log_alpha):
#             eps_normal = torch.randn_like(m, device=m.device)
#             # temp = m + eps_normal * torch.exp(input=log_s)

#             temp = m + eps_normal * torch.abs(m + 1e-10) * torch.sqrt(log_a.clamp(min=-9.,max=9.).exp() + 1e-10)

#             out.append(temp)
#         return out


# class BernoulliDropoutNet(torch.nn.Module):
#     """A simple neural network that simulate the
#     reparameterization trick. Its parameters are
#     the mean and std-vector
#     """
#     def __init__(self, base_weight: torch.nn.Module, num_users, **kwargs) -> None:
#         """
#         Args:
#             base_net: the base network
#         """
#         super(BernoulliDropoutNet, self).__init__()

#         # base_weight_copy = [torch.Tensor(w).clone().detach().requires_grad_(True) for w in base_weight]

#         # MAML possible (gradient flow)
#     # #    weights_after = [w.clone() for w in weights]
#         base_weight_copy = [torch.Tensor(w) for w in base_weight]

#         self.mean = torch.nn.ParameterList([torch.nn.Parameter(p.float()) for p in base_weight_copy])

#         #self.identity = torch.nn.Identity()
#         # self.logit = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros_like(p)) for p in base_weight_copy])

#         temp_logit = torch.nn.ParameterList([torch.nn.Parameter(torch.ones_like(p)*-9) for p in base_weight_copy])
#         self.logits = []
#         for i in range(num_users):
#             logit = copy.deepcopy(temp_logit.state_dict()).values()
#             self.logits.append(torch.nn.ParameterList(logit))
#         self.logits = torch.nn.ParameterList(self.logits)

#         # self.logit = nn.Parameter(torch.Tensor(in_features).fill_(np.log(0.5/(1.-0.5))))
#         # self.sigmoid = nn.Sigmoid()


#         # dict of parameters of based network
#         # base_state_dict = base_net.state_dict()

#         # mean = intialize_parameters(state_dict=base_state_dict)

#         # initialize parameters
#         # self.mean = torch.nn.ParameterList([torch.nn.Parameter(m) \
#         #     for m in mean])
#         # self.log_std = torch.nn.ParameterList([torch.nn.Parameter(torch.rand_like(v) - 4) for v in base_state_dict.values()])

#         # self.log_alpha = torch.nn.ParameterList([torch.nn.Parameter(torch.randn_like(v) - 4) for v in base_state_dict.values()])

#         # self.num_base_params = np.sum([torch.numel(p) for p in self.mean])

#     def forward(self, idx) -> typing.List[torch.Tensor]:
#         """Output the parameters of the base network in list format to pass into higher monkeypatch
#         """

#         out = []
#         # for m, log_s in zip(self.mean, self.log_std):
#         for m, logit in zip(self.mean, self.logits[idx]):

#             probs = torch.sigmoid(logit.clamp(min=-9., max=9.))

#             eps = torch.randn_like(m, device=m.device)

#             # temp = m + eps_normal * torch.exp(input=log_s)
#             # temp = m + eps_normal * torch.abs(m + 1e-10) * torch.sqrt(log_a.clamp(min=-9.,max=9.).exp() + 1e-10)
#             # temp = m + eps * torch.abs(m + 1e-10) * torch.sqrt(log_a.clamp(min=-9.,max=9.).exp() + 1e-10)

#             si = torch.abs(m+1e-10) * torch.sqrt(probs*(1-probs) +1e-10)
#             temp = (1-probs)*m + eps*si

#             out.append(temp)

#         return out

#         # if self.training:
#         #     if self.fc == True:
#         #         # si = input * torch.sqrt(log_alpha.exp() + self.eps).to(input.device)
#         #         si = input.abs() * torch.sqrt(probs*(1-probs) + self.eps).to(input.device)
#         #     else:
#         #         # si = input * torch.sqrt(log_alpha.exp().unsqueeze(-1).unsqueeze(-1) + self.eps).to(input.device)
#         #         probs = probs.unsqueeze(-1).unsqueeze(-1)
#         #         si = input.abs() * torch.sqrt((probs*(1-probs)) + self.eps).to(input.device)

#         #     eps = torch.randn(*input.size()).to(input.device)
#         #     assert si.shape == eps.shape

#         #     return (1-probs)*input + eps*si
#         #     # return input + eps*si

#         # else:
#         #     # if self.fc == False:
#         #     #     probs = probs.unsqueeze(-1).unsqueeze(-1)
#         #     return (1-probs)*input
#         #     # return input


# class EnsembleNet(torch.nn.Module):
#     """A hyper-net class for BMAML that stores a set of parameters (known as particles)
#     """
#     def __init__(self, base_net: torch.nn.Module, **kwargs) -> None:
#         """Initiate an instance of EnsembleNet

#         Args:
#             base_net: the network of interest
#             num_particles: number of models
#         """
#         super().__init__()

#         self.num_particles = kwargs["num_models"]

#         if (self.num_particles <= 1):
#             raise ValueError("Minimum number of particles is 2.")

#         # dict of parameters of based network
#         base_state_dict = base_net.state_dict()

#         # add shapes of parameters into self
#         self.parameter_shapes = []
#         for param in base_state_dict.values():
#             self.parameter_shapes.append(param.shape)

#         self.params = torch.nn.ParameterList(parameters=None) # empty parameter list

#         for _ in range(self.num_particles):
#             params_list = intialize_parameters(state_dict=base_state_dict) # list of tensors
#             params_vec = torch.nn.utils.parameters_to_vector(parameters=params_list) # flattened tensor
#             self.params.append(parameter=torch.nn.Parameter(data=params_vec))

#         self.num_base_params = np.sum([torch.numel(p) for p in self.params[0]])

#     def forward(self, i: int) -> typing.List[torch.Tensor]:
#         return vector_to_list_parameters(vec=self.params[i], parameter_shapes=self.parameter_shapes)

# class PlatipusNet(torch.nn.Module):
#     """A class to hold meta-parameters used in PLATIPUS algorithm

#     Meta-parameters include:
#         - mu_theta
#         - log_sigma_theta
#         - log_v_q - note that, here v_q is the "std", not the covariance as in the paper.
#         One can simply square it and get the one in the paper.
#         - learning rate: gamma_p
#         - learning rate: gamma_q

#     Note: since the package "higher" is designed to handle ParameterList,
#     the implementation requires to keep the order of such parameters mentioned above.
#     This is annoying, but hopefully, the authors of "higher" could extend to handle
#     ParameterDict.
#     """

#     def __init__(self, base_net: torch.nn.Module, **kwargs) -> None:
#         super().__init__()

#         # dict of parameters of based network
#         base_state_dict = base_net.state_dict()

#         # add shapes of parameters into self
#         self.parameter_shapes = []
#         self.num_base_params = 0
#         for param in base_state_dict.values():
#             self.parameter_shapes.append(param.shape)
#             self.num_base_params += np.prod(param.shape)

#         # initialize ParameterList
#         self.params = torch.nn.ParameterList(parameters=None)

#         # add parameters into ParameterList
#         self.params.append(parameter=torch.nn.Parameter(data=torch.randn(size=(self.num_base_params,))))
#         self.params.append(parameter=torch.nn.Parameter(data=torch.randn(size=(self.num_base_params,)) - 4))
#         self.params.append(parameter=torch.nn.Parameter(data=torch.randn(size=(self.num_base_params,)) - 4))
#         # for _ in ("mu_theta", "log_sigma_theta", "log_v_q"):
#         #     params_list = intialize_parameters(state_dict=base_state_dict)
#         #     params_vec = torch.nn.utils.parameters_to_vector(parameters=params_list) - 4 # flattened tensor
#         #     self.params.append(parameter=torch.nn.Parameter(data=params_vec))

#         self.params.append(parameter=torch.nn.Parameter(data=torch.tensor(0.01))) # gamma_p
#         self.params.append(parameter=torch.nn.Parameter(data=torch.tensor(0.01))) # gamma_q

#     def forward(self) -> dict:
#         """Generate a dictionary consisting of meta-paramters
#         """
#         meta_params = dict.fromkeys(("mu_theta", "log_sigma_theta", "log_v_q", "gamma_p", "gamma_q"))

#         meta_params["mu_theta"] = vector_to_list_parameters(vec=self.params[0], parameter_shapes=self.parameter_shapes)
#         meta_params["log_sigma_theta"] = vector_to_list_parameters(vec=self.params[1], parameter_shapes=self.parameter_shapes)
#         meta_params["log_v_q"] = vector_to_list_parameters(vec=self.params[2], parameter_shapes=self.parameter_shapes)
#         meta_params["gamma_p"] = self.params[3]
#         meta_params["gamma_q"] = self.params[4]

#         return meta_params
