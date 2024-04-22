import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FCmodel(nn.Module):
    def __init__(self, input_size, output_size, hdim=100):
        super().__init__()

        self.fc = torch.nn.Sequential(nn.Linear(input_size, hdim), nn.ReLU(), nn.Linear(hdim, hdim), nn.ReLU(), nn.Linear(hdim, output_size))

    def forward(self, img):
        x = img.view(img.size(0), -1)
        y = self.fc(x)

        return y


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        # nn.BatchNorm2d(out_channels, track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNNCifar(nn.Module):
    def __init__(self, output_size):
        super(CNNCifar, self).__init__()
        in_channels = 3
        num_classes = output_size

        hidden_size = 64
        # hidden_size = 256 # ! increase model size (multi_6)

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            # conv3x3(hidden_size, hidden_size)
        )

        self.linear0 = nn.Linear(hidden_size * 2 * 2 * 4, hidden_size * 2 * 2)  # added
        self.linear1 = nn.Linear(hidden_size * 2 * 2, hidden_size * 2)  # added
        self.linear2 = nn.Linear(hidden_size * 2, num_classes)

        # self.linear = nn.Linear(hidden_size*2*2, num_classes)

        self.apply(init_weights)

    def forward(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))

        features = torch.nn.functional.leaky_relu(self.linear0(features), negative_slope=0.1)  # added
        features = torch.nn.functional.leaky_relu(self.linear1(features), negative_slope=0.1)  # added
        logits = self.linear2(features)

        return logits

    def extract_features(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))

        return features


class CNNCifarVD(nn.Module):
    def __init__(self, output_size, gauss, rep=True, gamma=0):
        super().__init__()

        in_channels = 3
        num_classes = output_size

        hidden_size = 64

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            # conv3x3(hidden_size, hidden_size)
        )

        self.linear0 = nn.Linear(hidden_size * 2 * 2 * 4, hidden_size * 2 * 2)  # added

        self.gauss = gauss
        if rep == True:
            self.linear1 = LinearNVDPGDRep(hidden_size * 2 * 2, hidden_size * 2, gamma=gamma)  # Reparameterization Trick
        else:
            if self.gauss == False:
                self.linear1 = LinearNVDP(hidden_size * 2 * 2, hidden_size * 2)
            else:
                self.linear1 = LinearNVDPGD(hidden_size * 2 * 2, hidden_size * 2)

        self.kl_div = self.linear1.kl_div

        self.linear2 = nn.Linear(hidden_size * 2, num_classes)

        self.apply(init_weights)

    def forward(self, x):
        if self.training == False:
            sample_size = 1
        else:
            sample_size = 3

        features = self.features(x)
        features = features.view((features.size(0), -1))

        logits_pool = []
        for _ in range(sample_size):
            features0 = torch.nn.functional.leaky_relu(self.linear0(features), negative_slope=0.1)
            features1 = torch.nn.functional.leaky_relu(self.linear1(features0), negative_slope=0.1)
            logits = self.linear2(features1)
            logits_pool.append(logits)

        return torch.stack(logits_pool, 0).mean(0)


class CNNCifarNVDP(nn.Module):
    def __init__(self, output_size, gauss, rep=True, gamma=0):
        super().__init__()
        in_channels = 3
        num_classes = output_size

        hidden_size = 64
        hidden_size = 256  # ! increase model size (multi_6)

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            # conv3x3(hidden_size, hidden_size)
        )

        self.linear0 = nn.Linear(hidden_size * 2 * 2 * 4, hidden_size * 2 * 2)  # added

        self.gauss = gauss
        if rep == True:
            self.linear1 = LinearNVDPGDRep(hidden_size * 2 * 2, hidden_size * 2, gamma=gamma)  # Reparameterization Trick
        else:
            if self.gauss == False:
                self.linear1 = LinearNVDP(hidden_size * 2 * 2, hidden_size * 2)
            else:
                self.linear1 = LinearNVDPGD(hidden_size * 2 * 2, hidden_size * 2)

        self.kl_div = self.linear1.kl_div

        self.linear2 = nn.Linear(hidden_size * 2, num_classes)

        self.apply(init_weights)

    def forward(self, x):
        if self.training == False:
            sample_size = 1
        else:
            sample_size = 3

        features = self.features(x)
        features = features.view((features.size(0), -1))
        # logits = self.linear(self.linear_vd(features))
        # return logits

        logits_pool = []
        for _ in range(sample_size):
            features0 = torch.nn.functional.leaky_relu(self.linear0(features), negative_slope=0.1)
            features1 = torch.nn.functional.leaky_relu(self.linear1(features0), negative_slope=0.1)
            logits = self.linear2(features1)
            logits_pool.append(logits)

        return torch.stack(logits_pool, 0).mean(0)


# def kl_div(model):
#     kl = 0
#     numl = 0
#     for module in model.modules():
#         if isinstance(module, LinearNVDPGDRep):
#             kl += module.kl_div()
#             numl += 1
#     return kl / numl


class CNNCifarNVDPplus(nn.Module):
    def __init__(self, output_size, gauss, rep=True, gamma=0):
        super().__init__()
        in_channels = 3
        num_classes = output_size

        hidden_size = 64

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            # conv3x3(hidden_size, hidden_size)
        )

        # self.linear0 = nn.Linear(hidden_size*2*2*4, hidden_size*2*2) #added
        # self.gauss = gauss
        # if rep == True:
        #     self.linear1 = LinearNVDPGDRep(hidden_size*2*2, hidden_size*2, gamma=gamma) # Reparameterization Trick
        # else:
        #     if self.gauss == False:
        #         self.linear1 = LinearNVDP(hidden_size*2*2, hidden_size*2)
        #     else:
        #         self.linear1 = LinearNVDPGD(hidden_size*2*2, hidden_size*2)

        self.linear0 = LinearNVDPGDRep(hidden_size * 2 * 2 * 4, hidden_size * 2 * 2, gamma=gamma)  # added
        self.linear1 = LinearNVDPGDRep(hidden_size * 2 * 2, hidden_size * 2, gamma=gamma)
        self.linear2 = LinearNVDPGDRep(hidden_size * 2, num_classes, gamma=gamma)
        self.apply(init_weights)
        self.kl_div = self.linear1.kl_div

    def forward(self, x):
        if self.training == False:
            sample_size = 1
        else:
            sample_size = 3

        features = self.features(x)
        features = features.view((features.size(0), -1))
        # logits = self.linear(self.linear_vd(features))
        # return logits

        logits_pool = []
        for _ in range(sample_size):
            features0 = torch.nn.functional.leaky_relu(self.linear0(features), negative_slope=0.1)
            features1 = torch.nn.functional.leaky_relu(self.linear1(features0), negative_slope=0.1)
            logits = self.linear2(features1)
            logits_pool.append(logits)

        return torch.stack(logits_pool, 0).mean(0)


class CNNCifarNVDPV2(nn.Module):
    def __init__(self, output_size, gauss, rep=True, gamma=0):
        super().__init__()
        in_channels = 3
        num_classes = output_size

        hidden_size = 64

        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            # conv3x3(hidden_size, hidden_size)
        )

        self.linear0 = nn.Linear(hidden_size * 2 * 2 * 4, hidden_size * 2 * 2)  # added

        self.gauss = gauss
        if rep == True:
            self.linear1 = LinearNVDPGDRep(hidden_size * 2 * 2, hidden_size * 2, gamma=gamma)  # Reparameterization Trick
        else:
            if self.gauss == False:
                self.linear1 = LinearNVDP(hidden_size * 2 * 2, hidden_size * 2)
            else:
                self.linear1 = LinearNVDPGD(hidden_size * 2 * 2, hidden_size * 2)

        self.kl_div = self.linear1.kl_div

        self.linear2 = nn.Linear(hidden_size * 2, num_classes)

        self.apply(init_weights)

    def forward(self, x, fec=False):
        if self.training == False:
            sample_size = 1
        else:
            sample_size = 3

        features = self.features(x)
        features = features.view((features.size(0), -1))

        if fec is False:
            logits_pool = []
            for _ in range(sample_size):
                # logits = self.linear(self.linear_vd(features))
                features0 = torch.nn.functional.leaky_relu(self.linear0(features), negative_slope=0.1)
                features1 = torch.nn.functional.leaky_relu(self.linear1(features0), negative_slope=0.1)
                logits = self.linear2(features1)
                logits_pool.append(logits)
            return torch.stack(logits_pool, 0).mean(0)
        else:
            return features.mean(0)


class LinearNVDP(nn.Linear):
    def __init__(self, in_features, out_features, gamma=0, bias=True):
        super().__init__(in_features, out_features, bias)

        self.eps = 1e-10
        self.in_features = in_features
        self.out_features = out_features
        # self.probs = nn.Parameter(torch.ones(*self.weight.size())*0.5, requires_grad=True)
        # self.probs_bias = nn.Parameter(torch.ones(*self.bias.size())*0.5, requires_grad=True)
        # self.tau = torch.nn.Parameter(torch.ones(1)*1.5, requires_grad=True)
        # self.probs = nn.Parameter(torch.ones(1,in_features)*0.5, requires_grad=True)
        # self.probs_bias = nn.Parameter(torch.ones(out_features)*0.5, requires_grad=True)
        # self.gamma = torch.nn.Parameter(torch.ones(1)*5, requires_grad=True) # 0.1 ~ 10

        # self.probs_alpha = nn.Parameter(torch.ones(1,in_features).fill_(np.log(0.1/(1.0-0.1))), requires_grad=True)
        # self.probs_tau = nn.Parameter(torch.ones(1)*1.5, requires_grad=True)
        # self.gamma = gamma
        # self.sigmoid = nn.Sigmoid()

        # self.logit = nn.Parameter(torch.ones(1,in_features).fill_(np.log(0.1/(1.0-0.1))), requires_grad=True)

        # self.logit = nn.Parameter(torch.ones(1,in_features).fill_(np.log(0.1/(1.0-0.1))), requires_grad=True)

        self.logit = nn.Parameter(torch.zeros(*self.weight.size()))
        # self.logit_bias = nn.Parameter(torch.zeros(*self.bias.size()), requires_grads=True)

        # self.tau = nn.Parameter(torch.ones(1)*1.5)

        # self.probs_tau = nn.Parameter(torch.ones(1)*1.5, requires_grad=True)
        # self.gamma = gamma
        # self.sigmoid = nn.Sigmoid()

    # def kl(self):
    #     # probs = torch.cat([self.probs.view(1,-1), self.probs_bias.view(1,-1)],-1)
    #     probs = self.probs.view(1,-1)
    #     kld = -0.5 * torch.log(probs+self.eps) * (probs<0.999)
    #     return kld.mean()

    # def kl_focal(self):
    # def kl(self):
    #     # probs = torch.cat([self.probs.view(1,-1), self.probs_bias.view(1,-1)],-1)
    #     # probs = self.probs.view(1,-1)
    #     # probs = self.sigmoid(self.probs_alpha.clamp(-9,9).div(self.probs_tau.clamp(0.5,10))).view(1,-1)

    #     # probs = self.sigmoid(self.probs_alpha.clamp(-9,9).div(self.probs_tau.clamp(0.5,10))).view(1,-1)
    #     probs = self.gumbel(self.sigmoid(self.probs_alpha.clamp(-9,9)))
    #     kld = -0.5 * torch.log(probs+self.eps) * (probs<0.999) #* torch.pow((1-probs), self.gamma)#.clamp(0,10)
    #     return kld.mean()

    # def gumbel(self, probs):
    #     u_noise = torch.rand_like(probs).cuda()
    #     drop_probs = (torch.log(probs + self.eps) -
    #                 torch.log(1 - probs + self.eps) +
    #                 torch.log(u_noise + self.eps) -
    #                 torch.log(1 - u_noise + self.eps))
    #     gprobs = torch.sigmoid(drop_probs / self.tau.clamp(0.5,5))
    #     return gprobs

    def kl_div(self, logits):
        probs = torch.sigmoid(logits.clamp(-8.5, 8.5).view(1, -1))
        kld = -0.5 * torch.log(probs + 1e-10) * (probs < 0.999)
        return kld.mean()
        # return kld.sum()
        # return -torch.sum(-0.5*torch.nn.functional.softplus(-logits.clamp(min=-9.,max=9.)))

    def forward(self, input):
        # self.probs = self.gumbel(self.probs)
        # self.probs_bias = self.gumbel(self.probs_bias)
        # probs = self.probs
        # probs_bias = self.probs_bias
        # probs = self.sigmoid(self.probs_alpha.clamp(-9,9).div(self.probs_tau.clamp(0.5,10)))

        # probs = self.sigmoid(self.probs_alpha.clamp(-9,9)) #.div(self.probs_tau.clamp(0.5,10)))
        # probs = self.gumbel(self.sigmoid(self.probs_alpha.clamp(-9,9)))

        probs = torch.sigmoid(self.logit.clamp(-8.5, 8.5))

        if self.training:
            # probs = self.gumbel(probs)

            w_mean = (1 - probs) * self.weight
            # b_mean = (1-probs_bias) * self.bias
            # wb_mean = (input).mm(w_mean.t()) + b_mean.unsqueeze(0)
            # wb_mean = F.linear(input, w_mean, b_mean)
            wb_mean = F.linear(input, w_mean, self.bias)
            w_var = probs * (1 - probs) * self.weight**2
            # b_var = probs_bias * (1-probs_bias) * self.bias**2
            # wb_std = torch.sqrt( (input**2).mm(w_var.t()) + b_var.unsqueeze(0) + self.eps)
            wb_std = torch.sqrt(F.linear(input**2, w_var, None) + self.eps)
            # eps = torch.randn(*input.size()).cuda()
            eps = torch.randn(*wb_mean.size(), device=input.device)

            return wb_mean + wb_std * eps  # + self.bias

        # if self.training: # Sampling Weight
        #     w_mean = (1-probs) * self.weight
        #     # w_std = torch.sqrt( probs * (1-probs) * self.weight**2 + self.eps)
        #     w_std = torch.sqrt( probs * (1-probs) + self.eps) * self.weight
        #     eps = torch.randn(*input.size()).to(input.device)
        #     # phi = w_mean + w_std * eps
        #     phi = w_mean.unsqueeze(0) + w_std.unsqueeze(0) * eps.unsqueeze(1)
        #     # wb_mean = F.linear(input, phi, self.bias)
        #     wb_mean = torch.bmm(input.unsqueeze(1), phi.permute(0,2,1)) + self.bias.unsqueeze(0)
        #     return wb_mean.squeeze()

        else:  # Output Expectation in test
            # mask = !(log_alpha > 3) then weight is zeroed
            # mask = (probs > 0.9).logical_not().float() # then mask is zerod
            w_mean = (1 - probs) * self.weight
            # mask = torch.nn.functional.threshold((1-probs),0.1, 0)
            # w_mean = mask * self.weight
            # b_mean = (1-probs_bias) * self.bias
            # b_mean = self.bias
            # wb_mean = (input).mm(w_mean.t()) + b_mean.unsqueeze(0)
            wb_mean = F.linear(input, w_mean, self.bias)

            return wb_mean


class LinearNVDPGD(nn.Linear):
    def __init__(self, in_features, out_features, gamma=0, bias=True):
        super().__init__(in_features, out_features, bias)

        self.eps = 1e-10
        self.in_features = in_features
        self.out_features = out_features
        # self.probs = nn.Parameter(torch.ones(*self.weight.size())*0.5, requires_grad=True)
        # self.probs_bias = nn.Parameter(torch.ones(*self.bias.size())*0.5, requires_grad=True)
        # self.tau = torch.nn.Parameter(torch.ones(1)*1.5, requires_grad=True)
        # self.probs = nn.Parameter(torch.ones(1,in_features)*0.5, requires_grad=True)
        # self.probs_bias = nn.Parameter(torch.ones(out_features)*0.5, requires_grad=True)
        # self.gamma = torch.nn.Parameter(torch.ones(1)*5, requires_grad=True) # 0.1 ~ 10

        # self.probs_alpha = nn.Parameter(torch.ones(1,in_features).fill_(np.log(0.1/(1.0-0.1))), requires_grad=True)
        # self.probs_tau = nn.Parameter(torch.ones(1)*1.5, requires_grad=True)
        # self.gamma = gamma
        # self.sigmoid = nn.Sigmoid()

        # self.logit = nn.Parameter(torch.ones(1,in_features).fill_(np.log(0.1/(1.0-0.1))), requires_grad=True)

        # self.logit = nn.Parameter(torch.ones(1,in_features).fill_(np.log(0.1/(1.0-0.1))), requires_grad=True)

        self.logit = nn.Parameter(torch.zeros(*self.weight.size()) - 8.5)
        # self.logit_bias = nn.Parameter(torch.zeros(*self.bias.size()), requires_grads=True)

        # self.tau = nn.Parameter(torch.ones(1)*1.5)

        # self.probs_tau = nn.Parameter(torch.ones(1)*1.5, requires_grad=True)
        # self.gamma = gamma
        # self.sigmoid = nn.Sigmoid()

    # def kl(self):
    #     # probs = torch.cat([self.probs.view(1,-1), self.probs_bias.view(1,-1)],-1)
    #     probs = self.probs.view(1,-1)
    #     kld = -0.5 * torch.log(probs+self.eps) * (probs<0.999)
    #     return kld.mean()

    # def kl_focal(self):
    # def kl(self):
    #     # probs = torch.cat([self.probs.view(1,-1), self.probs_bias.view(1,-1)],-1)
    #     # probs = self.probs.view(1,-1)
    #     # probs = self.sigmoid(self.probs_alpha.clamp(-9,9).div(self.probs_tau.clamp(0.5,10))).view(1,-1)

    #     # probs = self.sigmoid(self.probs_alpha.clamp(-9,9).div(self.probs_tau.clamp(0.5,10))).view(1,-1)
    #     probs = self.gumbel(self.sigmoid(self.probs_alpha.clamp(-9,9)))
    #     kld = -0.5 * torch.log(probs+self.eps) * (probs<0.999) #* torch.pow((1-probs), self.gamma)#.clamp(0,10)
    #     return kld.mean()

    # def gumbel(self, probs):
    #     u_noise = torch.rand_like(probs).cuda()
    #     drop_probs = (torch.log(probs + self.eps) -
    #                 torch.log(1 - probs + self.eps) +
    #                 torch.log(u_noise + self.eps) -
    #                 torch.log(1 - u_noise + self.eps))
    #     gprobs = torch.sigmoid(drop_probs / self.tau.clamp(0.5,5))
    #     return gprobs

    def kl_div(self, logits):
        # probs = torch.sigmoid(logits.view(1,-1))
        # kld = -0.5 * torch.log(probs+1e-10) * (probs<0.999)
        # # return kld.mean()
        # return kld.sum()
        # return -torch.sum(-0.5*torch.nn.functional.softplus(-logits.clamp(min=-9.,max=9.)))

        # return torch.sum(0.5*torch.nn.functional.softplus(-logits.clamp(min=-9., max=9.))) # maybe clamp is important?
        return torch.mean(0.5 * torch.nn.functional.softplus(-logits.clamp(min=-8.5, max=8.5)))  # maybe clamp is important?

    def forward(self, input):
        # self.probs = self.gumbel(self.probs)
        # self.probs_bias = self.gumbel(self.probs_bias)
        # probs = self.probs
        # probs_bias = self.probs_bias
        # probs = self.sigmoid(self.probs_alpha.clamp(-9,9).div(self.probs_tau.clamp(0.5,10)))

        # probs = self.sigmoid(self.probs_alpha.clamp(-9,9)) #.div(self.probs_tau.clamp(0.5,10)))
        # probs = self.gumbel(self.sigmoid(self.probs_alpha.clamp(-9,9)))

        # probs = torch.sigmoid(self.logit.clamp(-9,9))

        if self.training:
            # probs = self.gumbel(probs)

            # w_mean = self.weight
            # b_mean = (1-probs_bias) * self.bias
            # wb_mean = (input).mm(w_mean.t()) + b_mean.unsqueeze(0)
            # wb_mean = F.linear(input, w_mean, b_mean)
            # wb_mean = F.linear(input, w_mean, self.bias)
            wb_mean = F.linear(input, self.weight, self.bias)

            # w_var =  probs * (1-probs) * self.weight**2
            w_var = self.logit.clamp(min=-8.5, max=8.5).exp() * self.weight**2
            # b_var = probs_bias * (1-probs_bias) * self.bias**2
            # wb_std = torch.sqrt( (input**2).mm(w_var.t()) + b_var.unsqueeze(0) + self.eps)
            wb_std = torch.sqrt(F.linear(input**2, w_var, None) + self.eps)

            # eps = torch.randn(*input.size()).cuda()
            eps = torch.randn(*wb_mean.size(), device=input.device)

            return wb_mean + wb_std * eps  # + self.bias

        # if self.training: # Sampling Weight
        #     w_mean = (1-probs) * self.weight
        #     # w_std = torch.sqrt( probs * (1-probs) * self.weight**2 + self.eps)
        #     w_std = torch.sqrt( probs * (1-probs) + self.eps) * self.weight
        #     eps = torch.randn(*input.size()).to(input.device)
        #     # phi = w_mean + w_std * eps
        #     phi = w_mean.unsqueeze(0) + w_std.unsqueeze(0) * eps.unsqueeze(1)
        #     # wb_mean = F.linear(input, phi, self.bias)
        #     wb_mean = torch.bmm(input.unsqueeze(1), phi.permute(0,2,1)) + self.bias.unsqueeze(0)
        #     return wb_mean.squeeze()

        else:  # Output Expectation in test
            # mask = !(log_alpha > 3) then weight is zeroed
            # mask = (probs > 0.9).logical_not().float() # then mask is zerod
            # w_mean = (1-probs) * self.weight
            # mask = torch.nn.functional.threshold((1-probs),0.1, 0)
            # w_mean = mask * self.weight
            # b_mean = (1-probs_bias) * self.bias
            # b_mean = self.bias
            # wb_mean = (input).mm(w_mean.t()) + b_mean.unsqueeze(0)
            # wb_mean = F.linear(input, w_mean, self.bias)
            wb_mean = F.linear(input, self.weight, self.bias)

            return wb_mean


class LinearNVDPGDRep(nn.Linear):
    def __init__(self, in_features, out_features, gamma=0, bias=True):
        super().__init__(in_features, out_features, bias)

        self.eps = 1e-10
        self.in_features = in_features
        self.out_features = out_features

        self.logit = nn.Parameter(torch.zeros(*self.weight.size()) - 8.5)
        # self.logit_bias = nn.Parameter(torch.zeros(*self.bias.size()), requires_grads=True)
        self.gamma = gamma

    def kl_div(self, logit, wt):
        # probs = torch.sigmoid(logits.view(1,-1))
        # kld = -0.5 * torch.log(probs+1e-10) * (probs<0.999)
        # # return kld.mean()
        # return kld.sum()
        # alpha = logits.exp() / (wt**2 + 1e-10)
        # kld = 0.5 * torch.log(1 + (wt.detach()**2) / (torch.nn.functional.softplus(logits) + 1e-10))
        # kld = 0.5 * torch.log(1 + (wt.detach()**2) / (torch.nn.functional.softplus(logit.clamp(-20,40))+1e-10)) #TODO: Detach WT(?)

        ## focal-loss variant

        # alpha = (logit.clamp(-8,8).exp()) / (wt.detach()**2 +1e-13)
        # probs = alpha / (1+alpha)
        # kld = 0.5 * torch.log(1 + (wt.detach()**2) / (logit.clamp(-8,8).exp() + 1e-13)) * torch.pow((1-probs), self.gamma)  #TODO: Detach probs?

        pow_const = wt.detach() ** 2 / (wt.detach() ** 2 + logit.clamp(-8.5, 8.5).exp() + 1e-13)  # detach logit??
        kld = 0.5 * torch.log(1 + (wt.detach() ** 2) / (logit.clamp(-8.5, 8.5).exp() + 1e-13)) * torch.pow(pow_const.detach(), self.gamma)  # TODO: Detach probs?
        # kld = 0.5 * -torch.log(pow_const) * torch.pow(pow_const, self.gamma)  #TODO: Detach probs?
        # kld = 0.5 * -torch.log(1-pow_const) * torch.pow(pow_const.detach(), self.gamma)#TODO: Detach WT(?)

        # Original
        # kld = 0.5 * torch.log(1 + (wt.detach()**2) / (logit.clamp(-8,8).exp() + 1e-13)) #TODO: Detach WT(?)

        # return kld.sum()
        return kld.mean()

        # return -torch.sum(-0.5*torch.nn.functional.softplus(-logits.clamp(min=-9.,max=9.)))

    # def kl_div_prior(self, var_q_logits, var_p_logits, wt):
    #     var_q = var_q_logits.clamp(-8,8).exp() / wt.detach()**2
    #     var_p = var_p_logits.clamp(-8,8).exp() / wt.detach()**2

    #     mu_q = torch.zeros_like(var_q)
    #     mu_p = torch.zeros_like(var_p)

    #     kld = 0.5 * ((var_q +(mu_q-mu_p)**2) / (var_p + 1e-10) + (torch.log(var_p+1e-10) - torch.log(var_q + 1e-10)) - 1)
    #     return torch.mean(kld)

    def forward(self, input):
        # self.probs = self.gumbel(self.probs)
        # self.probs_bias = self.gumbel(self.probs_bias)
        # probs = self.probs
        # probs_bias = self.probs_bias
        # probs = self.sigmoid(self.probs_alpha.clamp(-9,9).div(self.probs_tau.clamp(0.5,10)))

        # probs = self.sigmoid(self.probs_alpha.clamp(-9,9)) #.div(self.probs_tau.clamp(0.5,10)))
        # probs = self.gumbel(self.sigmoid(self.probs_alpha.clamp(-9,9)))

        # probs = torch.sigmoid(self.logit.clamp(-9,9))

        if self.training:
            # probs = self.gumbel(probs)

            # w_mean = self.weight
            # b_mean = (1-probs_bias) * self.bias
            # wb_mean = (input).mm(w_mean.t()) + b_mean.unsqueeze(0)
            # wb_mean = F.linear(input, w_mean, b_mean)
            # wb_mean = F.linear(input, w_mean, self.bias)
            wb_mean = F.linear(input, self.weight, self.bias)

            # w_var =  probs * (1-probs) * self.weight**2
            w_var = self.logit.clamp(-8.5, 8.5).exp()  # * self.weight**2
            # w_var = torch.nn.functional.softplus(self.logit) #* self.weight**2
            # b_var = probs_bias * (1-probs_bias) * self.bias**2
            # wb_std = torch.sqrt( (input**2).mm(w_var.t()) + b_var.unsqueeze(0) + self.eps)
            wb_std = torch.sqrt(F.linear(input**2, w_var, None) + self.eps)

            # eps = torch.randn(*input.size()).cuda()
            eps = torch.randn(*wb_mean.size(), device=input.device)

            return wb_mean + wb_std * eps  # + self.bias

        # if self.training: # Sampling Weight
        #     w_mean = (1-probs) * self.weight
        #     # w_std = torch.sqrt( probs * (1-probs) * self.weight**2 + self.eps)
        #     w_std = torch.sqrt( probs * (1-probs) + self.eps) * self.weight
        #     eps = torch.randn(*input.size()).to(input.device)
        #     # phi = w_mean + w_std * eps
        #     phi = w_mean.unsqueeze(0) + w_std.unsqueeze(0) * eps.unsqueeze(1)
        #     # wb_mean = F.linear(input, phi, self.bias)
        #     wb_mean = torch.bmm(input.unsqueeze(1), phi.permute(0,2,1)) + self.bias.unsqueeze(0)
        #     return wb_mean.squeeze()

        else:  # Output Expectation in test
            # mask = !(log_alpha > 3) then weight is zeroed
            # mask = (probs > 0.9).logical_not().float() # then mask is zerod
            # w_mean = (1-probs) * self.weight
            # mask = torch.nn.functional.threshold((1-probs),0.1, 0)
            # w_mean = mask * self.weight
            # b_mean = (1-probs_bias) * self.bias
            # b_mean = self.bias
            # wb_mean = (input).mm(w_mean.t()) + b_mean.unsqueeze(0)
            # wb_mean = F.linear(input, w_mean, self.bias)
            wb_mean = F.linear(input, self.weight, self.bias)

            return wb_mean


@torch.no_grad()
def init_weights(m):
    if type(m) == nn.Linear or type(m) == LinearNVDP or type(m) == LinearNVDPGD:
        if hasattr(m, "weight"):
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            # nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.01))
            nn.init.xavier_normal_(m.weight)
            # m.weight.fill_(0.01)
        if hasattr(m, "bias") and m.bias is not None:
            # m.bias.fill_(0.001)
            nn.init.normal_(m.bias, mean=0, std=1e-3)
            # m.bias.fill_(0.01)

    if type(m) == nn.Conv2d:
        if hasattr(m, "weight"):
            # gain = torch.nn.init.calculate_gain('relu')
            # nn.init.xavier_normal_(m.weight, gain=gain)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.xavier_normal_(m.weight)
            # m.weight.fill_(0.01)
        if hasattr(m, "bias") and m.bias is not None:
            #    m.bias.fill_(0.01)
            nn.init.normal_(m.bias, mean=0, std=1e-3)


class NVDPlayer(nn.Module):
    def __init__(self, in_features, p=0.3, deterministic_test=False, deterministic_limit=True, deterministic_sparse=False, eps=1e-10, fc=True):
        super(NVDPlayer, self).__init__()
        self.deterministic_test = deterministic_test
        self.eps = eps
        self.deterministic_limit = deterministic_limit
        self.deterministic_sparse = deterministic_sparse
        self.fc = fc

        # log_alpha = torch.Tensor(in_features).fill_(np.log(p/(1. - p)))
        # self.log_alpha = nn.Parameter(log_alpha)

        # self.probs = nn.Parameter(torch.ones(in_features)*0.5, requires_grad=True)

        self.logit = nn.Parameter(torch.Tensor(in_features).fill_(np.log(p / (1.0 - p))))

        # self.softplus = nn.Softplus()

        # self.probs_alpha = nn.Parameter(torch.ones(1,in_features).fill_(np.log(0.1/(1.0-0.1))), requires_grad=True)
        # self.probs_tau = nn.Parameter(torch.ones(1)*1.5, requires_grad=True)

    # def gumbel(self, probs):
    #     u_noise = torch.rand_like(probs).cuda()
    #     drop_probs = (torch.log(probs + self.eps) -
    #                 torch.log(1 - probs + self.eps) +
    #                 torch.log(u_noise + self.eps) -
    #                 torch.log(1 - u_noise + self.eps))
    #     gprobs = torch.sigmoid(drop_probs / self.probs_tau.clamp(0.5,10))
    #     return gprobs

    # def kl(self):
    #     # if self.deterministic_limit == True:
    #     #     c1, c2, c3 = 1.16145124, -1.50204118, 0.58629921
    #     #     C = -(c1+c2+c3)
    #     #     if self.deterministic_limit == True:
    #     #         log_alpha = torch.clamp(self.log_alpha, -8., 0)
    #     #     else:
    #     #         log_alpha = self.log_alpha
    #     #     alpha = log_alpha.exp()
    #     #     return -torch.sum(0.5 * torch.log(alpha) + c1 * alpha + c2 * (alpha**2) + c3 * (alpha**3) + C)
    #     # else:
    #     #     if self.deterministic_sparse == True:
    #     #         k1, k2, k3 = 0.63576, 1.8732, 1.48695
    #     #         return -torch.sum(k1 * self.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * self.softplus(-self.log_alpha) - k1)

    #     #     else:

    #     probs = self.probs.view(1,-1)
    #     kld = -0.5 * torch.log(probs+self.eps) * (probs<0.999)
    #     # return kld.mean()
    #     return kld.sum()
    #     # return -torch.sum(-0.5*self.softplus(-self.log_alpha.clamp(min=-9.,max=9.)))

    # def updateAlpha(self, input):
    #     self.log_alpha.data = input

    def kl_div(self, logits):
        probs = torch.sigmoid(logits.view(1, -1))
        kld = -0.5 * torch.log(probs + 1e-10) * (probs < 0.999)
        return kld.mean()
        # return kld.sum()

    # @weak_script_method
    def forward(self, input):
        # if self.deterministic_test:
        #     assert self.training == False,"Flag deterministic is True. This should not be used in training."
        #     return input
        # else:
        #     if self.deterministic_limit == True:
        #         log_alpha = torch.clamp(self.log_alpha, -8., 0) ############Todo:好像截断了，导致梯度不能传递？
        #     else:

        # log_alpha = self.log_alpha.clamp(min=-9.,max=9.)
        # probs = self.probs

        probs = torch.sigmoid(self.logit.clamp(min=-9.0, max=9.0))

        if self.training:
            if self.fc == True:
                # si = input * torch.sqrt(log_alpha.exp() + self.eps).to(input.device)
                si = input.abs() * torch.sqrt(probs * (1 - probs)).to(input.device)
            else:
                # si = input * torch.sqrt(log_alpha.exp().unsqueeze(-1).unsqueeze(-1) + self.eps).to(input.device)
                probs = probs.unsqueeze(-1).unsqueeze(-1)
                si = input.abs() * torch.sqrt((probs * (1 - probs))).to(input.device)

            eps = torch.randn(*input.size()).to(input.device)
            assert si.shape == eps.shape

            return (1 - probs) * input + eps * si
            # return input + eps*si

        else:
            # if self.fc == False:
            #     probs = probs.unsqueeze(-1).unsqueeze(-1)
            return (1 - probs) * input
            # return input


class NVDPlayerGD(nn.Module):
    def __init__(self, in_features, p=0.3, deterministic_test=False, deterministic_limit=True, deterministic_sparse=False, eps=1e-10, fc=True):
        super(NVDPlayerGD, self).__init__()
        self.deterministic_test = deterministic_test
        self.eps = eps
        self.deterministic_limit = deterministic_limit
        self.deterministic_sparse = deterministic_sparse
        self.fc = fc

        log_alpha = torch.Tensor(in_features).fill_(np.log(p / (1.0 - p)))
        self.logit = nn.Parameter(log_alpha)

        # self.probs = nn.Parameter(torch.ones(in_features)*0.5, requires_grad=True)
        # self.logit = nn.Parameter(torch.Tensor(in_features).fill_(np.log(p/(1.-p))))

        # self.softplus = nn.Softplus()

        # self.probs_alpha = nn.Parameter(torch.ones(1,in_features).fill_(np.log(0.1/(1.0-0.1))), requires_grad=True)
        # self.probs_tau = nn.Parameter(torch.ones(1)*1.5, requires_grad=True)

    # def kl(self):
    #     # if self.deterministic_limit == True:
    #     #     c1, c2, c3 = 1.16145124, -1.50204118, 0.58629921
    #     #     C = -(c1+c2+c3)
    #     #     if self.deterministic_limit == True:
    #     #         log_alpha = torch.clamp(self.log_alpha, -8., 0)
    #     #     else:
    #     #         log_alpha = self.log_alpha
    #     #     alpha = log_alpha.exp()
    #     #     return -torch.sum(0.5 * torch.log(alpha) + c1 * alpha + c2 * (alpha**2) + c3 * (alpha**3) + C)
    #     # else:
    #     #     if self.deterministic_sparse == True:
    #     #         k1, k2, k3 = 0.63576, 1.8732, 1.48695
    #     #         return -torch.sum(k1 * self.sigmoid(k2 + k3 * self.log_alpha) - 0.5 * self.softplus(-self.log_alpha) - k1)
    #     #     else:
    #     probs = self.probs.view(1,-1)
    #     kld = -0.5 * torch.log(probs+self.eps) * (probs<0.999)
    #     # return kld.mean()
    #     return kld.sum()
    #     # return -torch.sum(-0.5*self.softplus(-self.log_alpha.clamp(min=-9.,max=9.)))

    # def updateAlpha(self, input):
    #     self.log_alpha.data = input

    def kl_div(self, logits):
        # return torch.sum(0.5*torch.nn.functional.softplus(-logits.clamp(min=-9.,max=9.).view(1,-1)))
        return torch.mean(0.5 * torch.nn.functional.softplus(-logits.clamp(min=-9.0, max=9.0).view(1, -1)))

    # @weak_script_method
    def forward(self, input):
        # if self.deterministic_test:
        #     assert self.training == False,"Flag deterministic is True. This should not be used in training."
        #     return input
        # else:
        #     if self.deterministic_limit == True:
        #         log_alpha = torch.clamp(self.log_alpha, -8., 0) ############Todo:好像截断了，导致梯度不能传递？
        #     else:

        log_alpha = self.logit.clamp(min=-9.0, max=9.0)
        # probs = self.probs
        # probs = torch.sigmoid(self.logit.clamp(min=-9., max=9.))

        if self.training:
            if self.fc == True:
                si = input.abs() * torch.sqrt(log_alpha.exp() + self.eps).to(input.device)
                # si = input.abs() * torch.sqrt(probs*(1-probs)).to(input.device)
            else:
                si = input.abs() * torch.sqrt(log_alpha.exp().unsqueeze(-1).unsqueeze(-1) + self.eps).to(input.device)
                # probs = probs.unsqueeze(-1).unsqueeze(-1)
                # si = input.abs() * torch.sqrt((probs*(1-probs))).to(input.device)

            eps = torch.randn(*input.size()).to(input.device)
            assert si.shape == eps.shape

            # return (1-probs)*input + eps*si
            return input + eps * si
        else:
            # if self.fc == False:
            # probs = probs.unsqueeze(-1).unsqueeze(-1)
            # return (1-probs)*input
            return input


# def kl_div(model):
#     kl = 0
#     # numl = 0
#     for module in model.children():
#         if hasattr(module, 'kl'):
#             kl += module.kl()
#             # numl += 1.0
#     # return kl / numl
#     return kl


# @torch.no_grad()
# def probConst(m):
#     #if type(m) == LinearNVDP: # or type(m) == Conv2dNVDP:
#     if type(m) == NVDPlayer: # or type(m) == Conv2dNVDP:
#         if hasattr(m, 'probs'):
#             m.probs.data = m.probs.data.clamp(0.001, 0.999)
#         # if hasattr(m, 'probs_bias'):
#         #     m.probs_bias.data = m.probs_bias.data.clamp(0.001, 1-0.001)

from torch.distributions import kl_divergence
from torch.distributions.normal import Normal


def kl_div_gaussians(mu_q, std_q, mu_p, std_p):
    q_target_dist = Normal(mu_q, std_q)
    p_context_dist = Normal(mu_p, std_p)
    kl_div = kl_divergence(q_target_dist, p_context_dist).sum(-1).mean()
    return kl_div


def kl_div_adhoc(mu_q, std_q, mu_p, std_p):
    var_q = std_q**2
    var_p = std_p**2
    kld = 0.5 * ((var_p + (mu_p - mu_q) ** 2) / (var_q + 1e-10) + (torch.log(var_p + 1e-10) - torch.log(var_q + 1e-10)) - 1)
    return torch.mean(kld)
