import torch
from copy import deepcopy


class AveragedModel(torch.nn.Module):
    def __init__(self, model, device=None, avg_fn=None):
        super(AveragedModel, self).__init__()
        self.module = deepcopy(model)
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer("n_averaged", torch.tensor(0, dtype=torch.long, device=device))
        if avg_fn is None:

            def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
                return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)

        self.avg_fn = avg_fn

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def update_parameters(self, w):
        # self.module.parameters() == self.parameters()
        self_param = self.parameters()
        model_param = w
        for p_swa, p_model in zip(self_param, model_param):
            device = p_swa.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa.detach().copy_(p_model_)
            else:
                p_swa.detach().copy_(self.avg_fn(p_swa.detach(), p_model_, self.n_averaged.to(device)))
        self.n_averaged += 1
