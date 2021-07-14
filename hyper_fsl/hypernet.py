from copy import deepcopy
from typing import Dict

import torch
from torch import nn, Tensor


def get_param_dict(net: nn.Module) -> Dict[str, nn.Parameter]:
    return {
        n: p
        for (n, p) in net.named_parameters()
    }


def set_from_param_dict(net: nn.Module, param_dict: Dict[str, Tensor]):
    for (sdk, v) in param_dict.items():
        keys = sdk.split(".")
        param_name = keys[-1]
        m = net
        for k in keys[:-1]:
            try:
                k = int(k)
                m = m[k]
            except:
                m = getattr(m, k)

        param = getattr(m, param_name)
        assert param.shape == v.shape, (sdk, param.shape, v.shape)
        delattr(m, param_name)
        setattr(m, param_name, v)



class HyperNetwork(nn.Module):
    def __init__(
            self,
            target_network: nn.Module,
            n: int, k: int,
            fe_out_size: int = 1024,
    ):
        super().__init__()
        self.n = n
        self.k = k
        self.fe = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.Flatten(),
            nn.Linear(7*7*64, fe_out_size),
            nn.ReLU(),
            nn.Linear(fe_out_size, fe_out_size)
        )
        param_head_size = fe_out_size * n * k

        param_dict = get_param_dict(target_network)
        self.param_shapes = {
            name: p.shape
            for (name, p)
            in param_dict.items()
        }

        self.param_nets = nn.ModuleDict()

        for name, param in param_dict.items():
            #print(name, param.shape, param_head_size*fe_out_size, fe_out_size*param.numel())

            self.param_nets[name.replace(".", "-")] = nn.Sequential(
                nn.Linear(param_head_size, fe_out_size),
                nn.ReLU(),
                nn.Linear(fe_out_size, param.numel())
            )

        self.target_network = target_network

    def forward(self, support_set: torch.Tensor):
        assert support_set.shape[0] == self.n * self.k
        
        # embedding of each example in support set
        emb = self.fe(support_set)

        # concat embeddings
        emb = emb.reshape(-1).unsqueeze(0)
        
        # predict network params
        network_params = {
            name.replace("-", "."): param_net(emb).reshape(self.param_shapes[name])
            for name, param_net in self.param_nets.items()
        }

        tn = deepcopy(self.target_network)
        set_from_param_dict(tn, network_params)
        return tn






