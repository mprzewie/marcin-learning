from copy import deepcopy
from typing import Dict, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as nnf

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


def sorted_by_label(X: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(X) == len(Y)

    xy = sorted(zip(X, Y), key=lambda x_y: x_y[1])
    X = torch.stack([x for (x,y) in xy])
    Y = torch.stack([y for (x,y) in xy])

    return X, Y


class HyperNetwork(nn.Module):
    def __init__(
            self,
            # target_network: nn.Module,
            n: int, k: int,
            hidden_size: int = 1024,
    ):
        super().__init__()
        self.n = n
        self.k = k
        self.fe = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, hidden_size),
            nn.ReLU(),

        )

        target_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.n)
        )

        param_head_size = (
                                  hidden_size #+ n
                           ) * n * k

        param_dict = get_param_dict(target_network)

        param_dict = {
            name.replace(".", "-"): p
            for name, p in param_dict.items()
        }

        self.param_shapes = {
            name: p.shape
            for (name, p)
            in param_dict.items()
        }

        self.param_nets = nn.ModuleDict()

        for name, param in param_dict.items():
            self.param_nets[name] = nn.Sequential(
                nn.Linear(param_head_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, param.numel())
            )

        self.target_network = target_network

    def forward(self, support_set: torch.Tensor, support_labels: torch.Tensor):

        support_set, support_labels = sorted_by_label(support_set, support_labels)

        bs, c, h, w = support_set.shape
        bl = support_labels.shape[0]
        
        assert bs == bl == self.n * self.k, (bs, bl, self.n, self.k)
        # support_set_concat = append_onehot_channels(support_set, support_labels, self.n)
        emb = self.fe(support_set)


        onehots = nnf.one_hot(support_labels, num_classes=self.n).float()

        # emb = torch.cat([emb, onehots], dim=1)

        emb = emb.reshape(-1).unsqueeze(0)
        #
        # predict network params
        network_params = {
            name.replace("-", "."): param_net(emb).reshape(self.param_shapes[name])
            for name, param_net in self.param_nets.items()
        }
        #
        tn = deepcopy(self.target_network)
        set_from_param_dict(tn, network_params)

        return nn.Sequential(self.fe, tn)


def append_onehot_channels(images: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """

    Args:
        images: [b, c, h, w]
        labels: [b]

    Returns:
        images with one-hot channels: [b, c+n_classes, h, w]
    """

    b, c, h, w = images.shape
    onehots = nnf.one_hot(labels, num_classes=n_classes).float()
    onehot_channels = onehots.reshape(b, n_classes, 1, 1).repeat(1, 1, h, w).to(images.device) * 100

    images_concat =  torch.cat([onehot_channels, images], dim=1)
    return images_concat
