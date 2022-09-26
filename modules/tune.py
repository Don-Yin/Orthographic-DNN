import json
import os
from pathlib import Path

import torch
import torch.nn.functional as functional
import torchvision

from train import Train


class SelectedNetworks:
    def __init__(self):
        self.read_model_list()
        self.init_models()
        self.load_dict()

    def init_models(self):
        # these models typically have 1000 classes so the default value is used
        # but it is possible to define the value
        self.alexnet = torchvision.models.alexnet(pretrained=True, progress=True)  # 4096
        self.densenet169 = torchvision.models.densenet169(pretrained=True, progress=True)  # 1664
        self.efficientnet_b1 = torchvision.models.efficientnet_b1(pretrained=True, progress=True)  # 1280
        self.resnet50 = torchvision.models.resnet50(pretrained=True, progress=True)  # 2048
        self.resnet101 = torchvision.models.resnet101(pretrained=True, progress=True)  # 2048
        self.vgg16 = torchvision.models.vgg16(pretrained=True, progress=True)  # 4096
        self.vgg19 = torchvision.models.vgg19(pretrained=True, progress=True)  # 4096
        self.vit_b_16 = torchvision.models.vit_b_16(pretrained=True, progress=True)
        self.vit_b_32 = torchvision.models.vit_b_32(pretrained=True, progress=True)
        self.vit_l_16 = torchvision.models.vit_l_16(pretrained=True, progress=True)
        self.vit_l_32 = torchvision.models.vit_l_32(pretrained=True, progress=True)

    def read_model_list(self):
        self.selected_network_names = json.load(open(Path("assets", "dnn_model_labels.json"), "r")).keys()

    def load_dict(self):
        for i in self.selected_network_names:
            if os.path.exists(Path("params", f"{i}.pth")):
                getattr(self, i).load_state_dict(torch.load(Path("params", f"{i}.pth")))
                print(f"Params loaded: {i}")


def get_hyperparameters(which_model: int):
    selected_networks = SelectedNetworks()
    return {
        "model": getattr(selected_networks, selected_networks.selected_network_names[which_model]),
        "model_tag": selected_networks.selected_network_names[which_model],
        "ratio_train_test": [0.8, 0.2],
        "function_optimizer": torch.optim.Adam,
        "function_loss": functional.cross_entropy,
        "rate_learning": (1e-5) / 3,
        "size_batch": 4,
        "num_epochs": 1,
        "weight_decay": 0.0001,
        "min_moving_average_threshold": 0.0025,
        "batch_report_every": 8,
        "batch_save_every": 128,
        "cuda_available": cuda_available,
    }


if __name__ == "__main__":

    # 2022-03-15 13:43:14 - brain scores
    # http://www.brain-score.org/

    cuda_available = torch.cuda.is_available()

    print("Cuda available: ", cuda_available)

    Train(get_hyperparameters(10)).fit()
