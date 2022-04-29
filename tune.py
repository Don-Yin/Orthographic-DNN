import json
import os
from pathlib import Path

import torch
import torch.nn.functional as functional
import torchvision

from train import Train

if __name__ == "__main__":

    # 2022-03-15 13:43:14 - brain scores
    # http://www.brain-score.org/

    cuda_available = torch.cuda.is_available()

    print("Cuda available: ", cuda_available)

    class SelectedNetworks:
        def __init__(self):
            self.read_model_list()
            self.init_models()
            self.load_dict()

        def init_models(self):
            self.alexnet = torchvision.models.alexnet(pretrained=True, progress=True)
            self.densenet169 = torchvision.models.densenet169(pretrained=True, progress=True)
            self.efficientnet_b1 = torchvision.models.efficientnet_b1(pretrained=True, progress=True)
            self.resnet50 = torchvision.models.resnet50(pretrained=True, progress=True)
            self.resnet101 = torchvision.models.resnet101(pretrained=True, progress=True)
            self.vgg16 = torchvision.models.vgg16(pretrained=True, progress=True)
            self.vgg19 = torchvision.models.vgg19(pretrained=True, progress=True)
            self.vit_l_16 = torchvision.models.vit_l_16(pretrained=True, progress=True)

        def read_model_list(self):
            self.selected_network_names = json.load(open(Path("assets", "selected_models.json"), "r"))

        def load_dict(self):
            for i in self.selected_network_names:
                if os.path.exists(Path("params", f"{i}.pth")):
                    getattr(self, i).load_state_dict(torch.load(Path("params", f"{i}.pth")))
                    print(f"Params loaded: {i}")

    selected_networks = SelectedNetworks()

    def get_hyperparameters(which_model: int):
        return {
            "model": getattr(selected_networks, selected_networks.selected_network_names[which_model]),
            "model_tag": selected_networks.selected_network_names[which_model],
            "ratio_train_test": [0.8, 0.2],
            "function_optimizer": torch.optim.Adam,
            "function_loss": functional.cross_entropy,
            "rate_learning": (1e-5) / 3,
            "size_batch": 32,
            "num_epochs": 1,
            "weight_decay": 0.0001,
            "min_moving_average_threshold": 0.01,
            "batch_report_every": 32,
            "batch_save_every": 256,
            "cuda_available": cuda_available,
        }

    Train(get_hyperparameters(2)).fit()
