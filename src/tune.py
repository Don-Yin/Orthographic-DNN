import json
import os
from pathlib import Path

import torch
import torch.nn.functional as functional
import torchvision
from torchvision import models

from modules.train import Train


class SelectedNetworks:
    def __init__(self):
        self.read_model_list()
        self.init_models()
        self.load_dict()

    def init_models(self):
        self.alexnet = torchvision.models.alexnet(weights=models.AlexNet_Weights.DEFAULT, progress=True)
        self.densenet169 = torchvision.models.densenet169(weights=models.DenseNet169_Weights.DEFAULT, progress=True)
        self.efficientnet_b1 = torchvision.models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT, progress=True)
        self.resnet50 = torchvision.models.resnet50(weights=models.ResNet50_Weights.DEFAULT, progress=True)
        self.resnet101 = torchvision.models.resnet101(weights=models.ResNet101_Weights.DEFAULT, progress=True)
        self.vgg16 = torchvision.models.vgg16(weights=models.VGG16_Weights.DEFAULT, progress=True)
        self.vgg19 = torchvision.models.vgg19(weights=models.VGG19_Weights.DEFAULT, progress=True)
        self.vit_b_16 = torchvision.models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT, progress=True)
        self.vit_b_32 = torchvision.models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT, progress=True)
        self.vit_l_16 = torchvision.models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT, progress=True)
        self.vit_l_32 = torchvision.models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT, progress=True)

    def read_model_list(self):
        self.selected_network_names = json.load(open(Path("assets", "dnn_model_labels.json"), "r")).keys()
        self.selected_network_names = list(self.selected_network_names)

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
        "rate_learning": (1e-5),
        "size_batch": 32,
        "num_epochs": 3,
        "weight_decay": 0.0001,  # L2 regularization
        "min_moving_average_threshold": 0.2,
        "batch_report_every": 8,
        "batch_save_every": 128,
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    print("Cuda available: ", torch.cuda.is_available())
    for i in range(11):
        Train(get_hyperparameters(i)).fit()
