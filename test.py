import json
import logging
import os
import platform
import random
import sys
from itertools import product
from pathlib import Path

import psutil
import torch
import torchvision
from torchvision.datasets import ImageFolder as torch_image_folder
from tqdm import tqdm

from utils.data_generate.read_corpus import read_corpus
from utils.data_load.device_control import device_allocator
from utils.data_load.normalize import add_compute_stats


class Test:
    def __init__(self):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.read_words_and_primes()
        self.read_valid_data()
        self.load_vit()
        for i in range(100):
            self.index = i
            self.get_image_tensor()
            self.main()

    def read_words_and_primes(self):
        self.targets_2014 = read_corpus(Path("assets", "2014-targets.txt"))
        self.prime_types_2014 = read_corpus(Path("assets", "2014-prime-types.txt"))

    def read_valid_data(self):
        self.valid_data = add_compute_stats(torch_image_folder)(
            root=str(Path("data") / "data_valid"),
            stats=json.load(open(Path("data", "normalization_stats.json"), "r")),
        )

    def load_vit(self):
        self.vit_l_16 = torchvision.models.vit_l_16(pretrained=False)
        self.vit_l_16.load_state_dict(torch.load(Path("params", "vit_l_16.pth")))
        self.vit_l_16.eval()
        device_allocator(self.vit_l_16)

    def get_image_tensor(self):
        self.tensor = self.valid_data[self.index][0].unsqueeze(0).cuda().detach()

    def main(self):
        output = self.vit_l_16(self.tensor)
        print("label", self.valid_data.imgs[self.index][1])
        print("prediction", torch.max(output, dim=1))
        print("\n")


if __name__ == "__main__":
    Test()
