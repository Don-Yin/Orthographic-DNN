import json
import logging
import os
import random
import sys
from itertools import product
from pathlib import Path

import psutil
import torch
import torchvision
from torchvision import models
from torchvision.datasets import ImageFolder as torch_image_folder
from tqdm import tqdm
from utils.data_generate.read_corpus import read_corpus
from utils.data_load.device_control import device_allocator
from utils.data_load.normalize import add_compute_stats
from utils.evaluate.evaluate_model import Evaluate


def restart_program():
    """
    Restarts the current program, with file objects and descriptors cleanup
    Note that this is extremely hacky and only used for when the RAMs are 
    insufficient.
    """
    try:
        p = psutil.Process(os.getpid())
        for handler in p.open_files() + p.connections():
            os.close(handler.fd)
    except Exception as e:
        logging.error(e)

    python = sys.executable
    os.execl(python, python, *sys.argv)


class BatchEvaluate:
    """
    returns: dataframe -
    cols:
        model, word,
        prime_type_1 (ID), prime_type_2,
        cosine_similarity (penultimate),
        cosine_similarity (classification)
    """

    def __init__(self):
        self.analysis_settings = json.load(open(Path("analysis_settings.json"), "r"))
        self.path_output = Path("assets", "model_output", "random_strings.json")
        self.validate = False
        self.use_checkpoint = False
        if self.validate:
            self.load_validation_data()
        self.read_prime_data()
        self.read_words_and_primes()
        self.read_selected_models()
        self.make_combinations()
        self.load_models()
        self.main()
        self.save()

    def read_words_and_primes(self):
        self.targets_2014 = read_corpus(Path("assets", "2014-targets.txt"))
        self.prime_types_2014 = read_corpus(Path("assets", "2014-prime-types.txt"))

    def read_selected_models(self):
        self.selected_network_names = json.load(open(Path("assets", "dnn_model_labels.json"), "r")).keys()

    def read_prime_data(self):
        try:
            normalize_stats = json.load(open(Path("data", "normalization_stats.json"), "r"))
        except FileNotFoundError:
            normalize_stats = None

        self.prime_data = add_compute_stats(torch_image_folder)(
            root=str(Path("data") / self.analysis_settings["prime_data_folder"]),
            stats=normalize_stats,
        )

    def make_combinations(self):
        if not os.path.exists(self.path_output):
            self.combinations = list(
                product(*[self.selected_network_names, self.targets_2014, ["ID"], self.prime_types_2014])
            )
            self.combinations = [{"model": c[0], "word": c[1], "primes": (c[2], c[3])} for c in self.combinations]
            self.save()
        else:
            self.combinations = json.load(open(self.path_output, "r"))

    def load_validation_data(self):
        normalization_stats = json.load(open(Path("data", "normalization_stats.json"), "r"))
        self.data_valid = add_compute_stats(torch_image_folder)(
            root=str(Path("data") / "data_valid"), stats=normalization_stats
        )

    def main(self):
        """
        Add similarity stats to the main dictionary.
        """
        try:
            for i in tqdm(range(len(self.combinations))):
                if "cosine_similarities" not in self.combinations[i].keys():
                    self.combinations[i]["cosine_similarities"] = Evaluate(
                        model=getattr(self, self.combinations[i]["model"]),
                        word=self.combinations[i]["word"],
                        prime_types=self.combinations[i]["primes"],
                        which_layer="all",
                        prime_data=self.prime_data,
                    ).compute_similarity_layer_wise()

                    if i % 128 == 0:
                        self.save()
                        print(f"\nSaved: {i}")

                    if i % 1024 == 0:
                        restart_program()

        except RuntimeError:
            restart_program()

    def save(self):
        json.dump(self.combinations, open(self.path_output, "w"))

    def validate_model_prediction(self, model):
        correct = 0
        for i in random.sample(range(len(self.data_valid)), 1000):
            output = model(self.data_valid[i][0].unsqueeze(0).cuda())
            _, prediction = torch.max(output, dim=1)
            if int(prediction) == self.data_valid[i][1]:
                correct += 1
        if correct < 950:
            print(f"Accuracy: {correct/10}%")
            raise Exception(f"{model} accuracy is lower than 95%")
        else:
            print(f"Accuracy: {correct/10}%")

    def load_models(self):
        self.alexnet = torchvision.models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.densenet169 = torchvision.models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        self.efficientnet_b1 = torchvision.models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        self.resnet50 = torchvision.models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet101 = torchvision.models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.vgg16 = torchvision.models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.vgg19 = torchvision.models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.vit_b_16 = torchvision.models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.vit_b_32 = torchvision.models.vit_b_32(weights=models.ViT_B_32_Weights.DEFAULT)
        self.vit_l_16 = torchvision.models.vit_l_16(weights=models.ViT_L_16_Weights.DEFAULT)
        self.vit_l_32 = torchvision.models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT)

        if self.use_checkpoint:
            for model in self.selected_network_names:
                getattr(self, model).load_state_dict(torch.load(Path("params", f"{model}.pth")))
                getattr(self, model).eval()
                device_allocator(getattr(self, model))
                if self.validate:
                    self.validate_model_prediction(getattr(self, model))
                print(f"State dict loaded: {model}")
        else:
            for model in self.selected_network_names:
                getattr(self, model).eval()
                device_allocator(getattr(self, model))
                if self.validate:
                    self.validate_model_prediction(getattr(self, model))


if __name__ == "__main__":
    BatchEvaluate()
